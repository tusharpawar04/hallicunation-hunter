"""Core data models and environment implementation for Hallucination Hunter."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from openenv.env import Env as OpenEnvBase
    OPENENV_AVAILABLE = True
except ImportError:
    # Fallback if OpenEnv not installed
    OpenEnvBase = object
    OPENENV_AVAILABLE = False

if TYPE_CHECKING:
    from .episode_bank import EpisodeBank
    from .curriculum import CurriculumManager
    from .reward import RewardEngine
    from ..api.models import DetectionOutput, Observation, StepResult


@dataclass
class Claim:
    """Represents a single factual claim extracted from generated text.
    
    Attributes:
        claim_text: The text of the claim
        label: Classification of the claim - must be one of:
               "factual", "hallucinated", "unverifiable"
        ground_truth_fact: The correct fact for hallucinated claims, None otherwise
    """
    claim_text: str
    label: str
    ground_truth_fact: Optional[str] = None
    
    def __post_init__(self):
        """Validate label values."""
        valid_labels = {"factual", "hallucinated", "unverifiable"}
        if self.label not in valid_labels:
            raise ValueError(
                f"Invalid label '{self.label}'. Must be one of {valid_labels}"
            )


@dataclass
class Episode:
    """Represents a single training episode with ground truth claims.
    
    Attributes:
        episode_id: Unique identifier for the episode
        source_dataset: Origin dataset - e.g., "halueval_qa", "truthfulqa", 
                       "wikipedia_synthetic"
        difficulty_level: Difficulty classification - must be one of:
                         "L1", "L2", "L3", "L4"
        source_text: Original context (e.g., Wikipedia paragraph)
        generated_response: LLM-generated text to analyze for hallucinations
        claims: List of ground truth claim decompositions
        metadata: Additional information (topic, model, claim_count, etc.)
    """
    episode_id: str
    source_dataset: str
    difficulty_level: str
    source_text: str
    generated_response: str
    claims: List[Claim]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate difficulty level."""
        valid_levels = {"L1", "L2", "L3", "L4"}
        if self.difficulty_level not in valid_levels:
            raise ValueError(
                f"Invalid difficulty_level '{self.difficulty_level}'. "
                f"Must be one of {valid_levels}"
            )



class HallucinationEnvironment(OpenEnvBase):
    """OpenEnv-compatible RL environment for hallucination detection training.
    
    This environment orchestrates the RL training loop by:
    - Sampling episodes from the EpisodeBank based on curriculum state
    - Providing observations with generated text and task instructions
    - Calculating rewards using the RewardEngine
    - Managing curriculum progression via the CurriculumManager
    - Implementing single-turn episodes (one detection task per episode)
    
    Attributes:
        episode_bank: Storage and sampling system for training episodes
        curriculum_manager: Manages difficulty progression based on performance
        reward_engine: Calculates deterministic rewards for detection outputs
        current_episode: The currently active episode (None if no episode in progress)
        step_count: Number of steps taken in current episode (0 or 1 for single-turn)
    """
    
    # Task instruction template with all required components
    TASK_INSTRUCTION = """Analyze the following generated text for hallucinations.

Your task:
1. Identify each factual claim in the text
2. Label each claim as one of: "factual", "hallucinated", or "unverifiable"
3. For hallucinated claims, provide the correct fact
4. Provide a reason for each label assignment

Output format: JSON with the following structure:
{
  "detected_claims": [
    {
      "claim_text": "The specific claim from the text",
      "label": "factual|hallucinated|unverifiable",
      "reason": "Explanation for this label",
      "corrected_fact": "The correct fact (only for hallucinated claims, null otherwise)"
    }
  ]
}

Required fields for each claim:
- claim_text: The text of the claim
- label: Must be "factual", "hallucinated", or "unverifiable"
- reason: Explanation for the label
- corrected_fact: Correction if hallucinated, null otherwise"""
    
    def __init__(
        self,
        episode_bank: 'EpisodeBank',
        curriculum_manager: 'CurriculumManager',
        reward_engine: 'RewardEngine'
    ):
        """Initialize the hallucination detection environment.
        
        Args:
            episode_bank: Episode storage and sampling system
            curriculum_manager: Difficulty progression manager
            reward_engine: Reward calculation engine
        """
        self.episode_bank = episode_bank
        self.curriculum_manager = curriculum_manager
        self.reward_engine = reward_engine
        
        # Episode state
        self.current_episode: Optional[Episode] = None
        self.step_count: int = 0
    
    def reset(self, return_info=True):
        """Initialize a new episode and return the initial observation (OpenEnv-compatible).
        
        Samples an episode from the currently enabled difficulty levels,
        constructs an observation with the generated text and task instructions,
        and optionally returns episode metadata.
        
        Args:
            return_info: If True, return (observation, info). If False, return only observation.
                        Default True for OpenEnv compatibility.
        
        Returns:
            If return_info is True:
                Tuple of (observation_dict, info_dict) where:
                - observation_dict contains:
                    - generated_text: The LLM-generated text to analyze
                    - task_instruction: Instructions for the detection task
                - info_dict contains:
                    - episode_id: Unique identifier for the episode
                    - difficulty_level: Episode difficulty (L1-L4)
                    - source_dataset: Origin dataset name
            If return_info is False:
                Just observation_dict
        
        Raises:
            ValueError: If no episodes are available at enabled difficulty levels
        """
        # Get currently enabled difficulty levels from curriculum
        enabled_levels = self.curriculum_manager.get_enabled_levels()
        
        # Sample episode from enabled levels
        self.current_episode = self.episode_bank.sample_episode(enabled_levels)
        self.step_count = 0
        
        # Construct observation
        observation = {
            "generated_text": self.current_episode.generated_response,
            "task_instruction": self.TASK_INSTRUCTION
        }
        
        if return_info:
            # Construct info with episode metadata
            info = {
                "episode_id": self.current_episode.episode_id,
                "difficulty_level": self.current_episode.difficulty_level,
                "source_dataset": self.current_episode.source_dataset
            }
            return observation, info
        else:
            return observation
    
    def step(self, action):
        """Process agent's action and return next state (OpenEnv-compatible).
        
        This method supports both OpenEnv format (action dict) and legacy format
        (DetectionOutput object) for backward compatibility.
        
        Args:
            action: Either a dict with 'detection_output' key (OpenEnv format)
                   or a DetectionOutput object (legacy format)
        
        Returns:
            Tuple of (observation, reward, done, info) where:
            - observation: Dict with generated_text and task_instruction (empty for done episodes)
            - reward: Float reward score
            - done: Boolean indicating episode completion (always True for single-turn)
            - info: Dict with episode metadata and metrics
        
        Raises:
            RuntimeError: If step() is called before reset() or after episode completion
        """
        if self.current_episode is None:
            raise RuntimeError(
                "Cannot call step() before reset(). Call reset() first to initialize an episode."
            )
        
        if self.step_count > 0:
            raise RuntimeError(
                "Episode already completed. Call reset() to start a new episode."
            )
        
        # Handle both OpenEnv format (dict with 'detection_output') and legacy format
        if isinstance(action, dict) and 'detection_output' in action:
            detection_output = action['detection_output']
        else:
            # Assume it's already a DetectionOutput object (legacy format)
            detection_output = action
        
        # Calculate reward using reward engine
        reward, metrics = self.reward_engine.calculate_reward(
            detection_output=detection_output,
            ground_truth_claims=self.current_episode.claims,
            difficulty_level=self.current_episode.difficulty_level
        )
        
        # Record reward for curriculum tracking
        self.curriculum_manager.record_reward(
            difficulty_level=self.current_episode.difficulty_level,
            reward=reward
        )
        
        # Check for curriculum promotion
        promoted_level = self.curriculum_manager.check_promotion()
        if promoted_level:
            # Add promotion info to metrics
            metrics["promoted_to"] = promoted_level
        
        # Increment step count
        self.step_count += 1
        
        # Construct empty observation (single-turn episode)
        observation = {
            "generated_text": "",
            "task_instruction": ""
        }
        
        # Construct info with episode metadata and metrics
        info = {
            "episode_id": self.current_episode.episode_id,
            "difficulty_level": self.current_episode.difficulty_level,
            "source_dataset": self.current_episode.source_dataset,
            **metrics  # Include all metrics from reward calculation
        }
        
        # Return OpenEnv-compatible tuple
        done = True  # Single-turn episodes always complete after first step
        
        return observation, reward, done, info
    
    def get_current_episode(self) -> Optional[Episode]:
        """Get the currently active episode.
        
        Returns:
            Current episode if one is active, None otherwise
        """
        return self.current_episode
    
    def get_curriculum_state(self) -> Dict[str, Any]:
        """Get current curriculum state for monitoring.
        
        Returns:
            Dictionary containing:
            - enabled_levels: List of currently enabled difficulty levels
            - rolling_avg_rewards: Dict mapping difficulty level to rolling average reward
        """
        enabled_levels = self.curriculum_manager.get_enabled_levels()
        
        rolling_avg_rewards = {
            level: self.curriculum_manager.get_rolling_avg(level)
            for level in ["L1", "L2", "L3", "L4"]
        }
        
        return {
            "enabled_levels": enabled_levels,
            "rolling_avg_rewards": rolling_avg_rewards
        }
