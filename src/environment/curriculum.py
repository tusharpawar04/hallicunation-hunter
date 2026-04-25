"""Curriculum management for progressive difficulty scaling."""

from collections import deque
from typing import Dict, List, Optional


class CurriculumManager:
    """Manages difficulty progression based on rolling average rewards.
    
    The curriculum starts with only L1 enabled and progressively unlocks
    higher difficulty levels (L2, L3, L4) when performance thresholds are met.
    
    Promotion decisions are based on rolling average rewards over a fixed
    window size (default 50 episodes) to ensure stable performance before
    advancing to harder episodes.
    """
    
    def __init__(
        self,
        promotion_thresholds: Dict[str, float],
        window_size: int = 50
    ):
        """Initialize curriculum manager.
        
        Args:
            promotion_thresholds: Dict mapping difficulty levels to reward thresholds.
                Example: {"L1": 3.5, "L2": 4.0, "L3": 5.0}
                When L1 rolling avg exceeds 3.5, L2 is enabled.
                When L2 rolling avg exceeds 4.0, L3 is enabled.
                When L3 rolling avg exceeds 5.0, L4 is enabled.
            window_size: Number of recent episodes for rolling average calculation.
                Default is 50 episodes.
        """
        self.promotion_thresholds = promotion_thresholds
        self.window_size = window_size
        
        # Track rewards separately per difficulty level using deques for efficient rolling windows
        self._reward_history: Dict[str, deque] = {
            "L1": deque(maxlen=window_size),
            "L2": deque(maxlen=window_size),
            "L3": deque(maxlen=window_size),
            "L4": deque(maxlen=window_size)
        }
        
        # Track which levels are currently enabled
        # Start with only L1 enabled (Requirement 2.1)
        self._enabled_levels: List[str] = ["L1"]
    
    def record_reward(self, difficulty_level: str, reward: float) -> None:
        """Record episode reward for curriculum tracking.
        
        Rewards are tracked separately per difficulty level to enable
        independent promotion decisions.
        
        Args:
            difficulty_level: The difficulty level of the completed episode (L1-L4)
            reward: The total reward received for the episode
        """
        if difficulty_level not in self._reward_history:
            raise ValueError(
                f"Invalid difficulty_level '{difficulty_level}'. "
                f"Must be one of {list(self._reward_history.keys())}"
            )
        
        self._reward_history[difficulty_level].append(reward)
    
    def get_enabled_levels(self) -> List[str]:
        """Return currently enabled difficulty levels.
        
        Returns:
            List of enabled difficulty level strings (e.g., ["L1", "L2"])
        """
        return self._enabled_levels.copy()
    
    def check_promotion(self) -> Optional[str]:
        """Check if any level should be promoted and enable next level.
        
        Checks each enabled level's rolling average against its promotion threshold.
        If the threshold is exceeded, the next difficulty level is enabled.
        
        Returns:
            The newly enabled difficulty level (e.g., "L2"), or None if no promotion occurred
        """
        # Define level progression order
        level_progression = {
            "L1": "L2",
            "L2": "L3",
            "L3": "L4"
        }
        
        # Check each currently enabled level for promotion
        for current_level in self._enabled_levels[:]:  # Use slice to avoid modification during iteration
            # Skip if this level has no next level
            if current_level not in level_progression:
                continue
            
            next_level = level_progression[current_level]
            
            # Skip if next level is already enabled
            if next_level in self._enabled_levels:
                continue
            
            # Skip if this level has no promotion threshold defined
            if current_level not in self.promotion_thresholds:
                continue
            
            # Check if rolling average exceeds threshold
            rolling_avg = self.get_rolling_avg(current_level)
            threshold = self.promotion_thresholds[current_level]
            
            if rolling_avg > threshold:
                # Enable next level
                self._enabled_levels.append(next_level)
                return next_level
        
        return None
    
    def get_rolling_avg(self, difficulty_level: str) -> float:
        """Get rolling average reward for a difficulty level.
        
        Calculates the mean of the most recent rewards (up to window_size)
        for the specified difficulty level.
        
        Args:
            difficulty_level: The difficulty level to query (L1-L4)
            
        Returns:
            Rolling average reward, or 0.0 if no rewards have been recorded
        """
        if difficulty_level not in self._reward_history:
            raise ValueError(
                f"Invalid difficulty_level '{difficulty_level}'. "
                f"Must be one of {list(self._reward_history.keys())}"
            )
        
        rewards = self._reward_history[difficulty_level]
        
        if not rewards:
            return 0.0
        
        return sum(rewards) / len(rewards)
