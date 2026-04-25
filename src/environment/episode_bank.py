"""Episode bank for loading, storing, and sampling training episodes."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import Claim, Episode


class EpisodeBank:
    """Manages episode storage, difficulty assignment, and sampling.
    
    The EpisodeBank loads episodes from JSON files, assigns difficulty levels
    based on heuristics, and provides sampling with difficulty filtering for
    curriculum learning.
    
    Attributes:
        episodes: Dictionary mapping episode_id to Episode objects
        episodes_by_difficulty: Dictionary mapping difficulty level to list of episode_ids
    """
    
    def __init__(self):
        """Initialize an empty episode bank."""
        self.episodes: Dict[str, Episode] = {}
        self.episodes_by_difficulty: Dict[str, List[str]] = {
            "L1": [],
            "L2": [],
            "L3": [],
            "L4": []
        }
    
    def load_episodes(self, data_dir: str) -> None:
        """Load episodes from JSON files in data directory.
        
        Loads all JSON files from the specified directory and its subdirectories.
        Each JSON file should contain a single episode with all required fields.
        Episodes are validated and difficulty levels are assigned if not present.
        
        Args:
            data_dir: Path to directory containing episode JSON files.
                     Expected structure: data_dir/{source_dataset}/*.json
        
        Raises:
            FileNotFoundError: If data_dir does not exist
            ValueError: If episode files are malformed or missing required fields
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find all JSON files recursively
        json_files = list(data_path.rglob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {data_dir}")
        
        loaded_count = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
                
                # Validate required fields
                required_fields = [
                    "episode_id", "source_dataset", "source_text",
                    "generated_response", "claims", "metadata"
                ]
                missing_fields = [
                    field for field in required_fields 
                    if field not in episode_data
                ]
                if missing_fields:
                    raise ValueError(
                        f"Episode {json_file} missing required fields: {missing_fields}"
                    )
                
                # Parse claims
                claims = []
                for claim_data in episode_data["claims"]:
                    claim = Claim(
                        claim_text=claim_data["claim_text"],
                        label=claim_data["label"],
                        ground_truth_fact=claim_data.get("ground_truth_fact")
                    )
                    claims.append(claim)
                
                # Determine difficulty level
                difficulty_level = episode_data.get("difficulty_level", "")
                
                # Create temporary episode for difficulty assignment if needed
                if not difficulty_level or difficulty_level not in {"L1", "L2", "L3", "L4"}:
                    # Create temporary episode with L1 to pass validation
                    temp_episode = Episode(
                        episode_id=episode_data["episode_id"],
                        source_dataset=episode_data["source_dataset"],
                        difficulty_level="L1",  # Temporary value
                        source_text=episode_data["source_text"],
                        generated_response=episode_data["generated_response"],
                        claims=claims,
                        metadata=episode_data["metadata"]
                    )
                    difficulty_level = self._assign_difficulty(temp_episode)
                
                # Create final episode object with correct difficulty
                episode = Episode(
                    episode_id=episode_data["episode_id"],
                    source_dataset=episode_data["source_dataset"],
                    difficulty_level=difficulty_level,
                    source_text=episode_data["source_text"],
                    generated_response=episode_data["generated_response"],
                    claims=claims,
                    metadata=episode_data["metadata"]
                )
                
                # Store episode
                self.episodes[episode.episode_id] = episode
                self.episodes_by_difficulty[episode.difficulty_level].append(
                    episode.episode_id
                )
                loaded_count += 1
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Log error but continue loading other episodes
                print(f"Warning: Failed to load episode from {json_file}: {e}")
                continue
        
        if loaded_count == 0:
            raise ValueError(f"No valid episodes loaded from {data_dir}")
        
        print(f"Loaded {loaded_count} episodes from {data_dir}")
    
    def sample_episode(self, difficulty_levels: List[str]) -> Episode:
        """Sample random episode from enabled difficulty levels.
        
        Args:
            difficulty_levels: List of enabled difficulty levels to sample from.
                              Must be subset of ["L1", "L2", "L3", "L4"].
        
        Returns:
            Randomly sampled Episode from the specified difficulty levels.
        
        Raises:
            ValueError: If no episodes available at specified difficulty levels
            ValueError: If difficulty_levels contains invalid levels
        """
        # Validate difficulty levels
        valid_levels = {"L1", "L2", "L3", "L4"}
        invalid_levels = set(difficulty_levels) - valid_levels
        if invalid_levels:
            raise ValueError(
                f"Invalid difficulty levels: {invalid_levels}. "
                f"Must be subset of {valid_levels}"
            )
        
        # Collect all episode IDs from enabled difficulty levels
        available_episode_ids = []
        for level in difficulty_levels:
            available_episode_ids.extend(self.episodes_by_difficulty[level])
        
        if not available_episode_ids:
            raise ValueError(
                f"No episodes available at difficulty levels: {difficulty_levels}"
            )
        
        # Sample random episode
        episode_id = random.choice(available_episode_ids)
        return self.episodes[episode_id]
    
    def get_episode_by_id(self, episode_id: str) -> Episode:
        """Retrieve specific episode by ID.
        
        Args:
            episode_id: Unique identifier of the episode to retrieve.
        
        Returns:
            Episode with the specified ID.
        
        Raises:
            KeyError: If episode_id not found in the bank
        """
        if episode_id not in self.episodes:
            raise KeyError(f"Episode not found: {episode_id}")
        return self.episodes[episode_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return episode count and difficulty distribution.
        
        Returns:
            Dictionary containing:
                - episode_count: Total number of episodes
                - difficulty_distribution: Count of episodes per difficulty level
        """
        return {
            "episode_count": len(self.episodes),
            "difficulty_distribution": {
                level: len(episode_ids)
                for level, episode_ids in self.episodes_by_difficulty.items()
            }
        }
    
    def _assign_difficulty(self, episode: Episode) -> str:
        """Heuristic to assign L1-L4 based on claim count and complexity.
        
        Difficulty assignment heuristics:
        - L1 (Simple): 2-4 claims, single-domain, clear distinction, low ambiguity
        - L2 (Moderate): 4-6 claims, cross-domain, some inference, moderate ambiguity
        - L3 (Hard): 6-8 claims, complex reasoning, subtle hallucinations, higher ambiguity
        - L4 (Expert): 8+ claims, specialized knowledge, mixed claims, deep context
        
        Args:
            episode: Episode to assign difficulty level to.
        
        Returns:
            Difficulty level string: "L1", "L2", "L3", or "L4"
        """
        claim_count = len(episode.claims)
        
        # Count hallucinated and unverifiable claims (indicators of complexity)
        hallucinated_count = sum(
            1 for claim in episode.claims if claim.label == "hallucinated"
        )
        unverifiable_count = sum(
            1 for claim in episode.claims if claim.label == "unverifiable"
        )
        
        # Calculate hallucination rate
        hallucination_rate = (
            hallucinated_count / claim_count if claim_count > 0 else 0
        )
        
        # Calculate ambiguity score (presence of unverifiable claims)
        ambiguity_score = unverifiable_count / claim_count if claim_count > 0 else 0
        
        # Get metadata hints if available
        metadata_difficulty = episode.metadata.get("difficulty_level", "")
        if metadata_difficulty in {"L1", "L2", "L3", "L4"}:
            return metadata_difficulty
        
        # Heuristic-based assignment
        # L4: 8+ claims OR high ambiguity (>30% unverifiable)
        if claim_count >= 8 or ambiguity_score > 0.3:
            return "L4"
        
        # L3: 6-8 claims OR moderate ambiguity with subtle hallucinations
        if claim_count >= 6 or (ambiguity_score > 0.15 and 0.1 < hallucination_rate < 0.4):
            return "L3"
        
        # L2: 4-6 claims OR moderate hallucination rate
        if claim_count >= 4 or hallucination_rate > 0.3:
            return "L2"
        
        # L1: Default for simple cases (2-4 claims, clear patterns)
        return "L1"
