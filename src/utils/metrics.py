"""Metrics calculation, logging, and export utilities."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: str
    timestamp: float
    difficulty_level: str
    reward: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    correction_bonus: float
    calibration_bonus: float
    difficulty_multiplier: float
    gaming_penalty: float
    passivity_penalty: float


class MetricsLogger:
    """Logger for episode-level metrics and training statistics."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to save metrics logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_metrics: List[EpisodeMetrics] = []
        self.cumulative_reward: float = 0.0
        self.rewards_by_difficulty: Dict[str, List[float]] = {
            "L1": [],
            "L2": [],
            "L3": [],
            "L4": []
        }
    
    def log_episode(
        self,
        episode_id: str,
        difficulty_level: str,
        reward: float,
        metrics: Dict[str, Any]
    ) -> None:
        """Log metrics for a completed episode.
        
        Args:
            episode_id: Unique identifier for the episode
            difficulty_level: Episode difficulty (L1-L4)
            reward: Total reward received
            metrics: Dictionary of metrics from reward calculation
        """
        episode_metrics = EpisodeMetrics(
            episode_id=episode_id,
            timestamp=datetime.now().timestamp(),
            difficulty_level=difficulty_level,
            reward=reward,
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1_score=metrics.get("f1", 0.0),
            true_positives=metrics.get("true_positives", 0),
            false_positives=metrics.get("false_positives", 0),
            false_negatives=metrics.get("false_negatives", 0),
            true_negatives=metrics.get("true_negatives", 0),
            correction_bonus=metrics.get("correction_bonus", 0.0),
            calibration_bonus=metrics.get("calibration_bonus", 0.0),
            difficulty_multiplier=metrics.get("difficulty_multiplier", 1.0),
            gaming_penalty=metrics.get("gaming_penalty", 0.0),
            passivity_penalty=metrics.get("passivity_penalty", 0.0)
        )
        
        self.episode_metrics.append(episode_metrics)
        self.cumulative_reward += reward
        self.rewards_by_difficulty[difficulty_level].append(reward)
    
    def get_cumulative_reward(self) -> float:
        """Get cumulative reward over all episodes.
        
        Returns:
            Total cumulative reward
        """
        return self.cumulative_reward
    
    def get_rewards_by_difficulty(self, difficulty_level: str) -> List[float]:
        """Get reward history for a specific difficulty level.
        
        Args:
            difficulty_level: Difficulty level (L1-L4)
            
        Returns:
            List of rewards for that difficulty level
        """
        return self.rewards_by_difficulty.get(difficulty_level, [])
    
    def calculate_rolling_average(
        self,
        values: List[float],
        window_size: int = 50
    ) -> List[float]:
        """Calculate rolling average over a window.
        
        Args:
            values: List of values
            window_size: Size of rolling window
            
        Returns:
            List of rolling averages
        """
        if not values:
            return []
        
        rolling_avgs = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window = values[start_idx:i + 1]
            rolling_avgs.append(sum(window) / len(window))
        
        return rolling_avgs
    
    def export_time_series(self, output_path: str) -> None:
        """Export time series data for visualization.
        
        Args:
            output_path: Path to save JSON file
        """
        time_series = {
            "rewards_by_difficulty": {
                level: self.rewards_by_difficulty[level]
                for level in ["L1", "L2", "L3", "L4"]
            },
            "precision_over_time": [m.precision for m in self.episode_metrics],
            "recall_over_time": [m.recall for m in self.episode_metrics],
            "cumulative_reward": [
                sum(m.reward for m in self.episode_metrics[:i+1])
                for i in range(len(self.episode_metrics))
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(time_series, f, indent=2)
    
    def export_metrics(self, output_path: str) -> None:
        """Export all episode metrics to JSON.
        
        Args:
            output_path: Path to save JSON file
        """
        metrics_data = [asdict(m) for m in self.episode_metrics]
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all episodes.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.episode_metrics:
            return {}
        
        return {
            "total_episodes": len(self.episode_metrics),
            "cumulative_reward": self.cumulative_reward,
            "average_reward": self.cumulative_reward / len(self.episode_metrics),
            "average_precision": sum(m.precision for m in self.episode_metrics) / len(self.episode_metrics),
            "average_recall": sum(m.recall for m in self.episode_metrics) / len(self.episode_metrics),
            "average_f1": sum(m.f1_score for m in self.episode_metrics) / len(self.episode_metrics),
            "episodes_by_difficulty": {
                level: len(rewards)
                for level, rewards in self.rewards_by_difficulty.items()
            }
        }


def calculate_precision(true_positives: int, false_positives: int) -> float:
    """Calculate precision: TP / (TP + FP).
    
    Args:
        true_positives: Number of true positives
        false_positives: Number of false positives
        
    Returns:
        Precision score (0.0 to 1.0)
    """
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def calculate_recall(true_positives: int, false_negatives: int) -> float:
    """Calculate recall: TP / (TP + FN).
    
    Args:
        true_positives: Number of true positives
        false_negatives: Number of false negatives
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score: 2 * precision * recall / (precision + recall).
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
