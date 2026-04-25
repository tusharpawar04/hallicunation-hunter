"""Evaluation script for comparing baseline and trained models."""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client.env_client import HallucinationHunterEnv
from src.api.models import DetectionOutput, DetectedClaim


def evaluate_model(
    env: HallucinationHunterEnv,
    num_episodes: int = 100
) -> Dict[str, Any]:
    """Evaluate a model on test episodes.
    
    Args:
        env: Environment client
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of aggregate metrics
    """
    total_reward = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    episode_results = []
    
    print(f"Evaluating on {num_episodes} episodes...")
    
    for i in range(num_episodes):
        # Reset environment
        observation, info = env.reset()
        
        # TODO: Generate detection output using model
        # For now, this is a placeholder showing the structure
        # In actual implementation, you would:
        # 1. Pass observation["generated_text"] to your model
        # 2. Parse model output into DetectionOutput format
        # 3. Submit to environment
        
        # Placeholder detection (replace with actual model inference)
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="Placeholder claim",
                label="factual",
                reason="Placeholder reason",
                corrected_fact=None
            )
        ])
        
        # Get reward
        result = env.step(detection)
        
        # Accumulate metrics
        total_reward += result["reward"]
        total_precision += result["info"]["precision"]
        total_recall += result["info"]["recall"]
        total_f1 += result["info"]["f1"]
        
        episode_results.append({
            "episode_id": info["episode_id"],
            "difficulty": info["difficulty_level"],
            "reward": result["reward"],
            "precision": result["info"]["precision"],
            "recall": result["info"]["recall"],
            "f1": result["info"]["f1"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_episodes} episodes")
    
    # Calculate averages
    avg_reward = total_reward / num_episodes
    avg_precision = total_precision / num_episodes
    avg_recall = total_recall / num_episodes
    avg_f1 = total_f1 / num_episodes
    
    return {
        "num_episodes": num_episodes,
        "avg_reward": avg_reward,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "episode_results": episode_results
    }


def compare_models(
    baseline_results: Dict[str, Any],
    trained_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare baseline and trained model results.
    
    Args:
        baseline_results: Results from baseline model
        trained_results: Results from trained model
        
    Returns:
        Comparison metrics
    """
    improvement = {
        "reward_improvement": trained_results["avg_reward"] - baseline_results["avg_reward"],
        "reward_improvement_pct": (
            (trained_results["avg_reward"] - baseline_results["avg_reward"]) /
            abs(baseline_results["avg_reward"]) * 100
            if baseline_results["avg_reward"] != 0 else 0
        ),
        "precision_improvement": trained_results["avg_precision"] - baseline_results["avg_precision"],
        "recall_improvement": trained_results["avg_recall"] - baseline_results["avg_recall"],
        "f1_improvement": trained_results["avg_f1"] - baseline_results["avg_f1"]
    }
    
    return {
        "baseline": baseline_results,
        "trained": trained_results,
        "improvement": improvement
    }


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Hallucination Hunter Evaluation Script")
    print("=" * 60)
    
    # Configuration
    base_url = "http://localhost:7860"
    num_test_episodes = 100
    
    # Initialize environment
    print("\nInitializing environment...")
    env = HallucinationHunterEnv(base_url=base_url)
    
    # Check server health
    try:
        health = env.health()
        print(f"  Server status: {health['status']}")
        print(f"  Episodes available: {health['episode_count']}")
    except Exception as e:
        print(f"  Error connecting to server: {e}")
        print("  Make sure the server is running: python app.py")
        return
    
    print("\n" + "=" * 60)
    print("Evaluation Guide")
    print("=" * 60)
    
    print("""
To evaluate models, you would:

1. **Evaluate Baseline Model** (untrained):
   ```python
   baseline_results = evaluate_model(env, num_episodes=100)
   ```

2. **Evaluate Trained Model**:
   ```python
   # Load trained model
   model = load_model("./final_model")
   
   # Evaluate
   trained_results = evaluate_model(env, num_episodes=100)
   ```

3. **Compare Results**:
   ```python
   comparison = compare_models(baseline_results, trained_results)
   
   print(f"Reward improvement: {comparison['improvement']['reward_improvement']:.2f}")
   print(f"Precision improvement: {comparison['improvement']['precision_improvement']:.2f}")
   print(f"Recall improvement: {comparison['improvement']['recall_improvement']:.2f}")
   ```

4. **Save Results**:
   ```python
   with open("evaluation_results.json", "w") as f:
       json.dump(comparison, f, indent=2)
   ```

Expected Results:
- Baseline reward: ~0.5 (random guessing)
- Trained reward: >3.0 (6x improvement)
- Baseline precision/recall: ~0.3-0.4
- Trained precision/recall: >0.7
""")
    
    print("=" * 60)
    print("To run actual evaluation:")
    print("1. Train a model: python scripts/train_agent.py")
    print("2. Implement model inference in this script")
    print("3. Run evaluation: python scripts/evaluate.py")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
