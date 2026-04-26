#!/usr/bin/env python3
"""Quick test of environment locally"""

import sys
sys.path.insert(0, '.')

from src.environment.core import HallucinationEnvironment
from src.environment.episode_bank import EpisodeBank
from src.environment.curriculum import CurriculumManager
from src.environment.reward import RewardEngine

print("Initializing environment...")

# Initialize components
episode_bank = EpisodeBank()
episode_bank.load_episodes("data/episodes")

curriculum_manager = CurriculumManager(
    promotion_thresholds={"L1": 3.5, "L2": 4.0, "L3": 5.0},
    window_size=50
)

reward_engine = RewardEngine()

env = HallucinationEnvironment(
    episode_bank=episode_bank,
    curriculum_manager=curriculum_manager,
    reward_engine=reward_engine
)

print(f"✅ Environment initialized")
print(f"Episodes loaded: {len(episode_bank.episodes)}")

# Test reset
print("\nTesting reset...")
obs, info = env.reset()
print(f"✅ Reset successful")
print(f"Episode: {info['episode_id']}")
print(f"Difficulty: {info['difficulty_level']}")
print(f"Text: {obs['generated_text'][:100]}...")

# Test step with proper format
print("\nTesting step...")
action = {
    "detected_claims": [
        {
            "claim_text": "Test claim",
            "label": "factual",
            "reason": "Testing",
            "corrected_fact": None
        }
    ]
}

try:
    obs, reward, done, info = env.step(action)
    print(f"✅ Step successful")
    print(f"Reward: {reward:.4f}")
    print(f"Precision: {info.get('precision', 0):.4f}")
    print(f"Recall: {info.get('recall', 0):.4f}")
    print(f"Done: {done}")
except Exception as e:
    print(f"❌ Step failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Local environment test complete!")
