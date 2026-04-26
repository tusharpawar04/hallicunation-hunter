#!/usr/bin/env python3
"""Test GRPO training setup locally with a few steps"""

import sys
sys.path.insert(0, '.')

from src.environment.core import HallucinationEnvironment
from src.environment.episode_bank import EpisodeBank
from src.environment.curriculum import CurriculumManager
from src.environment.reward import RewardEngine

print("=" * 80)
print("TESTING GRPO TRAINING SETUP")
print("=" * 80)

# Initialize environment
print("\n1. Initializing environment...")
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

print(f"✅ Environment ready with {len(episode_bank.episodes)} episodes")

# Test multiple episodes
print("\n2. Testing reward calculation on 5 episodes...")
rewards = []

for i in range(5):
    # Reset
    obs, info = env.reset()
    
    # Create a simple detection (will get low reward, but tests the flow)
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
    
    # Step
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    
    print(f"   Episode {i+1}: Reward = {reward:.4f}, Precision = {info.get('precision', 0):.4f}, Recall = {info.get('recall', 0):.4f}")

print(f"\n✅ Average reward: {sum(rewards)/len(rewards):.4f}")
print(f"   Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")

# Test curriculum state
print("\n3. Testing curriculum state...")
state = env.get_curriculum_state()
print(f"   Current level: {state.get('current_level', 'N/A')}")
print(f"   Enabled levels: {state.get('enabled_levels', [])}")
print(f"   Rolling avg: {state.get('rolling_avg_rewards', {})}")

print("\n" + "=" * 80)
print("✅ TRAINING SETUP TEST COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("1. Wait for HuggingFace Space to rebuild (5-10 min)")
print("2. Test remote environment: python test_environment_connection.py")
print("3. Open training_grpo_final.ipynb in Google Colab")
print("4. Run training with GPU")
