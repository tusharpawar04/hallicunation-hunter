#!/usr/bin/env python3
"""
Local training demonstration
Shows how the training loop works and generates sample results
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from src.environment.core import HallucinationEnvironment
from src.environment.episode_bank import EpisodeBank
from src.environment.curriculum import CurriculumManager
from src.environment.reward import RewardEngine

print("=" * 80)
print("HALLUCINATION HUNTER - LOCAL TRAINING DEMO")
print("=" * 80)

# Initialize environment
print("\n📦 Initializing environment...")
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

# Simulate baseline (untrained model)
print("\n📊 Collecting baseline rewards (simulating untrained model)...")
baseline_rewards = []
baseline_precision = []
baseline_recall = []

for i in range(10):
    obs, info = env.reset()
    
    # Simulate poor detection (random guessing)
    action = {
        "detected_claims": [
            {
                "claim_text": "Random claim",
                "label": np.random.choice(["factual", "hallucinated"]),
                "reason": "Random guess",
                "corrected_fact": None
            }
        ]
    }
    
    obs, reward, done, info = env.step(action)
    baseline_rewards.append(reward)
    baseline_precision.append(info.get('precision', 0))
    baseline_recall.append(info.get('recall', 0))
    
    if (i + 1) % 5 == 0:
        print(f"   Baseline {i+1}/10: Avg Reward = {np.mean(baseline_rewards):.3f}")

print(f"\n✅ Baseline Results:")
print(f"   Reward: {np.mean(baseline_rewards):.3f} (±{np.std(baseline_rewards):.3f})")
print(f"   Precision: {np.mean(baseline_precision):.3f}")
print(f"   Recall: {np.mean(baseline_recall):.3f}")

# Simulate training (improving model)
print("\n🚀 Simulating training (100 steps)...")
training_rewards = []
training_losses = []
steps = []

# Simulate gradual improvement
for step in range(100):
    obs, info = env.reset()
    
    # Simulate improving detection (better over time)
    improvement_factor = min(step / 100.0, 0.8)  # Improve up to 80%
    
    # Better action as training progresses
    if np.random.random() < improvement_factor:
        # Good detection
        action = {
            "detected_claims": [
                {
                    "claim_text": "Improving claim detection",
                    "label": "factual",
                    "reason": "Better reasoning",
                    "corrected_fact": None
                }
            ]
        }
    else:
        # Still some mistakes
        action = {
            "detected_claims": [
                {
                    "claim_text": "Random claim",
                    "label": np.random.choice(["factual", "hallucinated"]),
                    "reason": "Still learning",
                    "corrected_fact": None
                }
            ]
        }
    
    obs, reward, done, info = env.step(action)
    training_rewards.append(reward)
    
    # Simulate loss decreasing
    loss = 2.0 * (1 - improvement_factor) + np.random.normal(0, 0.1)
    training_losses.append(max(0.5, loss))
    steps.append(step)
    
    if (step + 1) % 20 == 0:
        recent_reward = np.mean(training_rewards[-20:])
        print(f"   Step {step+1}/100: Avg Reward = {recent_reward:.3f}, Loss = {training_losses[-1]:.3f}")

# Simulate trained model
print("\n📊 Collecting trained model rewards...")
trained_rewards = []
trained_precision = []
trained_recall = []

for i in range(10):
    obs, info = env.reset()
    
    # Simulate better detection (trained model)
    action = {
        "detected_claims": [
            {
                "claim_text": "Better claim detection",
                "label": "factual",
                "reason": "Trained reasoning",
                "corrected_fact": None
            }
        ]
    }
    
    obs, reward, done, info = env.step(action)
    trained_rewards.append(reward)
    trained_precision.append(info.get('precision', 0))
    trained_recall.append(info.get('recall', 0))

print(f"\n✅ Trained Results:")
print(f"   Reward: {np.mean(trained_rewards):.3f} (±{np.std(trained_rewards):.3f})")
print(f"   Precision: {np.mean(trained_precision):.3f}")
print(f"   Recall: {np.mean(trained_recall):.3f}")

# Calculate improvement
improvement = np.mean(trained_rewards) - np.mean(baseline_rewards)
print(f"\n📈 Improvement: {improvement:+.3f} ({(improvement/abs(np.mean(baseline_rewards))*100):.1f}%)")

# Generate plots
print("\n📊 Generating plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Reward over training
axes[0].plot(steps, training_rewards, 'b-', linewidth=2, alpha=0.6, label='Training Rewards')
# Add smoothed line
window = 10
smoothed = np.convolve(training_rewards, np.ones(window)/window, mode='valid')
axes[0].plot(range(window-1, len(training_rewards)), smoothed, 'r-', linewidth=3, label='Smoothed')
axes[0].set_xlabel('Training Steps', fontsize=12)
axes[0].set_ylabel('Reward', fontsize=12)
axes[0].set_title('Reward Over Training (GRPO)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Loss over training
axes[1].plot(steps, training_losses, 'r-', linewidth=2)
axes[1].set_xlabel('Training Steps', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Loss Over Training', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Before vs After
comparison = {
    'Before\nTraining': np.mean(baseline_rewards),
    'After\nTraining': np.mean(trained_rewards)
}
colors = ['#ff6b6b', '#51cf66']
bars = axes[2].bar(comparison.keys(), comparison.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[2].set_ylabel('Average Reward', fontsize=12)
axes[2].set_title('Before vs After Training', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)

for i, (k, v) in enumerate(comparison.items()):
    axes[2].text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('local_training_demo_results.png', dpi=300, bbox_inches='tight')
print("✅ Plots saved to local_training_demo_results.png")

# Summary
print("\n" + "=" * 80)
print("TRAINING DEMO SUMMARY")
print("=" * 80)
print(f"\nModel: Simulated GRPO Training")
print(f"Training Steps: 100")
print(f"\nLoss:")
print(f"  Initial: {training_losses[0]:.4f}")
print(f"  Final: {training_losses[-1]:.4f}")
print(f"  Improvement: {((training_losses[0] - training_losses[-1]) / training_losses[0] * 100):.1f}%")
print(f"\nRewards:")
print(f"  Before: {np.mean(baseline_rewards):.3f} (±{np.std(baseline_rewards):.3f})")
print(f"  After: {np.mean(trained_rewards):.3f} (±{np.std(trained_rewards):.3f})")
print(f"  Improvement: {improvement:+.3f}")
print(f"\n✅ Demo complete! This shows the expected training behavior.")
print(f"\n📝 Next: Run actual GRPO training in Colab with training_grpo_final.ipynb")
print("=" * 80)

plt.show()
