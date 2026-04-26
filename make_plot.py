import matplotlib.pyplot as plt
import numpy as np

# Your actual training data from the logs
steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450]
rewards = [-4.55, -4.55, -4.675, -4.525, -4.4375, -4.2875, -4.15, -4.1, -3.95, -3.9625, -3.6, -3.575, -3.4, -3.375, -3.3625, -3.225, -3.175, -3.2125, -3.1125, -3.0625, -3.15, -3.0375, -3.025, -3.0375, -3.0375, -3.05, -3.0125, -3.0375, -3.05]

# Create beautiful plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot training curve
ax.plot(steps, rewards, 'b-', linewidth=2.5, marker='o', markersize=5, label='Training Reward')

# Baseline
ax.axhline(y=-4.55, color='r', linestyle='--', linewidth=2, label='Baseline (-4.55)', alpha=0.7)

# Styling
ax.set_xlabel('Training Steps', fontsize=15, fontweight='bold')
ax.set_ylabel('Average Reward', fontsize=15, fontweight='bold')
ax.set_title('GRPO Training Results - Hallucination Hunter\nQwen2.5-3B-Instruct', fontsize=17, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, loc='lower right')

# Add improvement annotation
improvement = rewards[-1] - rewards[0]
improvement_pct = (improvement / abs(rewards[0])) * 100
textstr = f'Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)\nFinal Reward: {rewards[-1]:.3f}'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', bbox=props, fontweight='bold')

# Add training info
info_text = f'Steps: {steps[-1]} | Method: GRPO | Model: Qwen2.5-3B-Instruct'
ax.text(0.5, -0.12, info_text, transform=ax.transAxes, fontsize=11,
        ha='center', style='italic', color='gray')

plt.tight_layout()
plt.savefig('grpo_training_results.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as grpo_training_results.png")
print(f"\n📊 Training Summary:")
print(f"   Steps: {steps[-1]}")
print(f"   Initial Reward: {rewards[0]:.3f}")
print(f"   Final Reward: {rewards[-1]:.3f}")
print(f"   Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
print(f"\n🎯 This is excellent training evidence for the hackathon!")
