"""Visualization script for training metrics and comparisons."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_visualization_guide():
    """Generate guide for creating visualizations."""
    
    print("=" * 60)
    print("Hallucination Hunter Visualization Guide")
    print("=" * 60)
    
    print("""
This script provides templates for visualizing training results.

## Required Libraries
```bash
pip install matplotlib plotly pandas
```

## 1. Reward Curves Per Difficulty Level

```python
import matplotlib.pyplot as plt
import json

# Load metrics
with open("logs/metrics.json", "r") as f:
    metrics = json.load(f)

# Extract rewards by difficulty
rewards_by_level = {
    "L1": [],
    "L2": [],
    "L3": [],
    "L4": []
}

for episode in metrics:
    level = episode["difficulty_level"]
    rewards_by_level[level].append(episode["reward"])

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

for level, rewards in rewards_by_level.items():
    if rewards:
        ax.plot(rewards, label=level, alpha=0.7)

ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title("Reward Curves by Difficulty Level")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("reward_curves.png", dpi=300, bbox_inches="tight")
plt.show()
```

## 2. Precision and Recall Over Time

```python
import matplotlib.pyplot as plt

# Extract precision and recall
precision = [m["precision"] for m in metrics]
recall = [m["recall"] for m in metrics]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Precision
ax1.plot(precision, color="blue", alpha=0.7)
ax1.axhline(y=0.6, color="red", linestyle="--", label="Target (0.6)")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Precision")
ax1.set_title("Precision Over Training")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Recall
ax2.plot(recall, color="green", alpha=0.7)
ax2.axhline(y=0.6, color="red", linestyle="--", label="Target (0.6)")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Recall")
ax2.set_title("Recall Over Training")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("precision_recall.png", dpi=300, bbox_inches="tight")
plt.show()
```

## 3. Cumulative Reward

```python
import numpy as np

# Calculate cumulative reward
cumulative_reward = np.cumsum([m["reward"] for m in metrics])

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cumulative_reward, color="purple", linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative Reward")
ax.set_title("Cumulative Reward Over Training")
ax.grid(True, alpha=0.3)
plt.savefig("cumulative_reward.png", dpi=300, bbox_inches="tight")
plt.show()
```

## 4. Confusion Matrix Heatmap

```python
import seaborn as sns
import pandas as pd

# Calculate average confusion matrix
avg_tp = np.mean([m["true_positives"] for m in metrics])
avg_fp = np.mean([m["false_positives"] for m in metrics])
avg_fn = np.mean([m["false_negatives"] for m in metrics])
avg_tn = np.mean([m["true_negatives"] for m in metrics])

# Create confusion matrix
cm = np.array([
    [avg_tp, avg_fp],
    [avg_fn, avg_tn]
])

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=["Predicted Hallucinated", "Predicted Factual"],
    yticklabels=["Actual Hallucinated", "Actual Factual"],
    ax=ax
)
ax.set_title("Average Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
```

## 5. Before-and-After Comparison

```python
import json

# Load comparison results
with open("evaluation_results.json", "r") as f:
    comparison = json.load(f)

baseline = comparison["baseline"]
trained = comparison["trained"]

# Create comparison bar chart
metrics_names = ["Reward", "Precision", "Recall", "F1"]
baseline_values = [
    baseline["avg_reward"],
    baseline["avg_precision"],
    baseline["avg_recall"],
    baseline["avg_f1"]
]
trained_values = [
    trained["avg_reward"],
    trained["avg_precision"],
    trained["avg_recall"],
    trained["avg_f1"]
]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, baseline_values, width, label="Baseline", color="lightcoral")
bars2 = ax.bar(x + width/2, trained_values, width, label="Trained", color="lightgreen")

ax.set_xlabel("Metric")
ax.set_ylabel("Value")
ax.set_title("Baseline vs Trained Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.tight_layout()
plt.savefig("comparison.png", dpi=300, bbox_inches="tight")
plt.show()
```

## 6. Interactive Dashboard (Plotly)

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Reward Over Time",
        "Precision vs Recall",
        "Confusion Matrix Components",
        "Difficulty Distribution"
    )
)

# Reward over time
fig.add_trace(
    go.Scatter(
        y=[m["reward"] for m in metrics],
        mode="lines",
        name="Reward"
    ),
    row=1, col=1
)

# Precision vs Recall
fig.add_trace(
    go.Scatter(
        x=[m["precision"] for m in metrics],
        y=[m["recall"] for m in metrics],
        mode="markers",
        name="Episodes",
        marker=dict(
            size=5,
            color=[m["reward"] for m in metrics],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Reward")
        )
    ),
    row=1, col=2
)

# Confusion matrix components
fig.add_trace(
    go.Bar(
        x=["TP", "FP", "FN", "TN"],
        y=[
            np.mean([m["true_positives"] for m in metrics]),
            np.mean([m["false_positives"] for m in metrics]),
            np.mean([m["false_negatives"] for m in metrics]),
            np.mean([m["true_negatives"] for m in metrics])
        ],
        name="Avg Count"
    ),
    row=2, col=1
)

# Difficulty distribution
difficulty_counts = {}
for m in metrics:
    level = m["difficulty_level"]
    difficulty_counts[level] = difficulty_counts.get(level, 0) + 1

fig.add_trace(
    go.Pie(
        labels=list(difficulty_counts.keys()),
        values=list(difficulty_counts.values()),
        name="Episodes"
    ),
    row=2, col=2
)

fig.update_layout(
    title_text="Hallucination Hunter Training Dashboard",
    showlegend=True,
    height=800
)

fig.write_html("dashboard.html")
fig.show()
```

## 7. Detection Examples Visualization

```python
def visualize_detection_example(episode_data, detection_output):
    \"\"\"Visualize a single detection example.\"\"\"
    
    print("=" * 60)
    print(f"Episode: {episode_data['episode_id']}")
    print(f"Difficulty: {episode_data['difficulty_level']}")
    print("=" * 60)
    
    print("\\nGenerated Text:")
    print(episode_data['generated_text'])
    
    print("\\nDetected Claims:")
    for i, claim in enumerate(detection_output['detected_claims'], 1):
        print(f"\\n{i}. {claim['claim_text']}")
        print(f"   Label: {claim['label']}")
        print(f"   Reason: {claim['reason']}")
        if claim['corrected_fact']:
            print(f"   Correction: {claim['corrected_fact']}")
    
    print("\\n" + "=" * 60)
```

## Usage

1. Train your model and save metrics to `logs/metrics.json`
2. Run evaluation and save results to `evaluation_results.json`
3. Run the visualization code above to generate plots
4. Open `dashboard.html` for interactive exploration

""")


def main():
    """Main visualization function."""
    generate_visualization_guide()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Train a model: python scripts/train_agent.py")
    print("2. Collect metrics during training")
    print("3. Run evaluation: python scripts/evaluate.py")
    print("4. Use the code above to create visualizations")
    print("=" * 60)


if __name__ == "__main__":
    main()
