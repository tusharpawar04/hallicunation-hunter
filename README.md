---
title: Hallucination Hunter
emoji: 🎯
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
  - hallucination-detection
  - reinforcement-learning
  - ai-safety
  - nlp
app_port: 7860
---

# 🎯 Hallucination Hunter RL Environment

An OpenEnv-compatible reinforcement learning environment for training language models to detect hallucinations at the claim level.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/hallucination-hunter)

🚀 **[Try it live](https://huggingface.co/spaces/YOUR_USERNAME/hallucination-hunter)** | 📖 **[API Docs](https://huggingface.co/spaces/YOUR_USERNAME/hallucination-hunter/docs)** | 💻 **[GitHub](https://github.com/yourusername/hallucination-hunter)**

## Features

- **Deterministic Rewards**: Reproducible scoring without human labeling based on precision, recall, and correction quality
- **Curriculum Learning**: Progressive difficulty scaling from L1 (simple) to L4 (expert) based on performance
- **Anti-Gaming Penalties**: Prevents trivial strategies like flagging all or no claims
- **Claim-Level Detection**: Fine-grained hallucination identification with corrections
- **GRPO Training Support**: Compatible with HuggingFace TRL for Group Relative Policy Optimization
- **FastAPI Server**: RESTful API for easy integration with training frameworks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Framework                       │
│                    (TRL/GRPO Trainer)                       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI Server                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   /reset     │  │    /step     │  │   /health    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            HallucinationEnvironment                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ EpisodeBank  │  │ Curriculum   │  │ RewardEngine │     │
│  │              │  │  Manager     │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hallucination-hunter.git
cd hallucination-hunter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Start the API Server

```bash
python app.py
```

The server will start on `http://localhost:7860`. Visit `http://localhost:7860/docs` for interactive API documentation.

### 2. Test the Environment

```python
from src.client.env_client import HallucinationHunterEnv

# Initialize client
env = HallucinationHunterEnv("http://localhost:7860")

# Reset environment
observation, info = env.reset()
print(f"Episode: {info['episode_id']}")
print(f"Difficulty: {info['difficulty_level']}")
print(f"Text to analyze: {observation['generated_text']}")

# Submit detection output
from src.api.models import DetectionOutput, DetectedClaim

detection = DetectionOutput(detected_claims=[
    DetectedClaim(
        claim_text="Example claim",
        label="hallucinated",
        reason="Incorrect information",
        corrected_fact="Correct information"
    )
])

result = env.step(detection)
print(f"Reward: {result['reward']}")
print(f"Precision: {result['info']['precision']}")
print(f"Recall: {result['info']['recall']}")
```

### 3. Train a Model

```bash
# See scripts/train_agent.py for training setup
python scripts/train_agent.py
```

## Dataset

The environment uses three data sources:

1. **HaluEval**: QA, summarization, and dialog samples with hallucinated answers
2. **TruthfulQA**: Questions with best answers and common misconceptions
3. **Wikipedia Synthetic**: LLM-generated summaries with fact labels

To add your own datasets, place JSON files in `data/raw/` and run:

```bash
python scripts/preprocess_datasets.py
```

## Reward Formula

The reward combines multiple components:

**Base Rewards:**
- True Positive (correctly identified hallucination): +3.0
- False Positive (incorrectly flagged factual claim): -2.0
- False Negative (missed hallucination): -1.5
- True Negative (correctly identified factual claim): +0.5

**Bonuses:**
- Correction Bonus: 0.0-1.0 based on keyword overlap with ground truth
- Calibration Bonus: +1.0 if both precision and recall > 0.6

**Penalties:**
- Gaming Penalty: -5.0 if >80% of claims flagged
- Passivity Penalty: -3.0 if <5% flagged when hallucinations exist

**Difficulty Multipliers:**
- L1: 1.0x
- L2: 1.5x
- L3: 2.0x
- L4: 2.5x

## Curriculum Learning

The curriculum starts with only L1 episodes enabled. As the agent's rolling average reward (over 50 episodes) exceeds thresholds, higher difficulty levels are unlocked:

- L1 → L2: avg reward > 3.5
- L2 → L3: avg reward > 4.0
- L3 → L4: avg reward > 5.0

## API Endpoints

### POST /reset
Initialize a new episode.

**Response:**
```json
{
  "observation": {
    "generated_text": "The Eiffel Tower was built in 1889...",
    "task_instruction": "Analyze the following text..."
  },
  "info": {
    "episode_id": "ep_001",
    "difficulty_level": "L1",
    "source_dataset": "halueval_qa"
  }
}
```

### POST /step
Submit detection output and get reward.

**Request:**
```json
{
  "action": {
    "detection_output": {
      "detected_claims": [
        {
          "claim_text": "The Eiffel Tower was built in 1889",
          "label": "factual",
          "reason": "Matches historical records",
          "corrected_fact": null
        }
      ]
    }
  }
}
```

**Response:**
```json
{
  "observation": {...},
  "reward": 4.2,
  "done": true,
  "info": {
    "precision": 0.85,
    "recall": 0.90,
    "f1": 0.87,
    ...
  }
}
```

### GET /health
Get server status and statistics.

**Response:**
```json
{
  "status": "healthy",
  "episode_count": 1247,
  "difficulty_distribution": {
    "L1": 312,
    "L2": 415,
    "L3": 320,
    "L4": 200
  },
  "curriculum_state": {
    "enabled_levels": ["L1", "L2"],
    "rolling_avg_rewards": {
      "L1": 3.8,
      "L2": 2.1,
      "L3": 0.0,
      "L4": 0.0
    }
  }
}
```

## Docker Deployment

```bash
# Build image
docker build -t hallucination-hunter .

# Run container
docker run -p 7860:7860 hallucination-hunter
```

## HuggingFace Spaces Deployment

1. Create a new Space on HuggingFace
2. Select "Docker" as the SDK
3. Push this repository to the Space
4. The Dockerfile will be used automatically

## Project Structure

```
hallucination-hunter/
├── src/
│   ├── api/              # FastAPI server and models
│   ├── client/           # Environment client wrappers
│   ├── environment/      # Core RL environment
│   ├── parsers/          # Dataset parsers
│   └── utils/            # Utilities (metrics, claim extraction)
├── data/
│   ├── episodes/         # Processed episode bank
│   └── raw/              # Raw datasets
├── scripts/              # Training and preprocessing scripts
├── tests/                # Unit and integration tests
├── app.py                # Main application entry point
├── Dockerfile            # Docker configuration
└── requirements.txt      # Python dependencies
```

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## Performance

- **Episode sampling**: <10ms
- **Reward calculation**: <50ms
- **API response time**: <100ms (p95)
- **Throughput**: 60 requests/minute (rate limited)

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{hallucination_hunter_2024,
  title={Hallucination Hunter: An RL Environment for Training Hallucination Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hallucination-hunter}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

- Documentation: [https://hallucination-hunter.readthedocs.io](https://hallucination-hunter.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/yourusername/hallucination-hunter/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/hallucination-hunter/discussions)
