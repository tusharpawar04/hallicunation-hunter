# Hallucination Hunter - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you up and running with the Hallucination Hunter RL environment.

---

## Prerequisites

- Python 3.10+
- 4GB RAM minimum
- (Optional) GPU for training

---

## Installation

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Verify Installation

```bash
# Run tests
pytest tests/unit/ -v

# Should see: 62 tests passed
```

---

## Running the Server

### Start the API Server

```bash
python app.py
```

You should see:

```
============================================================
Hallucination Hunter RL Environment
============================================================

Initializing components...
  Loading episode bank...
    Loaded 10 episodes
    Distribution: {'L1': 1, 'L2': 9, 'L3': 0, 'L4': 0}
  Initializing curriculum manager...
    Enabled levels: ['L1']
  Initializing reward engine...
    Reward engine ready

Creating FastAPI application...

============================================================
Server starting on http://0.0.0.0:7860
============================================================
API Documentation: http://0.0.0.0:7860/docs
Health Check: http://0.0.0.0:7860/health
============================================================
```

### Test the Server

Open another terminal and run:

```bash
curl http://localhost:7860/health
```

You should see:

```json
{
  "status": "healthy",
  "episode_count": 10,
  "difficulty_distribution": {
    "L1": 1,
    "L2": 9,
    "L3": 0,
    "L4": 0
  },
  "curriculum_state": {
    "enabled_levels": ["L1"],
    "rolling_avg_rewards": {
      "L1": 0.0,
      "L2": 0.0,
      "L3": 0.0,
      "L4": 0.0
    }
  }
}
```

---

## Using the Client

### Basic Usage

```python
from src.client.env_client import HallucinationHunterEnv
from src.api.models import DetectionOutput, DetectedClaim

# Initialize client
env = HallucinationHunterEnv("http://localhost:7860")

# Reset environment (get new episode)
observation, info = env.reset()

print(f"Episode ID: {info['episode_id']}")
print(f"Difficulty: {info['difficulty_level']}")
print(f"Dataset: {info['source_dataset']}")
print(f"\nText to analyze:\n{observation['generated_text']}")

# Create detection output (example)
detection = DetectionOutput(detected_claims=[
    DetectedClaim(
        claim_text="Example claim from the text",
        label="hallucinated",  # or "factual" or "unverifiable"
        reason="This contradicts the source information",
        corrected_fact="The correct information is..."
    )
])

# Submit detection and get reward
result = env.step(detection)

print(f"\nReward: {result['reward']:.2f}")
print(f"Precision: {result['info']['precision']:.2f}")
print(f"Recall: {result['info']['recall']:.2f}")
print(f"F1 Score: {result['info']['f1']:.2f}")

# Close client
env.close()
```

### TRL-Compatible Usage (for training)

```python
from src.client.env_client import HallucinationHunterEnvTRL

# Initialize TRL client (supports 8 parallel generations)
env = HallucinationHunterEnvTRL(
    base_url="http://localhost:7860",
    num_generations=8
)

# Reset batch of episodes
observations, infos = env.reset_batch(batch_size=4)

# ... generate 8 completions per prompt ...
# ... parse completions into DetectionOutput ...

# Submit batch and get rewards
results = env.step_batch(detection_outputs)

rewards = [r["reward"] for r in results]
print(f"Batch rewards: {rewards}")

env.close()
```

---

## API Endpoints

### 1. POST /reset

Initialize a new episode.

```bash
curl -X POST http://localhost:7860/reset
```

Response:
```json
{
  "observation": {
    "generated_text": "The Eiffel Tower was built in 1889...",
    "task_instruction": "Analyze the following text..."
  },
  "info": {
    "episode_id": "halueval_qa_001",
    "difficulty_level": "L2",
    "source_dataset": "halueval_qa"
  }
}
```

### 2. POST /step

Submit detection output and get reward.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 3. GET /health

Check server status.

```bash
curl http://localhost:7860/health
```

### 4. GET /docs

Interactive API documentation (open in browser):

```
http://localhost:7860/docs
```

---

## Adding More Episodes

### Option 1: Use Preprocessing Script

1. Add raw datasets to `data/raw/`:
   - `halueval.json`
   - `truthfulqa.json`
   - `wikipedia.json`

2. Run preprocessing:

```bash
python scripts/preprocess_datasets.py
```

3. Restart server to load new episodes

### Option 2: Create Custom Episodes

Create JSON files in `data/episodes/{dataset_name}/`:

```json
{
  "episode_id": "custom_001",
  "source_dataset": "custom",
  "difficulty_level": "L1",
  "source_text": "Original context...",
  "generated_response": "LLM-generated text...",
  "claims": [
    {
      "claim_text": "Individual claim",
      "label": "factual",
      "ground_truth_fact": null
    }
  ],
  "metadata": {
    "topic": "example",
    "claim_count": 1
  }
}
```

---

## Training a Model

### 1. Prepare Training Environment

```bash
# Install training dependencies (if not already installed)
pip install transformers trl peft unsloth

# Ensure server is running
python app.py
```

### 2. Run Training Script

```bash
# In another terminal
python scripts/train_agent.py
```

This will show you the training setup guide. To actually train:

1. Load model (Qwen2.5-7B with 4-bit quantization)
2. Add LoRA adapters
3. Configure GRPO trainer
4. Run training loop (1000 steps)
5. Save checkpoints

See `scripts/train_agent.py` for detailed code.

---

## Evaluation

### 1. Evaluate Baseline

```bash
python scripts/evaluate.py
```

### 2. Generate Visualizations

```bash
python scripts/visualize.py
```

This will show you templates for creating:
- Reward curves
- Precision/recall plots
- Confusion matrices
- Before-and-after comparisons
- Interactive dashboards

---

## Docker Deployment

### Build Image

```bash
docker build -t hallucination-hunter .
```

### Run Container

```bash
docker run -p 7860:7860 hallucination-hunter
```

### Access Server

```
http://localhost:7860
```

---

## Troubleshooting

### Server won't start

**Problem**: Port 7860 already in use

**Solution**:
```bash
# Find process using port 7860
netstat -ano | findstr :7860  # Windows
lsof -i :7860                 # Linux/Mac

# Kill the process or use different port
# Edit app.py and change port = 7860 to port = 8000
```

### No episodes loaded

**Problem**: Episode bank is empty

**Solution**:
```bash
# Check if episodes exist
ls data/episodes/

# If empty, run preprocessing
python scripts/preprocess_datasets.py
```

### Import errors

**Problem**: Module not found

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Tests failing

**Problem**: Some tests fail

**Solution**:
```bash
# Run specific test file
pytest tests/unit/test_core_models.py -v

# Check error messages
# Most common: missing dependencies or incorrect paths
```

---

## Next Steps

### 1. Explore the API

- Open http://localhost:7860/docs
- Try the interactive API documentation
- Test different endpoints

### 2. Run Example Code

- See `examples/` directory (if available)
- Try the client usage examples above
- Experiment with different detection outputs

### 3. Add More Data

- Collect raw datasets
- Run preprocessing
- Expand episode bank to 1000+

### 4. Train a Model

- Follow training script guide
- Run for 1000 steps
- Evaluate results

### 5. Deploy

- Build Docker image
- Deploy to cloud
- Share with others

---

## Resources

### Documentation

- **README.md**: Comprehensive project overview
- **IMPLEMENTATION_REPORT.md**: Technical deep dive
- **PROJECT_STATUS.md**: Current status and roadmap
- **Design Document**: `.kiro/specs/hallucination-hunter/design.md`
- **Requirements**: `.kiro/specs/hallucination-hunter/requirements.md`

### Code Structure

```
hallucination-hunter/
├── src/
│   ├── api/              # FastAPI server
│   ├── client/           # Client wrappers
│   ├── environment/      # Core RL environment
│   ├── parsers/          # Dataset parsers
│   └── utils/            # Utilities
├── tests/                # Test suites
├── data/                 # Episode bank
├── scripts/              # Training/evaluation
├── configs/              # Configuration files
└── app.py                # Entry point
```

### Support

- **Issues**: Report bugs or request features
- **Discussions**: Ask questions or share ideas
- **Documentation**: Read the comprehensive docs

---

## Quick Reference

### Common Commands

```bash
# Start server
python app.py

# Run tests
pytest tests/unit/ -v

# Preprocess data
python scripts/preprocess_datasets.py

# Train model
python scripts/train_agent.py

# Evaluate
python scripts/evaluate.py

# Visualize
python scripts/visualize.py

# Docker build
docker build -t hallucination-hunter .

# Docker run
docker run -p 7860:7860 hallucination-hunter
```

### Key Files

- `app.py` - Server entry point
- `src/environment/core.py` - Main environment
- `src/api/server.py` - API endpoints
- `src/client/env_client.py` - Client wrappers
- `requirements.txt` - Dependencies

---

## Success Checklist

- [ ] Installed dependencies
- [ ] Downloaded spaCy model
- [ ] Ran tests (62 passing)
- [ ] Started server
- [ ] Tested health endpoint
- [ ] Ran client example
- [ ] Explored API docs
- [ ] Added custom episode
- [ ] Ready for training!

---

**Congratulations!** You're now ready to use Hallucination Hunter for training language models to detect hallucinations.

For more details, see:
- **README.md** - Full documentation
- **IMPLEMENTATION_REPORT.md** - Technical details
- **API Docs** - http://localhost:7860/docs

Happy training! 🚀
