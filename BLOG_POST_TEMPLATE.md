# Hallucination Hunter: Teaching LLMs to Detect Their Own Hallucinations with GRPO

*A reinforcement learning approach to claim-level hallucination detection*

---

## The Problem: LLMs Hallucinate, But Don't Know It

Large Language Models are incredibly powerful, but they have a critical flaw: they confidently generate false information. A model might tell you that "The Eiffel Tower was built in 1887" (it was 1889) or that "Python was created by Linus Torvalds" (it was Guido van Rossum) with complete confidence.

Current hallucination detection methods work at the response level - they classify an entire answer as hallucinated or not. But this is too coarse-grained. A response might contain 5 claims, where 4 are factual and 1 is hallucinated. We need **claim-level detection** to identify exactly which parts are wrong.

## Our Approach: Reinforcement Learning for Claim Detection

Hallucination Hunter is an OpenEnv-compatible RL environment that trains language models to:
1. **Decompose text into individual claims**
2. **Label each claim** as factual, hallucinated, or unverifiable
3. **Provide corrections** for hallucinated claims
4. **Explain their reasoning** for each label

We use **GRPO (Group Relative Policy Optimization)**, a reinforcement learning algorithm that's particularly effective for language model fine-tuning.

## Technical Architecture

### 1. Deterministic Reward System

Unlike traditional RL environments that require human feedback, our reward system is completely deterministic. We calculate rewards based on 8 components:

**Base Rewards:**
- True Positive (correctly identified hallucination): **+3.0**
- False Positive (incorrectly flagged factual claim): **-2.0**
- False Negative (missed hallucination): **-1.5**
- True Negative (correctly identified factual claim): **+0.5**

**Bonuses:**
- **Correction Quality**: 0.0-1.0 based on how well the correction matches the ground truth
- **Calibration Bonus**: +1.0 if both precision and recall exceed 0.6

**Anti-Gaming Penalties:**
- **Gaming Penalty**: -5.0 if the model flags >80% of claims (prevents "flag everything" strategy)
- **Passivity Penalty**: -3.0 if the model flags <5% when hallucinations exist (prevents "flag nothing" strategy)

These penalties force the model to learn genuine detection rather than exploiting simple strategies.

### 2. Curriculum Learning

The environment implements progressive difficulty scaling:

- **L1 (Simple)**: Single obvious hallucinations
- **L2 (Moderate)**: Plausible near-misses that require fact-checking
- **L3 (Hard)**: Mixed factual and hallucinated claims
- **L4 (Expert)**: Partial hallucinations and subtle errors

The curriculum automatically unlocks higher levels as the model's performance improves, ensuring optimal learning progression.

### 3. Episode Bank

We curated 10 high-quality episodes from three sources:
- **HaluEval**: QA and summarization with known hallucinations
- **TruthfulQA**: Questions with common misconceptions
- **Wikipedia Synthetic**: LLM-generated summaries with fact labels

Each episode includes ground truth claim decompositions with labels and corrections.

## Training Results

We trained **Qwen2.5-3B-Instruct** using GRPO for 200 steps. The results show significant improvement:

![Training Results](https://github.com/tusharpawar04/hallicunation-hunter/raw/main/grpo_training_results.png)

### Key Metrics

| Metric | Before Training | After Training | Improvement |
|--------|----------------|----------------|-------------|
| **Reward** | -4.500 | -3.046 | **+1.454 (32.3%)** |
| **Precision** | 0.150 | 0.620 | **+313%** |
| **Recall** | 0.120 | 0.580 | **+383%** |
| **Loss** | 4.313 | 1.635 | **-62%** |

The model learned to:
- ✅ **Identify claims more accurately** - Precision improved from 15% to 62%
- ✅ **Detect more hallucinations** - Recall improved from 12% to 58%
- ✅ **Generate better corrections** - Correction quality bonus increased
- ✅ **Avoid gaming penalties** - Learned balanced detection strategy

### Training Configuration

```python
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct",
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Configure GRPO
config = GRPOConfig(
    num_generations=4,
    max_steps=200,
    learning_rate=1e-5,
    temperature=0.8,
)

# Train
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    reward_fn=env.compute_reward,
)
trainer.train()
```

## Try It Yourself

### 🌐 Live Demo
Test the environment at: [https://huggingface.co/spaces/tusharpawar21/hallicunation-Hunt](https://huggingface.co/spaces/tusharpawar21/hallicunation-Hunt)

### 📓 Training Notebook
Train your own model: [Open in Colab](https://colab.research.google.com/github/tusharpawar04/hallicunation-hunter/blob/main/training_grpo_final.ipynb)

### 💻 Source Code
Full implementation: [GitHub Repository](https://github.com/tusharpawar04/hallicunation-hunter)

### Quick API Example

```python
import httpx

client = httpx.Client()

# Get an episode
response = client.post("https://tusharpawar21-hallicunation-hunt.hf.space/reset")
data = response.json()
text = data['observation']['generated_text']

# Submit detection
action = {
    "detection_output": {
        "detected_claims": [
            {
                "claim_text": "The Eiffel Tower was built in 1889",
                "label": "factual",
                "reason": "Matches historical records",
                "corrected_fact": None
            }
        ]
    }
}

response = client.post("https://tusharpawar21-hallicunation-hunt.hf.space/step", json=action)
result = response.json()

print(f"Reward: {result['reward']:.2f}")
print(f"Precision: {result['info']['precision']:.2f}")
print(f"Recall: {result['info']['recall']:.2f}")
```

## Why This Matters

Hallucination detection is critical for AI safety. As LLMs are deployed in high-stakes applications like healthcare, legal advice, and education, we need reliable methods to identify when they're generating false information.

**Claim-level detection** is more useful than response-level detection because:
1. **Granular feedback**: Identifies exactly which parts are wrong
2. **Partial credit**: Recognizes when most of a response is correct
3. **Better corrections**: Enables targeted fact-checking and correction
4. **Improved trust**: Users can verify specific claims rather than discarding entire responses

## Future Work

We're planning several enhancements:

1. **Larger episode bank**: Expand from 10 to 1000+ episodes
2. **Multi-domain coverage**: Add scientific, medical, and legal domains
3. **Larger models**: Train 7B and 13B parameter models
4. **Human evaluation**: Compare with human annotators
5. **Real-world deployment**: Integrate with production LLM systems

## Conclusion

Hallucination Hunter demonstrates that reinforcement learning can effectively train language models to detect their own hallucinations at the claim level. With a 32.3% improvement in reward and 313% improvement in precision after just 200 training steps, the approach shows promise for improving AI safety and reliability.

The environment is open-source, OpenEnv-compatible, and ready for researchers and practitioners to use. We invite the community to try it, extend it, and help make LLMs more trustworthy.

---

**Links:**
- 🌐 [Live Demo](https://huggingface.co/spaces/tusharpawar21/hallicunation-Hunt)
- 📓 [Training Notebook](https://colab.research.google.com/github/tusharpawar04/hallicunation-hunter/blob/main/training_grpo_final.ipynb)
- 💻 [GitHub Repository](https://github.com/tusharpawar04/hallicunation-hunter)
- 📖 [API Documentation](https://tusharpawar21-hallicunation-hunt.hf.space/docs)

**Built for OpenEnv Hackathon 2026** 🚀

---

*Word count: ~950 words*
