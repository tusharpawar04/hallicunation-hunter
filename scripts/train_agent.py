"""Training script for Hallucination Hunter using GRPO.

This script demonstrates how to train a language model to detect hallucinations
using Group Relative Policy Optimization (GRPO) with the Hallucination Hunter environment.

Requirements:
- transformers
- trl
- unsloth (for 4-bit quantization)
- peft (for LoRA)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client.env_client import HallucinationHunterEnvTRL


def main():
    """Main training function."""
    print("=" * 60)
    print("Hallucination Hunter Training Script")
    print("=" * 60)
    
    # Configuration
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_url = "http://localhost:7860"
    num_generations = 8
    max_steps = 1000
    learning_rate = 1e-5
    checkpoint_interval = 100
    
    print("\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  API URL: {base_url}")
    print(f"  Generations per prompt: {num_generations}")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate}")
    
    # Initialize environment client
    print("\nInitializing environment client...")
    env = HallucinationHunterEnvTRL(base_url=base_url, num_generations=num_generations)
    
    # Check server health
    try:
        health = env.session.get(f"{base_url}/health").json()
        print(f"  Server status: {health['status']}")
        print(f"  Episodes available: {health['episode_count']}")
        print(f"  Enabled levels: {health['curriculum_state']['enabled_levels']}")
    except Exception as e:
        print(f"  Error connecting to server: {e}")
        print("  Make sure the server is running: python app.py")
        return
    
    print("\n" + "=" * 60)
    print("Training Setup")
    print("=" * 60)
    
    print("""
To train the model, you would:

1. Load the model with Unsloth 4-bit quantization:
   ```python
   from unsloth import FastLanguageModel
   
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="Qwen/Qwen2.5-7B-Instruct",
       max_seq_length=2048,
       dtype=None,
       load_in_4bit=True,
   )
   ```

2. Add LoRA adapters:
   ```python
   model = FastLanguageModel.get_peft_model(
       model,
       r=16,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
       lora_alpha=16,
       lora_dropout=0,
       bias="none",
       use_gradient_checkpointing=True,
   )
   ```

3. Configure GRPO trainer:
   ```python
   from trl import GRPOConfig, GRPOTrainer
   
   config = GRPOConfig(
       num_generations=8,
       learning_rate=1e-5,
       max_steps=1000,
       per_device_train_batch_size=1,
       gradient_accumulation_steps=8,
   )
   
   trainer = GRPOTrainer(
       model=model,
       tokenizer=tokenizer,
       config=config,
       # Custom reward function that calls env.step()
   )
   ```

4. Training loop:
   ```python
   for step in range(max_steps):
       # Reset environment
       observations, infos = env.reset_batch(batch_size=1)
       
       # Generate 8 completions per prompt
       prompts = [obs["generated_text"] for obs in observations]
       completions = model.generate(prompts, num_return_sequences=8)
       
       # Parse completions into DetectionOutput
       detection_outputs = parse_completions(completions)
       
       # Get rewards from environment
       results = env.step_batch(detection_outputs)
       rewards = [r["reward"] for r in results]
       
       # Update model with GRPO
       trainer.step(prompts, completions, rewards)
       
       # Save checkpoint
       if step % 100 == 0:
           model.save_pretrained(f"./checkpoints/step_{step}")
   ```

5. Save final model:
   ```python
   model.save_pretrained("./final_model")
   tokenizer.save_pretrained("./final_model")
   ```
""")
    
    print("=" * 60)
    print("To run actual training:")
    print("1. Install dependencies: pip install transformers trl unsloth peft")
    print("2. Start the API server: python app.py")
    print("3. Implement the training loop above")
    print("4. Run for 1000 steps (~2-4 hours on GPU)")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
