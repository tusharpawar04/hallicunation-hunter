#!/usr/bin/env python3
"""Quick training demo - tests environment interaction without GPU."""

import httpx
import json
import re
import time
from typing import List, Dict

# Environment URL
ENV_URL = "https://tusharpawar21-hallicunation-hunt.hf.space"

class TrainingDemo:
    """Simulates training loop to test environment."""
    
    def __init__(self, env_url: str):
        self.env_url = env_url.rstrip('/')
        self.client = httpx.Client(timeout=60.0)
        self.rewards_history = []
        
    def get_episode(self) -> Dict:
        """Get a new episode from environment."""
        response = self.client.post(f"{self.env_url}/reset")
        response.raise_for_status()
        return response.json()
    
    def submit_detection(self, detection: Dict) -> Dict:
        """Submit detection and get reward."""
        action = {"detection_output": detection}
        response = self.client.post(f"{self.env_url}/step", json=action)
        response.raise_for_status()
        return response.json()
    
    def generate_simple_detection(self, text: str) -> Dict:
        """Generate a simple detection (simulates model output)."""
        # Simple heuristic: split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        detected_claims = []
        for sentence in sentences[:3]:  # Max 3 claims
            if len(sentence) > 10:
                detected_claims.append({
                    "claim_text": sentence,
                    "label": "factual",  # Simple baseline
                    "reason": "Appears to be a factual statement",
                    "corrected_fact": None
                })
        
        return {"detected_claims": detected_claims}
    
    def run_training_simulation(self, num_episodes: int = 20):
        """Simulate training for N episodes."""
        print("=" * 80)
        print("TRAINING SIMULATION")
        print("=" * 80)
        print(f"\nRunning {num_episodes} episodes to test environment interaction...")
        print("(This simulates what GRPO training does)\n")
        
        rewards = []
        precisions = []
        recalls = []
        
        for i in range(num_episodes):
            try:
                # Get episode
                episode_data = self.get_episode()
                obs = episode_data['observation']
                info = episode_data['info']
                
                # Generate detection (simulates model inference)
                detection = self.generate_simple_detection(obs['generated_text'])
                
                # Submit and get reward
                result = self.submit_detection(detection)
                
                reward = result['reward']
                precision = result['info'].get('precision', 0)
                recall = result['info'].get('recall', 0)
                
                rewards.append(reward)
                precisions.append(precision)
                recalls.append(recall)
                
                # Print progress
                if (i + 1) % 5 == 0:
                    avg_reward = sum(rewards[-5:]) / 5
                    print(f"Episodes {i-3:2d}-{i+1:2d}: Avg Reward = {avg_reward:+.3f}")
                
            except Exception as e:
                print(f"❌ Episode {i+1} failed: {e}")
                continue
        
        # Summary
        print("\n" + "=" * 80)
        print("SIMULATION RESULTS")
        print("=" * 80)
        
        if rewards:
            print(f"\nEpisodes completed: {len(rewards)}/{num_episodes}")
            print(f"\nRewards:")
            print(f"  Average: {sum(rewards)/len(rewards):.3f}")
            print(f"  Min: {min(rewards):.3f}")
            print(f"  Max: {max(rewards):.3f}")
            
            print(f"\nMetrics:")
            print(f"  Avg Precision: {sum(precisions)/len(precisions):.3f}")
            print(f"  Avg Recall: {sum(recalls)/len(recalls):.3f}")
            
            # Show trend
            first_half = rewards[:len(rewards)//2]
            second_half = rewards[len(rewards)//2:]
            
            if first_half and second_half:
                improvement = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
                print(f"\nTrend:")
                print(f"  First half avg: {sum(first_half)/len(first_half):.3f}")
                print(f"  Second half avg: {sum(second_half)/len(second_half):.3f}")
                print(f"  Change: {improvement:+.3f}")
            
            print("\n✅ Environment is working correctly!")
            print("\nThis demonstrates that:")
            print("  • Environment accepts episodes")
            print("  • Detections are processed")
            print("  • Rewards are calculated")
            print("  • Ready for GRPO training")
            
            print("\n" + "=" * 80)
            print("NEXT STEP: Run full GRPO training in Google Colab")
            print("=" * 80)
            print("\n1. Go to https://colab.research.google.com/")
            print("2. Upload training_grpo_final.ipynb")
            print("3. Enable GPU (Runtime → Change runtime type → T4)")
            print("4. Run all cells")
            print("5. Training will take 2-3 hours")
            
        else:
            print("\n❌ No episodes completed successfully")
            print("Check environment connection")


if __name__ == "__main__":
    print("\n🎯 Hallucination Hunter - Training Demo\n")
    
    # Test connection
    print("Testing environment connection...")
    try:
        client = httpx.Client(timeout=30.0)
        health = client.get(f"{ENV_URL}/health").json()
        print(f"✅ Connected to environment")
        print(f"   Episodes available: {health['episode_count']}")
        print(f"   Difficulty distribution: {health['difficulty_distribution']}\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nMake sure the HuggingFace Space is running:")
        print(f"   {ENV_URL}")
        exit(1)
    
    # Run simulation
    demo = TrainingDemo(ENV_URL)
    demo.run_training_simulation(num_episodes=20)
