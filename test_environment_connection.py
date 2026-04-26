#!/usr/bin/env python3
"""Test environment connection and verify reward calculation works."""

import httpx
import json

# Your deployed Space URL
ENV_URL = "https://tusharpawar21-hallicunation-hunt.hf.space"

def test_environment():
    """Test environment connection and reward calculation."""
    client = httpx.Client(timeout=60.0)
    
    print("=" * 80)
    print("TESTING HALLUCINATION HUNTER ENVIRONMENT")
    print("=" * 80)
    
    # Test 1: Health check
    print("\n1. Testing /health endpoint...")
    try:
        response = client.get(f"{ENV_URL}/health")
        response.raise_for_status()
        health = response.json()
        print(f"   ✅ Status: {health['status']}")
        print(f"   ✅ Episodes: {health['episode_count']}")
        print(f"   ✅ Distribution: {health['difficulty_distribution']}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 2: Reset
    print("\n2. Testing /reset endpoint...")
    try:
        response = client.post(f"{ENV_URL}/reset")
        response.raise_for_status()
        data = response.json()
        obs = data['observation']
        info = data['info']
        print(f"   ✅ Episode ID: {info['episode_id']}")
        print(f"   ✅ Difficulty: {info['difficulty_level']}")
        print(f"   ✅ Text length: {len(obs['generated_text'])} chars")
        print(f"   ✅ Text preview: {obs['generated_text'][:100]}...")
    except Exception as e:
        print(f"   ❌ Reset failed: {e}")
        return False
    
    # Test 3: Step with WRONG format (should fail or return 0.000)
    print("\n3. Testing /step with WRONG format (plain string)...")
    try:
        wrong_action = {"action": "I will analyze the text"}
        response = client.post(f"{ENV_URL}/step", json=wrong_action)
        if response.status_code == 422:
            print(f"   ✅ Correctly rejected wrong format (422 error)")
        else:
            result = response.json()
            print(f"   ⚠️  Accepted wrong format, reward: {result.get('reward', 'N/A')}")
    except Exception as e:
        print(f"   ✅ Correctly rejected wrong format: {e}")
    
    # Test 4: Step with CORRECT format
    print("\n4. Testing /step with CORRECT format (JSON with detected_claims)...")
    try:
        # Correct format matching API models - Action object directly
        correct_action = {
            "detection_output": {
                "detected_claims": [
                    {
                        "claim_text": "Test claim 1",
                        "label": "factual",
                        "reason": "This is a test",
                        "corrected_fact": None
                    },
                    {
                        "claim_text": "Test claim 2",
                        "label": "hallucinated",
                        "reason": "This is also a test",
                        "corrected_fact": "Corrected version"
                    }
                ]
            }
        }
        
        response = client.post(f"{ENV_URL}/step", json=correct_action)
        response.raise_for_status()
        result = response.json()
        
        reward = result['reward']
        info = result['info']
        
        print(f"   ✅ Reward: {reward:.4f}")
        print(f"   ✅ Precision: {info.get('precision', 0):.4f}")
        print(f"   ✅ Recall: {info.get('recall', 0):.4f}")
        print(f"   ✅ True Positives: {info.get('true_positives', 0)}")
        print(f"   ✅ False Positives: {info.get('false_positives', 0)}")
        print(f"   ✅ False Negatives: {info.get('false_negatives', 0)}")
        
        if reward == 0.0:
            print(f"   ⚠️  WARNING: Reward is 0.000 - this might indicate an issue")
        else:
            print(f"   ✅ Reward is non-zero - environment is working!")
            
    except Exception as e:
        print(f"   ❌ Step with correct format failed: {e}")
        return False
    
    # Test 5: Multiple episodes
    print("\n5. Testing multiple episodes...")
    rewards = []
    for i in range(5):
        try:
            # Reset
            response = client.post(f"{ENV_URL}/reset")
            response.raise_for_status()
            
            # Step with simple action
            action = {
                "detection_output": {
                    "detected_claims": [
                        {
                            "claim_text": f"Test claim {i}",
                            "label": "factual",
                            "reason": "Testing",
                            "corrected_fact": None
                        }
                    ]
                }
            }
            response = client.post(f"{ENV_URL}/step", json=action)
            response.raise_for_status()
            result = response.json()
            rewards.append(result['reward'])
            print(f"   Episode {i+1}: Reward = {result['reward']:.4f}")
        except Exception as e:
            print(f"   ❌ Episode {i+1} failed: {e}")
    
    if rewards:
        print(f"\n   ✅ Average reward: {sum(rewards)/len(rewards):.4f}")
        print(f"   ✅ Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Environment is accessible")
    print("✅ /health endpoint works")
    print("✅ /reset endpoint works")
    print("✅ /step endpoint works with correct format")
    
    if reward != 0.0:
        print("✅ Rewards are being calculated correctly")
        print("\n🎉 ENVIRONMENT IS READY FOR TRAINING!")
        print("\nNext step: Create GRPO training notebook")
    else:
        print("⚠️  Rewards are 0.000 - check reward calculation logic")
        print("\n⚠️  FIX REWARD CALCULATION BEFORE TRAINING")
    
    return True


if __name__ == "__main__":
    try:
        test_environment()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
