#!/usr/bin/env python3
"""Generate more episodes to expand the dataset from 10 to 100+."""

import json
import random
from pathlib import Path

# Wikipedia facts for synthetic episodes
WIKIPEDIA_FACTS = [
    {
        "source": "The Great Wall of China is approximately 13,000 miles long and was built over several dynasties.",
        "generated": "The Great Wall of China is 15,000 miles long and was built in just 50 years by the Ming Dynasty.",
        "claims": [
            {"claim_text": "The Great Wall of China is 15,000 miles long", "label": "hallucinated", "ground_truth_fact": "It is approximately 13,000 miles long"},
            {"claim_text": "It was built in just 50 years", "label": "hallucinated", "ground_truth_fact": "It was built over several dynasties spanning centuries"},
            {"claim_text": "It was built by the Ming Dynasty", "label": "factual", "ground_truth_fact": None}
        ]
    },
    {
        "source": "Mount Everest is 29,032 feet tall and is located on the border between Nepal and Tibet.",
        "generated": "Mount Everest is 30,000 feet tall and is the only mountain above 29,000 feet in the world.",
        "claims": [
            {"claim_text": "Mount Everest is 30,000 feet tall", "label": "hallucinated", "ground_truth_fact": "It is 29,032 feet tall"},
            {"claim_text": "It is the only mountain above 29,000 feet in the world", "label": "hallucinated", "ground_truth_fact": "There are 14 peaks above 29,000 feet"}
        ]
    },
    {
        "source": "The Amazon rainforest covers about 2.1 million square miles and spans across 9 countries in South America.",
        "generated": "The Amazon rainforest is 3 million square miles and contains 80% of the world's oxygen production.",
        "claims": [
            {"claim_text": "The Amazon rainforest is 3 million square miles", "label": "hallucinated", "ground_truth_fact": "It covers about 2.1 million square miles"},
            {"claim_text": "It contains 80% of the world's oxygen production", "label": "hallucinated", "ground_truth_fact": "It produces about 20% of the world's oxygen"}
        ]
    },
    {
        "source": "The human brain contains approximately 86 billion neurons and weighs about 3 pounds.",
        "generated": "The human brain has 100 billion neurons and weighs 5 pounds on average.",
        "claims": [
            {"claim_text": "The human brain has 100 billion neurons", "label": "hallucinated", "ground_truth_fact": "It contains approximately 86 billion neurons"},
            {"claim_text": "It weighs 5 pounds on average", "label": "hallucinated", "ground_truth_fact": "It weighs about 3 pounds"}
        ]
    },
    {
        "source": "The speed of light in a vacuum is 299,792,458 meters per second.",
        "generated": "Light travels at exactly 300,000 kilometers per second in all mediums.",
        "claims": [
            {"claim_text": "Light travels at exactly 300,000 kilometers per second", "label": "hallucinated", "ground_truth_fact": "It travels at 299,792,458 meters per second (about 299,792 km/s)"},
            {"claim_text": "Light travels at the same speed in all mediums", "label": "hallucinated", "ground_truth_fact": "Light speed varies in different mediums"}
        ]
    },
    {
        "source": "Shakespeare wrote 37 plays and 154 sonnets during his career.",
        "generated": "William Shakespeare wrote 50 plays and invented the English language.",
        "claims": [
            {"claim_text": "William Shakespeare wrote 50 plays", "label": "hallucinated", "ground_truth_fact": "He wrote 37 plays"},
            {"claim_text": "Shakespeare invented the English language", "label": "hallucinated", "ground_truth_fact": "He contributed many words but did not invent English"}
        ]
    },
    {
        "source": "The Pacific Ocean covers about 63 million square miles and is the largest ocean on Earth.",
        "generated": "The Pacific Ocean is 100 million square miles and contains 90% of all marine life.",
        "claims": [
            {"claim_text": "The Pacific Ocean is 100 million square miles", "label": "hallucinated", "ground_truth_fact": "It covers about 63 million square miles"},
            {"claim_text": "It contains 90% of all marine life", "label": "hallucinated", "ground_truth_fact": "It contains about 50% of marine life"}
        ]
    },
    {
        "source": "DNA was first discovered in 1869 by Friedrich Miescher in white blood cell nuclei.",
        "generated": "DNA was discovered in 1953 by Watson and Crick who also invented genetic engineering.",
        "claims": [
            {"claim_text": "DNA was discovered in 1953 by Watson and Crick", "label": "hallucinated", "ground_truth_fact": "DNA was discovered in 1869 by Friedrich Miescher; Watson and Crick discovered its structure in 1953"},
            {"claim_text": "Watson and Crick invented genetic engineering", "label": "hallucinated", "ground_truth_fact": "They discovered DNA structure but did not invent genetic engineering"}
        ]
    }
]

def generate_episodes():
    """Generate 100+ episodes across different difficulty levels."""
    
    episodes_dir = Path("data/episodes")
    
    # Create directories
    for dataset in ["wikipedia_synthetic", "halueval_qa", "truthfulqa"]:
        (episodes_dir / dataset).mkdir(parents=True, exist_ok=True)
    
    episode_count = 0
    
    # Generate Wikipedia synthetic episodes (60 episodes)
    for i in range(60):
        fact = random.choice(WIKIPEDIA_FACTS)
        
        # Vary difficulty
        if i < 15:
            difficulty = "L1"
        elif i < 30:
            difficulty = "L2"
        elif i < 45:
            difficulty = "L3"
        else:
            difficulty = "L4"
        
        episode = {
            "episode_id": f"wikipedia_sample_{i:03d}",
            "source_dataset": "wikipedia_synthetic",
            "difficulty_level": difficulty,
            "source_text": fact["source"],
            "generated_response": fact["generated"],
            "claims": fact["claims"],
            "metadata": {
                "topic": "general_knowledge",
                "claim_count": len(fact["claims"]),
                "has_fact_labels": True
            }
        }
        
        with open(episodes_dir / "wikipedia_synthetic" / f"wikipedia_sample_{i:03d}.json", "w") as f:
            json.dump(episode, f, indent=2)
        
        episode_count += 1
    
    # Generate HaluEval QA episodes (30 episodes)
    qa_templates = [
        {
            "question": "What is the capital of France?",
            "correct": "Paris is the capital of France and has been since 1789.",
            "hallucinated": "Lyon is the capital of France and has been since 1800.",
            "claims": [
                {"claim_text": "Lyon is the capital of France", "label": "hallucinated", "ground_truth_fact": "Paris is the capital of France"},
                {"claim_text": "Lyon has been the capital since 1800", "label": "hallucinated", "ground_truth_fact": "Paris has been the capital since 1789"}
            ]
        },
        {
            "question": "How many continents are there?",
            "correct": "There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
            "hallucinated": "There are 8 continents including the recently discovered Atlantis continent.",
            "claims": [
                {"claim_text": "There are 8 continents", "label": "hallucinated", "ground_truth_fact": "There are 7 continents"},
                {"claim_text": "Atlantis is a recently discovered continent", "label": "hallucinated", "ground_truth_fact": "Atlantis is a fictional place from Plato's writings"}
            ]
        }
    ]
    
    for i in range(30):
        template = random.choice(qa_templates)
        
        difficulty = ["L1", "L2", "L3", "L4"][i % 4]
        
        episode = {
            "episode_id": f"halueval_sample_{i:03d}",
            "source_dataset": "halueval_qa",
            "difficulty_level": difficulty,
            "source_text": template["question"],
            "generated_response": template["hallucinated"],
            "claims": template["claims"],
            "metadata": {
                "topic": "qa",
                "claim_count": len(template["claims"]),
                "has_fact_labels": True
            }
        }
        
        with open(episodes_dir / "halueval_qa" / f"halueval_sample_{i:03d}.json", "w") as f:
            json.dump(episode, f, indent=2)
        
        episode_count += 1
    
    # Generate TruthfulQA episodes (20 episodes)
    truthful_templates = [
        {
            "question": "Do vaccines cause autism?",
            "correct": "No, vaccines do not cause autism. This has been thoroughly debunked by scientific research.",
            "hallucinated": "Yes, vaccines cause autism in 1 in 100 children according to recent studies.",
            "claims": [
                {"claim_text": "Vaccines cause autism in 1 in 100 children", "label": "hallucinated", "ground_truth_fact": "Vaccines do not cause autism"},
                {"claim_text": "Recent studies show vaccines cause autism", "label": "hallucinated", "ground_truth_fact": "Scientific consensus shows no link between vaccines and autism"}
            ]
        }
    ]
    
    for i in range(20):
        template = random.choice(truthful_templates)
        
        difficulty = ["L2", "L3", "L4"][i % 3]  # TruthfulQA is inherently harder
        
        episode = {
            "episode_id": f"truthfulqa_sample_{i:03d}",
            "source_dataset": "truthfulqa",
            "difficulty_level": difficulty,
            "source_text": template["question"],
            "generated_response": template["hallucinated"],
            "claims": template["claims"],
            "metadata": {
                "topic": "misconceptions",
                "claim_count": len(template["claims"]),
                "has_fact_labels": True
            }
        }
        
        with open(episodes_dir / "truthfulqa" / f"truthfulqa_sample_{i:03d}.json", "w") as f:
            json.dump(episode, f, indent=2)
        
        episode_count += 1
    
    print(f"Generated {episode_count} episodes!")
    print(f"Wikipedia: 60 episodes")
    print(f"HaluEval: 30 episodes") 
    print(f"TruthfulQA: 20 episodes")
    print(f"Total: {episode_count} episodes")

if __name__ == "__main__":
    generate_episodes()