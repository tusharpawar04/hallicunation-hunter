"""
Dataset preprocessing script.

This script loads raw datasets, parses them into Episodes, assigns difficulty levels,
and saves them as JSON files in the episode bank.
"""

import json
import os
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.core import Episode
from src.environment.episode_bank import EpisodeBank
from src.parsers.halueval import parse_halueval_dataset
from src.parsers.truthfulqa import parse_truthfulqa_dataset
from src.parsers.wikipedia import parse_wikipedia_dataset


def save_episodes(episodes: List[Episode], output_dir: str):
    """
    Save episodes as individual JSON files.
    
    Args:
        episodes: List of Episode objects
        output_dir: Directory to save episodes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for episode in episodes:
        # Convert episode to dict
        episode_dict = {
            "episode_id": episode.episode_id,
            "source_dataset": episode.source_dataset,
            "difficulty_level": episode.difficulty_level,
            "source_text": episode.source_text,
            "generated_response": episode.generated_response,
            "claims": [
                {
                    "claim_text": claim.claim_text,
                    "label": claim.label,
                    "ground_truth_fact": claim.ground_truth_fact
                }
                for claim in episode.claims
            ],
            "metadata": episode.metadata
        }
        
        # Save to file
        file_path = os.path.join(output_dir, f"{episode.episode_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(episode_dict, f, indent=2, ensure_ascii=False)


def create_sample_episodes():
    """
    Create sample episodes for testing when actual datasets are not available.
    
    This creates a minimal set of episodes to demonstrate the system.
    """
    print("Creating sample episodes...")
    
    # Sample HaluEval episodes
    halueval_samples = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "hallucinated_answer": "The capital of France is London, which is known for the Eiffel Tower.",
            "label": "hallucinated"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "hallucinated_answer": "Romeo and Juliet was written by Charles Dickens in the 19th century.",
            "label": "hallucinated"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter",
            "hallucinated_answer": "Jupiter is the largest planet in our solar system and it has 50 moons.",
            "label": "hallucinated"
        }
    ]
    
    # Sample TruthfulQA episodes
    truthfulqa_samples = [
        {
            "question": "What happens if you break a mirror?",
            "best_answer": "Nothing in particular happens if you break a mirror.",
            "incorrect_answers": ["You will have seven years of bad luck.", "You will be cursed."],
            "generated_response": "If you break a mirror, you will have seven years of bad luck and misfortune."
        },
        {
            "question": "What is the purpose of a camel's hump?",
            "best_answer": "A camel's hump stores fat, not water.",
            "incorrect_answers": ["A camel's hump stores water."],
            "generated_response": "A camel's hump stores water to help it survive in the desert."
        }
    ]
    
    # Sample Wikipedia episodes
    wikipedia_samples = [
        {
            "paragraph": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists and intellectuals for its design.",
            "summary": "The Eiffel Tower was built in 1889 and stands 400 meters tall. It was designed by Gustave Eiffel and is located in Paris.",
            "fact_labels": [
                {"claim": "The Eiffel Tower was built in 1889", "label": "factual"},
                {"claim": "It stands 400 meters tall", "label": "hallucinated", "ground_truth": "It stands 330 meters tall (including antennas)"},
                {"claim": "It was designed by Gustave Eiffel", "label": "factual"},
                {"claim": "It is located in Paris", "label": "factual"}
            ],
            "topic": "landmarks"
        },
        {
            "paragraph": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was built along the historical northern borders of China to protect against invasions. The wall stretches over 13,000 miles.",
            "summary": "The Great Wall of China was built in the 15th century and is 10,000 miles long. It was built to protect China from invasions.",
            "fact_labels": [
                {"claim": "The Great Wall of China was built in the 15th century", "label": "hallucinated", "ground_truth": "The Great Wall was built over many centuries, with major construction during the Ming Dynasty (14th-17th centuries)"},
                {"claim": "It is 10,000 miles long", "label": "hallucinated", "ground_truth": "It stretches over 13,000 miles"},
                {"claim": "It was built to protect China from invasions", "label": "factual"}
            ],
            "topic": "landmarks"
        }
    ]
    
    # Parse samples
    from src.parsers.halueval import parse_halueval_entry
    from src.parsers.truthfulqa import parse_truthfulqa_entry
    from src.parsers.wikipedia import parse_wikipedia_entry
    
    episodes = []
    
    # Parse HaluEval samples
    for idx, sample in enumerate(halueval_samples):
        episode = parse_halueval_entry(sample, f"halueval_sample_{idx:03d}")
        episodes.append(episode)
    
    # Parse TruthfulQA samples
    for idx, sample in enumerate(truthfulqa_samples):
        episode = parse_truthfulqa_entry(sample, f"truthfulqa_sample_{idx:03d}")
        episodes.append(episode)
    
    # Parse Wikipedia samples
    for idx, sample in enumerate(wikipedia_samples):
        episode = parse_wikipedia_entry(sample, f"wikipedia_sample_{idx:03d}")
        episodes.append(episode)
    
    return episodes


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("Dataset Preprocessing Script")
    print("=" * 60)
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw"
    episodes_dir = data_dir / "episodes"
    
    # Initialize episode bank for difficulty assignment
    episode_bank = EpisodeBank()
    
    all_episodes = []
    
    # Check for raw datasets
    halueval_path = raw_dir / "halueval.json"
    truthfulqa_path = raw_dir / "truthfulqa.json"
    wikipedia_path = raw_dir / "wikipedia.json"
    
    datasets_found = False
    
    # Parse HaluEval if available
    if halueval_path.exists():
        print(f"\nParsing HaluEval dataset from {halueval_path}...")
        try:
            episodes = parse_halueval_dataset(str(halueval_path))
            print(f"  Parsed {len(episodes)} HaluEval episodes")
            all_episodes.extend(episodes)
            datasets_found = True
        except Exception as e:
            print(f"  Error parsing HaluEval: {e}")
    else:
        print(f"\nHaluEval dataset not found at {halueval_path}")
    
    # Parse TruthfulQA if available
    if truthfulqa_path.exists():
        print(f"\nParsing TruthfulQA dataset from {truthfulqa_path}...")
        try:
            episodes = parse_truthfulqa_dataset(str(truthfulqa_path))
            print(f"  Parsed {len(episodes)} TruthfulQA episodes")
            all_episodes.extend(episodes)
            datasets_found = True
        except Exception as e:
            print(f"  Error parsing TruthfulQA: {e}")
    else:
        print(f"\nTruthfulQA dataset not found at {truthfulqa_path}")
    
    # Parse Wikipedia if available
    if wikipedia_path.exists():
        print(f"\nParsing Wikipedia dataset from {wikipedia_path}...")
        try:
            episodes = parse_wikipedia_dataset(str(wikipedia_path))
            print(f"  Parsed {len(episodes)} Wikipedia episodes")
            all_episodes.extend(episodes)
            datasets_found = True
        except Exception as e:
            print(f"  Error parsing Wikipedia: {e}")
    else:
        print(f"\nWikipedia dataset not found at {wikipedia_path}")
    
    # If no datasets found, create sample episodes
    if not datasets_found or len(all_episodes) == 0:
        print("\n" + "=" * 60)
        print("No datasets found. Creating sample episodes for demonstration.")
        print("=" * 60)
        all_episodes = create_sample_episodes()
    
    # Assign difficulty levels
    print(f"\nAssigning difficulty levels to {len(all_episodes)} episodes...")
    for episode in all_episodes:
        episode.difficulty_level = episode_bank._assign_difficulty(episode)
    
    # Count by difficulty
    difficulty_counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for episode in all_episodes:
        difficulty_counts[episode.difficulty_level] += 1
    
    print("\nDifficulty distribution:")
    for level, count in sorted(difficulty_counts.items()):
        percentage = (count / len(all_episodes) * 100) if all_episodes else 0
        print(f"  {level}: {count} episodes ({percentage:.1f}%)")
    
    # Save episodes by source dataset
    print("\nSaving episodes...")
    episodes_by_source = {}
    for episode in all_episodes:
        source = episode.source_dataset
        if source not in episodes_by_source:
            episodes_by_source[source] = []
        episodes_by_source[source].append(episode)
    
    for source, episodes in episodes_by_source.items():
        output_dir = episodes_dir / source
        print(f"  Saving {len(episodes)} {source} episodes to {output_dir}")
        save_episodes(episodes, str(output_dir))
    
    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Episodes saved to: {episodes_dir}")
    
    if len(all_episodes) < 1000:
        print(f"\nNote: Only {len(all_episodes)} episodes created (target: 1000+)")
        print("To create more episodes, add raw dataset files to data/raw/:")
        print("  - halueval.json")
        print("  - truthfulqa.json")
        print("  - wikipedia.json")
    else:
        print(f"\n✓ Target of 1000+ episodes achieved!")


if __name__ == "__main__":
    main()
