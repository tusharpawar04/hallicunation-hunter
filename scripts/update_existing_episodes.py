"""
Script to update existing episode files with difficulty levels.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.core import Episode, Claim
from src.environment.episode_bank import EpisodeBank


def update_episode_file(file_path: str, episode_bank: EpisodeBank):
    """Update a single episode file with difficulty level."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if difficulty_level already exists
    if "difficulty_level" in data and data["difficulty_level"]:
        return False  # No update needed
    
    # Create Episode object
    claims = [
        Claim(
            claim_text=c["claim_text"],
            label=c["label"],
            ground_truth_fact=c.get("ground_truth_fact")
        )
        for c in data["claims"]
    ]
    
    episode = Episode(
        episode_id=data["episode_id"],
        source_dataset=data["source_dataset"],
        difficulty_level="L1",  # Placeholder
        source_text=data["source_text"],
        generated_response=data["generated_response"],
        claims=claims,
        metadata=data.get("metadata", {})
    )
    
    # Assign difficulty level
    difficulty_level = episode_bank._assign_difficulty(episode)
    
    # Update data
    data["difficulty_level"] = difficulty_level
    
    # Save back
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return True


def main():
    """Main function to update all episode files."""
    print("Updating existing episode files with difficulty levels...")
    
    data_dir = Path(__file__).parent.parent / "data" / "episodes"
    episode_bank = EpisodeBank()
    
    updated_count = 0
    total_count = 0
    
    # Walk through all episode directories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                total_count += 1
                
                try:
                    if update_episode_file(file_path, episode_bank):
                        updated_count += 1
                        print(f"  Updated: {file_path}")
                except Exception as e:
                    print(f"  Error updating {file_path}: {e}")
    
    print(f"\nUpdated {updated_count} out of {total_count} episode files.")


if __name__ == "__main__":
    main()
