"""Test script to verify episode bank."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.episode_bank import EpisodeBank

def main():
    print("Testing Episode Bank...")
    
    eb = EpisodeBank()
    eb.load_episodes('data/episodes')
    
    stats = eb.get_statistics()
    
    print(f"\nTotal episodes: {stats['episode_count']}")
    print(f"Difficulty distribution: {stats['difficulty_distribution']}")
    
    # Sample an episode
    if stats['episode_count'] > 0:
        episode = eb.sample_episode(['L1', 'L2', 'L3', 'L4'])
        print(f"\nSample episode:")
        print(f"  ID: {episode.episode_id}")
        print(f"  Source: {episode.source_dataset}")
        print(f"  Difficulty: {episode.difficulty_level}")
        print(f"  Claims: {len(episode.claims)}")
    
    print("\n✓ Episode bank loaded successfully!")

if __name__ == "__main__":
    main()
