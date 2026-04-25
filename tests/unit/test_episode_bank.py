"""Unit tests for EpisodeBank class."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.environment.episode_bank import EpisodeBank
from src.environment.core import Episode, Claim


class TestEpisodeBank:
    """Test suite for EpisodeBank functionality."""
    
    def test_load_episodes_from_json_files(self):
        """Test episode loading from JSON files."""
        bank = EpisodeBank()
        
        # Load from the sample data directory
        bank.load_episodes("data/episodes")
        
        # Verify episodes were loaded
        stats = bank.get_statistics()
        assert stats["episode_count"] >= 3, "Should load at least 3 sample episodes"
        
        # Verify episodes are accessible
        assert "halueval_qa_001" in bank.episodes
        assert "wikipedia_001" in bank.episodes
        assert "truthfulqa_001" in bank.episodes
    
    def test_load_episodes_validates_structure(self):
        """Test episode structure validation during loading."""
        bank = EpisodeBank()
        
        with TemporaryDirectory() as tmpdir:
            # Create invalid episode (missing required field)
            invalid_episode = {
                "episode_id": "invalid_001",
                "source_dataset": "test",
                # Missing other required fields
            }
            
            invalid_path = Path(tmpdir) / "invalid.json"
            with open(invalid_path, 'w') as f:
                json.dump(invalid_episode, f)
            
            # Should raise ValueError for missing fields
            with pytest.raises(ValueError, match="No valid episodes loaded"):
                bank.load_episodes(tmpdir)
    
    def test_difficulty_assignment_heuristics(self):
        """Test difficulty assignment based on claim count and complexity."""
        bank = EpisodeBank()
        
        # L1: Simple episode with 3 claims, low hallucination rate (<30%)
        episode_l1 = Episode(
            episode_id="test_l1",
            source_dataset="test",
            difficulty_level="L1",  # Use valid level for creation
            source_text="Test source",
            generated_response="Test response",
            claims=[
                Claim("Claim 1", "factual", None),
                Claim("Claim 2", "factual", None),
                Claim("Claim 3", "factual", None)
            ],
            metadata={}
        )
        assert bank._assign_difficulty(episode_l1) == "L1"
        
        # L2: Moderate episode with 5 claims
        episode_l2 = Episode(
            episode_id="test_l2",
            source_dataset="test",
            difficulty_level="L1",  # Use valid level for creation
            source_text="Test source",
            generated_response="Test response",
            claims=[
                Claim(f"Claim {i}", "factual" if i % 2 == 0 else "hallucinated", None)
                for i in range(5)
            ],
            metadata={}
        )
        assert bank._assign_difficulty(episode_l2) == "L2"
        
        # L2: Episode with moderate hallucination rate (>30%)
        episode_l2_hallucination = Episode(
            episode_id="test_l2_hallucination",
            source_dataset="test",
            difficulty_level="L1",  # Use valid level for creation
            source_text="Test source",
            generated_response="Test response",
            claims=[
                Claim("Claim 1", "factual", None),
                Claim("Claim 2", "factual", None),
                Claim("Claim 3", "hallucinated", "Truth 3")
            ],
            metadata={}
        )
        # 33% hallucination rate should be L2
        assert bank._assign_difficulty(episode_l2_hallucination) == "L2"
        
        # L3: Hard episode with 7 claims
        episode_l3 = Episode(
            episode_id="test_l3",
            source_dataset="test",
            difficulty_level="L1",  # Use valid level for creation
            source_text="Test source",
            generated_response="Test response",
            claims=[
                Claim(f"Claim {i}", "factual", None)
                for i in range(7)
            ],
            metadata={}
        )
        assert bank._assign_difficulty(episode_l3) == "L3"
        
        # L4: Expert episode with 10 claims
        episode_l4 = Episode(
            episode_id="test_l4",
            source_dataset="test",
            difficulty_level="L1",  # Use valid level for creation
            source_text="Test source",
            generated_response="Test response",
            claims=[
                Claim(f"Claim {i}", "factual", None)
                for i in range(10)
            ],
            metadata={}
        )
        assert bank._assign_difficulty(episode_l4) == "L4"
        
        # L4: High ambiguity (many unverifiable claims)
        episode_l4_ambiguous = Episode(
            episode_id="test_l4_ambiguous",
            source_dataset="test",
            difficulty_level="L1",  # Use valid level for creation
            source_text="Test source",
            generated_response="Test response",
            claims=[
                Claim("Claim 1", "unverifiable", None),
                Claim("Claim 2", "unverifiable", None),
                Claim("Claim 3", "factual", None),
            ],
            metadata={}
        )
        assert bank._assign_difficulty(episode_l4_ambiguous) == "L4"
    
    def test_sample_episode_with_difficulty_filtering(self):
        """Test sampling with different enabled difficulty levels."""
        bank = EpisodeBank()
        bank.load_episodes("data/episodes")
        
        # Sample from L1 only
        episode = bank.sample_episode(["L1"])
        assert episode.difficulty_level == "L1"
        
        # Sample from L2 only
        episode = bank.sample_episode(["L2"])
        assert episode.difficulty_level == "L2"
        
        # Sample from multiple levels
        episode = bank.sample_episode(["L1", "L2"])
        assert episode.difficulty_level in ["L1", "L2"]
    
    def test_sample_episode_raises_on_invalid_difficulty(self):
        """Test that sampling raises error for invalid difficulty levels."""
        bank = EpisodeBank()
        bank.load_episodes("data/episodes")
        
        with pytest.raises(ValueError, match="Invalid difficulty levels"):
            bank.sample_episode(["L5", "L6"])
    
    def test_sample_episode_raises_on_no_episodes(self):
        """Test that sampling raises error when no episodes available."""
        bank = EpisodeBank()
        bank.load_episodes("data/episodes")
        
        # Try to sample from a level that might not have episodes
        # This depends on the difficulty assignment, so we'll create a scenario
        with TemporaryDirectory() as tmpdir:
            bank_empty = EpisodeBank()
            
            # Create episode that will be assigned L1
            episode_data = {
                "episode_id": "test_001",
                "source_dataset": "test",
                "source_text": "Test",
                "generated_response": "Test",
                "claims": [
                    {"claim_text": "Test", "label": "factual", "ground_truth_fact": None}
                ],
                "metadata": {}
            }
            
            episode_path = Path(tmpdir) / "test.json"
            with open(episode_path, 'w') as f:
                json.dump(episode_data, f)
            
            bank_empty.load_episodes(tmpdir)
            
            # Try to sample from L4 when only L1 exists
            with pytest.raises(ValueError, match="No episodes available"):
                bank_empty.sample_episode(["L4"])
    
    def test_get_episode_by_id(self):
        """Test specific episode retrieval."""
        bank = EpisodeBank()
        bank.load_episodes("data/episodes")
        
        # Retrieve existing episode
        episode = bank.get_episode_by_id("halueval_qa_001")
        assert episode.episode_id == "halueval_qa_001"
        assert episode.source_dataset == "halueval_qa"
        
        # Try to retrieve non-existent episode
        with pytest.raises(KeyError, match="Episode not found"):
            bank.get_episode_by_id("nonexistent_id")
    
    def test_get_statistics(self):
        """Test episode count and difficulty distribution statistics."""
        bank = EpisodeBank()
        bank.load_episodes("data/episodes")
        
        stats = bank.get_statistics()
        
        # Verify structure
        assert "episode_count" in stats
        assert "difficulty_distribution" in stats
        
        # Verify counts
        assert stats["episode_count"] >= 3
        assert isinstance(stats["difficulty_distribution"], dict)
        
        # Verify all difficulty levels are present
        for level in ["L1", "L2", "L3", "L4"]:
            assert level in stats["difficulty_distribution"]
        
        # Verify sum of distribution equals total count
        total_distributed = sum(stats["difficulty_distribution"].values())
        assert total_distributed == stats["episode_count"]
    
    def test_load_episodes_handles_missing_directory(self):
        """Test that loading raises error for missing directory."""
        bank = EpisodeBank()
        
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            bank.load_episodes("nonexistent/directory")
    
    def test_load_episodes_handles_empty_directory(self):
        """Test that loading raises error for directory with no JSON files."""
        bank = EpisodeBank()
        
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No JSON files found"):
                bank.load_episodes(tmpdir)
    
    def test_difficulty_assignment_respects_metadata(self):
        """Test that difficulty assignment uses metadata if available."""
        bank = EpisodeBank()
        
        # Episode with difficulty in metadata
        episode = Episode(
            episode_id="test_metadata",
            source_dataset="test",
            difficulty_level="L1",  # Use valid level for creation
            source_text="Test",
            generated_response="Test",
            claims=[Claim("Test", "factual", None)],
            metadata={"difficulty_level": "L3"}
        )
        
        # Should use metadata value
        assert bank._assign_difficulty(episode) == "L3"
    
    def test_episode_bank_tracks_by_difficulty(self):
        """Test that episodes are correctly tracked by difficulty level."""
        bank = EpisodeBank()
        bank.load_episodes("data/episodes")
        
        # Verify episodes_by_difficulty is populated
        for level in ["L1", "L2", "L3", "L4"]:
            episode_ids = bank.episodes_by_difficulty[level]
            
            # Verify all episodes in this level have correct difficulty
            for episode_id in episode_ids:
                episode = bank.episodes[episode_id]
                assert episode.difficulty_level == level
