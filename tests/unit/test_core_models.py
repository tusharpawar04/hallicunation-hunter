"""Unit tests for core data models."""

import pytest
from src.environment.core import Claim, Episode


class TestClaim:
    """Tests for the Claim dataclass."""
    
    def test_claim_creation_with_valid_label(self):
        """Test creating a claim with a valid label."""
        claim = Claim(
            claim_text="The Eiffel Tower is in Paris",
            label="factual",
            ground_truth_fact=None
        )
        assert claim.claim_text == "The Eiffel Tower is in Paris"
        assert claim.label == "factual"
        assert claim.ground_truth_fact is None
    
    def test_claim_with_hallucinated_label(self):
        """Test creating a hallucinated claim with correction."""
        claim = Claim(
            claim_text="The Eiffel Tower is 400 meters tall",
            label="hallucinated",
            ground_truth_fact="The Eiffel Tower is 330 meters tall"
        )
        assert claim.label == "hallucinated"
        assert claim.ground_truth_fact == "The Eiffel Tower is 330 meters tall"
    
    def test_claim_with_unverifiable_label(self):
        """Test creating an unverifiable claim."""
        claim = Claim(
            claim_text="The Eiffel Tower is beautiful",
            label="unverifiable",
            ground_truth_fact=None
        )
        assert claim.label == "unverifiable"
    
    def test_claim_with_invalid_label_raises_error(self):
        """Test that invalid label raises ValueError."""
        with pytest.raises(ValueError, match="Invalid label 'invalid'"):
            Claim(
                claim_text="Some text",
                label="invalid",
                ground_truth_fact=None
            )
    
    def test_all_valid_labels_accepted(self):
        """Test that all valid labels are accepted."""
        valid_labels = ["factual", "hallucinated", "unverifiable"]
        for label in valid_labels:
            claim = Claim(
                claim_text="Test claim",
                label=label,
                ground_truth_fact=None
            )
            assert claim.label == label


class TestEpisode:
    """Tests for the Episode dataclass."""
    
    def test_episode_creation_with_valid_difficulty(self):
        """Test creating an episode with valid difficulty level."""
        claims = [
            Claim("The Eiffel Tower is in Paris", "factual", None),
            Claim("It was built in 1889", "factual", None)
        ]
        episode = Episode(
            episode_id="ep_001",
            source_dataset="halueval_qa",
            difficulty_level="L1",
            source_text="The Eiffel Tower is a landmark in Paris.",
            generated_response="The Eiffel Tower is in Paris. It was built in 1889.",
            claims=claims,
            metadata={"topic": "landmarks"}
        )
        assert episode.episode_id == "ep_001"
        assert episode.source_dataset == "halueval_qa"
        assert episode.difficulty_level == "L1"
        assert len(episode.claims) == 2
        assert episode.metadata["topic"] == "landmarks"
    
    def test_episode_with_invalid_difficulty_raises_error(self):
        """Test that invalid difficulty level raises ValueError."""
        claims = [Claim("Test", "factual", None)]
        with pytest.raises(ValueError, match="Invalid difficulty_level 'L5'"):
            Episode(
                episode_id="ep_001",
                source_dataset="test",
                difficulty_level="L5",
                source_text="Test",
                generated_response="Test",
                claims=claims
            )
    
    def test_all_valid_difficulty_levels_accepted(self):
        """Test that all valid difficulty levels are accepted."""
        valid_levels = ["L1", "L2", "L3", "L4"]
        claims = [Claim("Test", "factual", None)]
        for level in valid_levels:
            episode = Episode(
                episode_id=f"ep_{level}",
                source_dataset="test",
                difficulty_level=level,
                source_text="Test",
                generated_response="Test",
                claims=claims
            )
            assert episode.difficulty_level == level
    
    def test_episode_with_empty_metadata(self):
        """Test creating an episode without metadata."""
        claims = [Claim("Test", "factual", None)]
        episode = Episode(
            episode_id="ep_001",
            source_dataset="test",
            difficulty_level="L1",
            source_text="Test",
            generated_response="Test",
            claims=claims
        )
        assert episode.metadata == {}
    
    def test_episode_with_multiple_claim_types(self):
        """Test episode with factual, hallucinated, and unverifiable claims."""
        claims = [
            Claim("Paris is in France", "factual", None),
            Claim("Paris has 20 million people", "hallucinated", 
                  "Paris has about 2.2 million people"),
            Claim("Paris is the most beautiful city", "unverifiable", None)
        ]
        episode = Episode(
            episode_id="ep_mixed",
            source_dataset="wikipedia_synthetic",
            difficulty_level="L2",
            source_text="Paris is the capital of France.",
            generated_response="Paris is in France. It has 20 million people. "
                             "It is the most beautiful city.",
            claims=claims,
            metadata={"claim_count": 3, "hallucination_rate": 0.33}
        )
        assert len(episode.claims) == 3
        assert episode.claims[0].label == "factual"
        assert episode.claims[1].label == "hallucinated"
        assert episode.claims[2].label == "unverifiable"
