"""Unit tests for API Pydantic models."""

import pytest
from pydantic import ValidationError
from src.api.models import (
    DetectedClaim, DetectionOutput, Observation, Action, StepResult
)


class TestDetectedClaim:
    """Tests for the DetectedClaim model."""
    
    def test_detected_claim_creation_with_valid_label(self):
        """Test creating a detected claim with valid label."""
        claim = DetectedClaim(
            claim_text="The Eiffel Tower is in Paris",
            label="factual",
            reason="Matches historical records",
            corrected_fact=None
        )
        assert claim.claim_text == "The Eiffel Tower is in Paris"
        assert claim.label == "factual"
        assert claim.reason == "Matches historical records"
        assert claim.corrected_fact is None
    
    def test_detected_claim_with_hallucinated_label(self):
        """Test creating a hallucinated claim with correction."""
        claim = DetectedClaim(
            claim_text="The Eiffel Tower is 400 meters tall",
            label="hallucinated",
            reason="Actual height is 330 meters",
            corrected_fact="The Eiffel Tower is 330 meters tall"
        )
        assert claim.label == "hallucinated"
        assert claim.corrected_fact == "The Eiffel Tower is 330 meters tall"
    
    def test_detected_claim_with_invalid_label_raises_error(self):
        """Test that invalid label raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DetectedClaim(
                claim_text="Test",
                label="invalid",
                reason="Test",
                corrected_fact=None
            )
        assert "Invalid label 'invalid'" in str(exc_info.value)
    
    def test_all_valid_labels_accepted(self):
        """Test that all valid labels are accepted."""
        valid_labels = ["factual", "hallucinated", "unverifiable"]
        for label in valid_labels:
            claim = DetectedClaim(
                claim_text="Test claim",
                label=label,
                reason="Test reason",
                corrected_fact=None
            )
            assert claim.label == label


class TestDetectionOutput:
    """Tests for the DetectionOutput model."""
    
    def test_detection_output_with_multiple_claims(self):
        """Test creating detection output with multiple claims."""
        claims = [
            DetectedClaim(
                claim_text="Claim 1",
                label="factual",
                reason="Reason 1",
                corrected_fact=None
            ),
            DetectedClaim(
                claim_text="Claim 2",
                label="hallucinated",
                reason="Reason 2",
                corrected_fact="Correction 2"
            )
        ]
        output = DetectionOutput(detected_claims=claims)
        assert len(output.detected_claims) == 2
        assert output.detected_claims[0].label == "factual"
        assert output.detected_claims[1].label == "hallucinated"
    
    def test_detection_output_with_empty_list(self):
        """Test creating detection output with empty claims list."""
        output = DetectionOutput(detected_claims=[])
        assert len(output.detected_claims) == 0


class TestObservation:
    """Tests for the Observation model."""
    
    def test_observation_creation(self):
        """Test creating an observation."""
        obs = Observation(
            generated_text="The Eiffel Tower is in Paris.",
            task_instruction="Identify factual claims and label them."
        )
        assert obs.generated_text == "The Eiffel Tower is in Paris."
        assert "Identify factual claims" in obs.task_instruction


class TestAction:
    """Tests for the Action model."""
    
    def test_action_creation(self):
        """Test creating an action with detection output."""
        detection = DetectionOutput(
            detected_claims=[
                DetectedClaim(
                    claim_text="Test",
                    label="factual",
                    reason="Test reason",
                    corrected_fact=None
                )
            ]
        )
        action = Action(detection_output=detection)
        assert len(action.detection_output.detected_claims) == 1


class TestStepResult:
    """Tests for the StepResult model."""
    
    def test_step_result_creation(self):
        """Test creating a step result."""
        obs = Observation(
            generated_text="",
            task_instruction=""
        )
        result = StepResult(
            observation=obs,
            reward=4.5,
            done=True,
            info={
                "episode_id": "ep_001",
                "difficulty_level": "L1",
                "precision": 0.85,
                "recall": 0.90
            }
        )
        assert result.reward == 4.5
        assert result.done is True
        assert result.info["episode_id"] == "ep_001"
        assert result.info["precision"] == 0.85
    
    def test_step_result_with_empty_info(self):
        """Test creating a step result with empty info dict."""
        obs = Observation(
            generated_text="Test",
            task_instruction="Test"
        )
        result = StepResult(
            observation=obs,
            reward=0.0,
            done=False
        )
        assert result.info == {}


class TestModelSerialization:
    """Tests for JSON serialization/deserialization."""
    
    def test_detected_claim_json_serialization(self):
        """Test that DetectedClaim can be serialized to JSON."""
        claim = DetectedClaim(
            claim_text="Test",
            label="factual",
            reason="Test reason",
            corrected_fact=None
        )
        json_data = claim.model_dump()
        assert json_data["claim_text"] == "Test"
        assert json_data["label"] == "factual"
    
    def test_detection_output_json_deserialization(self):
        """Test that DetectionOutput can be deserialized from JSON."""
        json_data = {
            "detected_claims": [
                {
                    "claim_text": "Test",
                    "label": "factual",
                    "reason": "Test reason",
                    "corrected_fact": None
                }
            ]
        }
        output = DetectionOutput(**json_data)
        assert len(output.detected_claims) == 1
        assert output.detected_claims[0].claim_text == "Test"
