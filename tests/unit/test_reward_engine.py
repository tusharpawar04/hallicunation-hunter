"""Unit tests for RewardEngine class."""

import pytest
from src.environment.reward import RewardEngine
from src.environment.core import Claim
from src.api.models import DetectedClaim, DetectionOutput


class TestRewardEngine:
    """Test suite for RewardEngine functionality."""
    
    @pytest.fixture
    def reward_engine(self):
        """Create a RewardEngine instance for testing."""
        return RewardEngine()
    
    def _create_detected_claim(self, claim_text: str, label: str, reason: str, corrected_fact: str = None) -> DetectedClaim:
        """Helper to create DetectedClaim with keyword arguments."""
        return DetectedClaim(
            claim_text=claim_text,
            label=label,
            reason=reason,
            corrected_fact=corrected_fact
        )
    
    def test_base_reward_true_positive(self, reward_engine):
        """Test base reward for correctly identified hallucination (TP)."""
        # Ground truth: one hallucinated claim
        ground_truth = [
            Claim("The Eiffel Tower is 400 meters tall", "hallucinated", "The Eiffel Tower is 330 meters tall")
        ]
        
        # Detection: correctly identified as hallucinated
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="The Eiffel Tower is 400 meters tall",
                label="hallucinated",
                reason="Incorrect height",
                corrected_fact="The Eiffel Tower is 330 meters tall"
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get TP reward: +3.0
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0
        assert metrics["base_reward"] == 3.0
    
    def test_base_reward_false_positive(self, reward_engine):
        """Test base reward for incorrectly flagged factual claim (FP)."""
        # Ground truth: one factual claim
        ground_truth = [
            Claim("The Eiffel Tower is in Paris", "factual", None)
        ]
        
        # Detection: incorrectly flagged as hallucinated
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="The Eiffel Tower is in Paris",
                label="hallucinated",
                reason="Wrong",
                corrected_fact="Something else"
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get FP penalty: -2.0
        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 0
        assert metrics["base_reward"] == -2.0
    
    def test_base_reward_false_negative(self, reward_engine):
        """Test base reward for missed hallucination (FN)."""
        # Ground truth: one hallucinated claim
        ground_truth = [
            Claim("The Eiffel Tower is 400 meters tall", "hallucinated", "The Eiffel Tower is 330 meters tall")
        ]
        
        # Detection: missed, labeled as factual
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="The Eiffel Tower is 400 meters tall",
                label="factual",
                reason="Seems correct",
                corrected_fact=None
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get FN penalty: -1.5
        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 1
        assert metrics["base_reward"] == -1.5
    
    def test_base_reward_true_negative(self, reward_engine):
        """Test base reward for correctly identified factual claim (TN)."""
        # Ground truth: one factual claim
        ground_truth = [
            Claim("The Eiffel Tower is in Paris", "factual", None)
        ]
        
        # Detection: correctly identified as factual
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="The Eiffel Tower is in Paris",
                label="factual",
                reason="Correct",
                corrected_fact=None
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get TN reward: +0.5
        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0
        assert metrics["true_negatives"] == 1
        assert metrics["base_reward"] == 0.5
    
    def test_base_reward_mixed_confusion_matrix(self, reward_engine):
        """Test base reward with mixed TP, FP, FN, TN."""
        # Ground truth: 2 hallucinated, 2 factual
        ground_truth = [
            Claim("Claim 1 is wrong", "hallucinated", "Claim 1 is correct"),
            Claim("Claim 2 is wrong", "hallucinated", "Claim 2 is correct"),
            Claim("Claim 3 is right", "factual", None),
            Claim("Claim 4 is right", "factual", None)
        ]
        
        # Detection: 1 TP, 1 FP, 1 FN, 1 TN
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="Claim 1 is wrong", label="hallucinated", reason="Detected", corrected_fact="Claim 1 is correct"),  # TP
            DetectedClaim(claim_text="Claim 2 is wrong", label="factual", reason="Missed", corrected_fact=None),  # FN
            DetectedClaim(claim_text="Claim 3 is right", label="hallucinated", reason="Wrong", corrected_fact="Something"),  # FP
            DetectedClaim(claim_text="Claim 4 is right", label="factual", reason="Correct", corrected_fact=None)  # TN
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Expected: 1*3.0 + 1*(-2.0) + 1*(-1.5) + 1*0.5 = 3.0 - 2.0 - 1.5 + 0.5 = 0.0
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["base_reward"] == 0.0
    
    def test_correction_bonus_perfect_match(self, reward_engine):
        """Test correction bonus with perfect keyword overlap."""
        # Ground truth
        ground_truth = [
            Claim("Wrong fact", "hallucinated", "The Eiffel Tower is 330 meters tall")
        ]
        
        # Detection with perfect correction
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="Wrong fact",
                label="hallucinated",
                reason="Corrected",
                corrected_fact="The Eiffel Tower is 330 meters tall"
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get high correction bonus (close to 1.0 after stopword removal)
        assert metrics["correction_bonus"] > 0.5
        assert metrics["correction_bonus"] <= 1.0
    
    def test_correction_bonus_partial_match(self, reward_engine):
        """Test correction bonus with partial keyword overlap."""
        # Ground truth
        ground_truth = [
            Claim("Wrong fact", "hallucinated", "The Eiffel Tower is 330 meters tall")
        ]
        
        # Detection with partial correction
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="Wrong fact",
                label="hallucinated",
                reason="Corrected",
                corrected_fact="The tower is 330 meters"
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get moderate correction bonus
        assert 0.0 < metrics["correction_bonus"] < 1.0
    
    def test_correction_bonus_no_match(self, reward_engine):
        """Test correction bonus with no keyword overlap."""
        # Ground truth
        ground_truth = [
            Claim("Wrong fact", "hallucinated", "The Eiffel Tower is 330 meters tall")
        ]
        
        # Detection with completely wrong correction
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="Wrong fact",
                label="hallucinated",
                reason="Corrected",
                corrected_fact="Something completely different"
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get low or zero correction bonus
        assert metrics["correction_bonus"] < 0.5
    
    def test_calibration_bonus_awarded(self, reward_engine):
        """Test calibration bonus when precision and recall > 0.6."""
        # Ground truth: 3 hallucinated, 3 factual
        ground_truth = [
            Claim("H1", "hallucinated", "Correct 1"),
            Claim("H2", "hallucinated", "Correct 2"),
            Claim("H3", "hallucinated", "Correct 3"),
            Claim("F1", "factual", None),
            Claim("F2", "factual", None),
            Claim("F3", "factual", None)
        ]
        
        # Detection: 2 TP, 1 FN, 0 FP, 3 TN
        # Precision = 2/(2+0) = 1.0 > 0.6
        # Recall = 2/(2+1) = 0.67 > 0.6
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="Correct 1"),  # TP
            DetectedClaim(claim_text="H2", label="hallucinated", reason="Found", corrected_fact="Correct 2"),  # TP
            DetectedClaim(claim_text="H3", label="factual", reason="Missed", corrected_fact=None),  # FN
            DetectedClaim(claim_text="F1", label="factual", reason="Correct", corrected_fact=None),  # TN
            DetectedClaim(claim_text="F2", label="factual", reason="Correct", corrected_fact=None),  # TN
            DetectedClaim(claim_text="F3", label="factual", reason="Correct", corrected_fact=None)   # TN
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get calibration bonus
        assert metrics["precision"] > 0.6
        assert metrics["recall"] > 0.6
        assert metrics["calibration_bonus"] == 1.0
    
    def test_calibration_bonus_not_awarded_low_precision(self, reward_engine):
        """Test calibration bonus not awarded when precision <= 0.6."""
        # Ground truth: 1 hallucinated, 3 factual
        ground_truth = [
            Claim("H1", "hallucinated", "Correct 1"),
            Claim("F1", "factual", None),
            Claim("F2", "factual", None),
            Claim("F3", "factual", None)
        ]
        
        # Detection: 1 TP, 2 FP, 0 FN
        # Precision = 1/(1+2) = 0.33 < 0.6
        # Recall = 1/(1+0) = 1.0 > 0.6
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="Correct 1"),  # TP
            DetectedClaim(claim_text="F1", label="hallucinated", reason="Wrong", corrected_fact="Something"),  # FP
            DetectedClaim(claim_text="F2", label="hallucinated", reason="Wrong", corrected_fact="Something"),  # FP
            DetectedClaim(claim_text="F3", label="factual", reason="Correct", corrected_fact=None)   # TN
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should NOT get calibration bonus
        assert metrics["precision"] <= 0.6
        assert metrics["calibration_bonus"] == 0.0
    
    def test_calibration_bonus_not_awarded_low_recall(self, reward_engine):
        """Test calibration bonus not awarded when recall <= 0.6."""
        # Ground truth: 3 hallucinated, 1 factual
        ground_truth = [
            Claim("H1", "hallucinated", "Correct 1"),
            Claim("H2", "hallucinated", "Correct 2"),
            Claim("H3", "hallucinated", "Correct 3"),
            Claim("F1", "factual", None)
        ]
        
        # Detection: 1 TP, 2 FN, 0 FP
        # Precision = 1/(1+0) = 1.0 > 0.6
        # Recall = 1/(1+2) = 0.33 < 0.6
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="Correct 1"),  # TP
            DetectedClaim(claim_text="H2", label="factual", reason="Missed", corrected_fact=None),  # FN
            DetectedClaim(claim_text="H3", label="factual", reason="Missed", corrected_fact=None),  # FN
            DetectedClaim(claim_text="F1", label="factual", reason="Correct", corrected_fact=None)   # TN
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should NOT get calibration bonus
        assert metrics["recall"] <= 0.6
        assert metrics["calibration_bonus"] == 0.0
    
    def test_difficulty_multiplier_l1(self, reward_engine):
        """Test difficulty multiplier for L1 (1.0x)."""
        ground_truth = [Claim("Test", "factual", None)]
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="Test", label="factual", reason="Correct", corrected_fact=None)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        assert metrics["difficulty_multiplier"] == 1.0
        # Base reward is 0.5 (TN), multiplied by 1.0
        assert reward == 0.5
    
    def test_difficulty_multiplier_l2(self, reward_engine):
        """Test difficulty multiplier for L2 (1.5x)."""
        ground_truth = [Claim("Test", "factual", None)]
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="Test", label="factual", reason="Correct", corrected_fact=None)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L2")
        
        assert metrics["difficulty_multiplier"] == 1.5
        # Base reward is 0.5 (TN), multiplied by 1.5
        assert reward == 0.75
    
    def test_difficulty_multiplier_l3(self, reward_engine):
        """Test difficulty multiplier for L3 (2.0x)."""
        ground_truth = [Claim("Test", "factual", None)]
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="Test", label="factual", reason="Correct", corrected_fact=None)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L3")
        
        assert metrics["difficulty_multiplier"] == 2.0
        # Base reward is 0.5 (TN), multiplied by 2.0
        assert reward == 1.0
    
    def test_difficulty_multiplier_l4(self, reward_engine):
        """Test difficulty multiplier for L4 (2.5x)."""
        ground_truth = [Claim("Test", "factual", None)]
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="Test", label="factual", reason="Correct", corrected_fact=None)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L4")
        
        assert metrics["difficulty_multiplier"] == 2.5
        # Base reward is 0.5 (TN), multiplied by 2.5
        assert reward == 1.25
    
    def test_gaming_penalty_applied(self, reward_engine):
        """Test gaming penalty when >80% flagged."""
        # Ground truth: 1 hallucinated, 4 factual
        ground_truth = [
            Claim("H1", "hallucinated", "Correct"),
            Claim("F1", "factual", None),
            Claim("F2", "factual", None),
            Claim("F3", "factual", None),
            Claim("F4", "factual", None)
        ]
        
        # Detection: flag all 5 claims (100% > 80%)
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="Correct"),
            DetectedClaim(claim_text="F1", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F2", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F3", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F4", label="hallucinated", reason="Wrong", corrected_fact="Something")
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get gaming penalty
        assert metrics["gaming_penalty"] == -5.0
    
    def test_gaming_penalty_not_applied(self, reward_engine):
        """Test gaming penalty not applied when <=80% flagged."""
        # Ground truth: 2 hallucinated, 3 factual
        ground_truth = [
            Claim("H1", "hallucinated", "Correct 1"),
            Claim("H2", "hallucinated", "Correct 2"),
            Claim("F1", "factual", None),
            Claim("F2", "factual", None),
            Claim("F3", "factual", None)
        ]
        
        # Detection: flag 4 out of 5 (80% exactly, not > 80%)
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="Correct 1"),
            DetectedClaim(claim_text="H2", label="hallucinated", reason="Found", corrected_fact="Correct 2"),
            DetectedClaim(claim_text="F1", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F2", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F3", label="factual", reason="Correct", corrected_fact=None)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should NOT get gaming penalty (exactly 80%)
        assert metrics["gaming_penalty"] == 0.0
    
    def test_passivity_penalty_applied(self, reward_engine):
        """Test passivity penalty when <5% flagged with hallucinations."""
        # Ground truth: 1 hallucinated, 20 factual
        ground_truth = [Claim("H1", "hallucinated", "Correct")]
        ground_truth.extend([Claim(f"F{i}", "factual", None) for i in range(20)])
        
        # Detection: flag 0 out of 21 (0% < 5%)
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="factual", reason="Missed", corrected_fact=None)
        ] + [
            DetectedClaim(claim_text=f"F{i}", label="factual", reason="Correct", corrected_fact=None) for i in range(20)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should get passivity penalty
        assert metrics["passivity_penalty"] == -3.0
    
    def test_passivity_penalty_not_applied_no_hallucinations(self, reward_engine):
        """Test passivity penalty not applied when no hallucinations exist."""
        # Ground truth: all factual
        ground_truth = [Claim(f"F{i}", "factual", None) for i in range(10)]
        
        # Detection: flag 0 (0% < 5% but no hallucinations)
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text=f"F{i}", label="factual", reason="Correct", corrected_fact=None) for i in range(10)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should NOT get passivity penalty (no hallucinations)
        assert metrics["passivity_penalty"] == 0.0
    
    def test_passivity_penalty_not_applied_above_threshold(self, reward_engine):
        """Test passivity penalty not applied when >=5% flagged."""
        # Ground truth: 1 hallucinated, 19 factual
        ground_truth = [Claim("H1", "hallucinated", "Correct")]
        ground_truth.extend([Claim(f"F{i}", "factual", None) for i in range(19)])
        
        # Detection: flag 1 out of 20 (5% exactly, not < 5%)
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="Correct")
        ] + [
            DetectedClaim(claim_text=f"F{i}", label="factual", reason="Correct", corrected_fact=None) for i in range(19)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should NOT get passivity penalty (exactly 5%)
        assert metrics["passivity_penalty"] == 0.0
    
    def test_anti_gaming_flag_all_vs_flag_none(self, reward_engine):
        """Test that flagging all scores lower than flagging none."""
        # Ground truth: 1 hallucinated, 4 factual
        ground_truth = [
            Claim("H1", "hallucinated", "Correct"),
            Claim("F1", "factual", None),
            Claim("F2", "factual", None),
            Claim("F3", "factual", None),
            Claim("F4", "factual", None)
        ]
        
        # Strategy 1: Flag all
        flag_all = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="Correct"),
            DetectedClaim(claim_text="F1", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F2", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F3", label="hallucinated", reason="Wrong", corrected_fact="Something"),
            DetectedClaim(claim_text="F4", label="hallucinated", reason="Wrong", corrected_fact="Something")
        ])
        
        # Strategy 2: Flag none
        flag_none = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="factual", reason="Missed", corrected_fact=None),
            DetectedClaim(claim_text="F1", label="factual", reason="Correct", corrected_fact=None),
            DetectedClaim(claim_text="F2", label="factual", reason="Correct", corrected_fact=None),
            DetectedClaim(claim_text="F3", label="factual", reason="Correct", corrected_fact=None),
            DetectedClaim(claim_text="F4", label="factual", reason="Correct", corrected_fact=None)
        ])
        
        reward_all, _ = reward_engine.calculate_reward(flag_all, ground_truth, "L1")
        reward_none, _ = reward_engine.calculate_reward(flag_none, ground_truth, "L1")
        
        # Flag all should score lower than flag none
        assert reward_all < reward_none
    
    def test_fuzzy_matching_high_similarity(self, reward_engine):
        """Test fuzzy matching with high similarity (>70%)."""
        ground_truth = [
            Claim("The Eiffel Tower is in Paris", "factual", None)
        ]
        
        # Detection with slightly different wording
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="The Eiffel Tower is located in Paris",
                label="factual",
                reason="Correct",
                corrected_fact=None
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should match and count as TN
        assert metrics["true_negatives"] == 1
    
    def test_fuzzy_matching_low_similarity(self, reward_engine):
        """Test fuzzy matching with low similarity (<70%)."""
        ground_truth = [
            Claim("The Eiffel Tower is in Paris", "factual", None)
        ]
        
        # Detection with completely different claim
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(
                claim_text="The Statue of Liberty is in New York",
                label="factual",
                reason="Different",
                corrected_fact=None
            )
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should not match, ground truth becomes FN (if hallucinated) or ignored
        # Since ground truth is factual and unmatched, it doesn't affect confusion matrix
        assert metrics["true_negatives"] == 0
    
    def test_unverifiable_claims_treated_as_factual(self, reward_engine):
        """Test that unverifiable claims are treated as factual for rewards."""
        # Ground truth: unverifiable claim
        ground_truth = [
            Claim("This might be true", "unverifiable", None)
        ]
        
        # Detection: correctly identified as unverifiable/factual
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="This might be true", label="unverifiable", reason="Cannot verify", corrected_fact=None)
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Should count as TN (correctly identified as not hallucinated)
        assert metrics["true_negatives"] == 1
        assert metrics["base_reward"] == 0.5
    
    def test_empty_detection_output(self, reward_engine):
        """Test handling of empty detection output."""
        ground_truth = [
            Claim("H1", "hallucinated", "Correct"),
            Claim("F1", "factual", None)
        ]
        
        # Empty detection
        detection = DetectionOutput(detected_claims=[])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # All ground truth hallucinations become FN
        assert metrics["false_negatives"] == 1
        # No penalties should be applied for empty output
        assert metrics["gaming_penalty"] == 0.0
        assert metrics["passivity_penalty"] == 0.0
    
    def test_empty_ground_truth(self, reward_engine):
        """Test handling of empty ground truth."""
        ground_truth = []
        
        # Detection with claims
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="Test", label="hallucinated", reason="Wrong", corrected_fact="Something")
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Unmatched detected hallucinations become FP
        assert metrics["false_positives"] == 1
    
    def test_precision_recall_calculation(self, reward_engine):
        """Test precision and recall calculation."""
        # Ground truth: 3 hallucinated, 2 factual
        ground_truth = [
            Claim("H1", "hallucinated", "C1"),
            Claim("H2", "hallucinated", "C2"),
            Claim("H3", "hallucinated", "C3"),
            Claim("F1", "factual", None),
            Claim("F2", "factual", None)
        ]
        
        # Detection: 2 TP, 1 FP, 1 FN, 1 TN
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="C1"),  # TP
            DetectedClaim(claim_text="H2", label="hallucinated", reason="Found", corrected_fact="C2"),  # TP
            DetectedClaim(claim_text="H3", label="factual", reason="Missed", corrected_fact=None),  # FN
            DetectedClaim(claim_text="F1", label="hallucinated", reason="Wrong", corrected_fact="X"),  # FP
            DetectedClaim(claim_text="F2", label="factual", reason="Correct", corrected_fact=None)  # TN
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L1")
        
        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 0.667
        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 0.667
        assert abs(metrics["precision"] - 0.667) < 0.01
        assert abs(metrics["recall"] - 0.667) < 0.01
        assert abs(metrics["f1"] - 0.667) < 0.01
    
    def test_complete_reward_formula(self, reward_engine):
        """Test complete reward formula with all components."""
        # Ground truth: 2 hallucinated, 2 factual
        ground_truth = [
            Claim("H1", "hallucinated", "The tower is 330 meters"),
            Claim("H2", "hallucinated", "Built in 1889"),
            Claim("F1", "factual", None),
            Claim("F2", "factual", None)
        ]
        
        # Detection: 2 TP with good corrections, 0 FP, 0 FN, 2 TN
        # Precision = 2/2 = 1.0 > 0.6
        # Recall = 2/2 = 1.0 > 0.6
        detection = DetectionOutput(detected_claims=[
            DetectedClaim(claim_text="H1", label="hallucinated", reason="Found", corrected_fact="The tower is 330 meters tall"),  # TP
            DetectedClaim(claim_text="H2", label="hallucinated", reason="Found", corrected_fact="Built in year 1889"),  # TP
            DetectedClaim(claim_text="F1", label="factual", reason="Correct", corrected_fact=None),  # TN
            DetectedClaim(claim_text="F2", label="factual", reason="Correct", corrected_fact=None)   # TN
        ])
        
        reward, metrics = reward_engine.calculate_reward(detection, ground_truth, "L2")
        
        # Base reward: 2*3.0 + 0*(-2.0) + 0*(-1.5) + 2*0.5 = 6.0 + 1.0 = 7.0
        # Correction bonus: ~0.5-1.0 per correction = ~1.0-2.0 total
        # Calibration bonus: 1.0 (precision and recall > 0.6)
        # No penalties
        # Difficulty multiplier: 1.5 (L2)
        # Total: (7.0 + correction_bonus + 1.0) * 1.5
        
        assert metrics["base_reward"] == 7.0
        assert metrics["correction_bonus"] > 0.0
        assert metrics["calibration_bonus"] == 1.0
        assert metrics["gaming_penalty"] == 0.0
        assert metrics["passivity_penalty"] == 0.0
        assert metrics["difficulty_multiplier"] == 1.5
        assert reward > 10.0  # Should be substantial positive reward
