"""Reward calculation engine for Hallucination Hunter environment."""

from typing import Any, Dict, List, Tuple
import numpy as np
from fuzzywuzzy import fuzz
from scipy.optimize import linear_sum_assignment

from src.environment.core import Claim
from src.api.models import DetectedClaim, DetectionOutput


class RewardEngine:
    """Calculates deterministic rewards for hallucination detection tasks.
    
    The reward formula combines:
    - Base rewards for confusion matrix components (TP, FP, FN, TN)
    - Correction bonus based on keyword overlap with ground truth
    - Calibration bonus for balanced precision and recall
    - Anti-gaming penalties for extreme flagging rates
    - Difficulty multipliers to incentivize progression
    """
    
    # Base reward values
    TRUE_POSITIVE_REWARD = 3.0
    FALSE_POSITIVE_PENALTY = -2.0
    FALSE_NEGATIVE_PENALTY = -1.5
    TRUE_NEGATIVE_REWARD = 0.5
    
    # Bonus and penalty values
    CALIBRATION_BONUS = 1.0
    GAMING_PENALTY = -5.0
    PASSIVITY_PENALTY = -3.0
    
    # Thresholds
    CALIBRATION_THRESHOLD = 0.6
    GAMING_THRESHOLD = 0.8
    PASSIVITY_THRESHOLD = 0.05
    FUZZY_MATCH_THRESHOLD = 70
    
    # Difficulty multipliers
    DIFFICULTY_MULTIPLIERS = {
        "L1": 1.0,
        "L2": 1.5,
        "L3": 2.0,
        "L4": 2.5
    }
    
    def calculate_reward(
        self,
        detection_output: DetectionOutput,
        ground_truth_claims: List[Claim],
        difficulty_level: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate total reward for a detection output.
        
        Args:
            detection_output: Agent's detection results
            ground_truth_claims: Ground truth claims from the episode
            difficulty_level: Episode difficulty level (L1-L4)
            
        Returns:
            Tuple of (total_reward, metrics_dict) where metrics_dict contains:
            - precision, recall, f1
            - true_positives, false_positives, false_negatives, true_negatives
            - correction_bonus, calibration_bonus
            - difficulty_multiplier
            - penalties (gaming_penalty, passivity_penalty)
        """
        # Match detected claims to ground truth
        matches = self._match_claims(
            detection_output.detected_claims,
            ground_truth_claims
        )
        
        # Calculate confusion matrix
        tp, fp, fn, tn = self._calculate_confusion_matrix(
            matches,
            detection_output.detected_claims,
            ground_truth_claims
        )
        
        # Base reward
        base_reward = (
            tp * self.TRUE_POSITIVE_REWARD +
            fp * self.FALSE_POSITIVE_PENALTY +
            fn * self.FALSE_NEGATIVE_PENALTY +
            tn * self.TRUE_NEGATIVE_REWARD
        )
        
        # Correction bonus
        correction_bonus = self._calculate_total_correction_bonus(
            matches,
            detection_output.detected_claims,
            ground_truth_claims
        )
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # Calibration bonus
        calibration_bonus = (
            self.CALIBRATION_BONUS
            if (precision > self.CALIBRATION_THRESHOLD and recall > self.CALIBRATION_THRESHOLD)
            else 0.0
        )
        
        # Anti-gaming penalties
        gaming_penalty = self._check_gaming_penalty(
            detection_output,
            ground_truth_claims
        )
        
        passivity_penalty = self._check_passivity_penalty(
            detection_output,
            ground_truth_claims
        )
        
        # Difficulty multiplier
        difficulty_multiplier = self.DIFFICULTY_MULTIPLIERS.get(difficulty_level, 1.0)
        
        # Total reward
        total_reward = (
            base_reward +
            correction_bonus +
            calibration_bonus +
            gaming_penalty +
            passivity_penalty
        ) * difficulty_multiplier
        
        # Build metrics dictionary
        metrics = {
            "base_reward": base_reward,
            "correction_bonus": correction_bonus,
            "calibration_bonus": calibration_bonus,
            "gaming_penalty": gaming_penalty,
            "passivity_penalty": passivity_penalty,
            "difficulty_multiplier": difficulty_multiplier,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }
        
        return total_reward, metrics
    
    def _match_claims(
        self,
        detected: List[DetectedClaim],
        ground_truth: List[Claim]
    ) -> List[Tuple[DetectedClaim, Claim]]:
        """Match detected claims to ground truth using fuzzy string matching.
        
        Uses FuzzyWuzzy ratio > 70% and Hungarian algorithm for optimal matching.
        
        Args:
            detected: List of detected claims from agent
            ground_truth: List of ground truth claims
            
        Returns:
            List of matched (detected, ground_truth) pairs
        """
        if not detected or not ground_truth:
            return []
        
        # Build similarity matrix
        similarity_matrix = np.zeros((len(detected), len(ground_truth)))
        for i, det in enumerate(detected):
            for j, gt in enumerate(ground_truth):
                similarity_matrix[i, j] = fuzz.ratio(det.claim_text, gt.claim_text)
        
        # Find optimal matching using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Filter by threshold
        matches = []
        for i, j in zip(row_ind, col_ind):
            if similarity_matrix[i, j] > self.FUZZY_MATCH_THRESHOLD:
                matches.append((detected[i], ground_truth[j]))
        
        return matches
    
    def _calculate_confusion_matrix(
        self,
        matches: List[Tuple[DetectedClaim, Claim]],
        detected: List[DetectedClaim],
        ground_truth: List[Claim]
    ) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix components.
        
        Args:
            matches: List of matched (detected, ground_truth) pairs
            detected: All detected claims
            ground_truth: All ground truth claims
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, true_negatives)
        """
        # Track which claims have been matched using indices
        matched_detected_indices = {id(det) for det, _ in matches}
        matched_ground_truth_indices = {id(gt) for _, gt in matches}
        
        # Count confusion matrix components
        tp = 0  # Correctly identified hallucinations
        tn = 0  # Correctly identified factual claims
        
        for det, gt in matches:
            if det.label == "hallucinated" and gt.label == "hallucinated":
                tp += 1
            elif det.label == "factual" and gt.label == "factual":
                tn += 1
            elif det.label == "hallucinated" and gt.label == "factual":
                # This is a false positive (incorrectly flagged factual as hallucinated)
                pass
            elif det.label == "factual" and gt.label == "hallucinated":
                # This is a false negative (missed hallucination)
                pass
            # Note: "unverifiable" labels are treated as factual for reward purposes
            elif det.label == "hallucinated" and gt.label == "unverifiable":
                pass  # Counted as FP below
            elif det.label in ["factual", "unverifiable"] and gt.label == "unverifiable":
                tn += 1  # Correctly identified as not hallucinated
        
        # False positives: detected as hallucinated but actually factual/unverifiable
        fp = sum(
            1 for det, gt in matches
            if det.label == "hallucinated" and gt.label in ["factual", "unverifiable"]
        )
        
        # False negatives: missed hallucinations (ground truth hallucinated but detected as factual/unverifiable)
        fn = sum(
            1 for det, gt in matches
            if det.label in ["factual", "unverifiable"] and gt.label == "hallucinated"
        )
        
        # Add unmatched detected claims as false positives if flagged as hallucinated
        fp += sum(
            1 for det in detected
            if id(det) not in matched_detected_indices and det.label == "hallucinated"
        )
        
        # Add unmatched ground truth hallucinations as false negatives
        fn += sum(
            1 for gt in ground_truth
            if id(gt) not in matched_ground_truth_indices and gt.label == "hallucinated"
        )
        
        # Add unmatched factual/unverifiable ground truth as false negatives if detected as hallucinated
        # (already counted in unmatched detected claims)
        
        return tp, fp, fn, tn
    
    def _calculate_correction_bonus(
        self,
        corrected_fact: str,
        ground_truth_fact: str
    ) -> float:
        """Calculate bonus based on keyword overlap (Jaccard similarity).
        
        Args:
            corrected_fact: Agent's correction for a hallucinated claim
            ground_truth_fact: Ground truth correction
            
        Returns:
            Bonus score from 0.0 to 1.0
        """
        if not corrected_fact or not ground_truth_fact:
            return 0.0
        
        # Tokenize and convert to lowercase
        corrected_tokens = set(corrected_fact.lower().split())
        ground_truth_tokens = set(ground_truth_fact.lower().split())
        
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'it', 'its'
        }
        corrected_tokens -= stopwords
        ground_truth_tokens -= stopwords
        
        if not ground_truth_tokens:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = corrected_tokens & ground_truth_tokens
        union = corrected_tokens | ground_truth_tokens
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_total_correction_bonus(
        self,
        matches: List[Tuple[DetectedClaim, Claim]],
        detected: List[DetectedClaim],
        ground_truth: List[Claim]
    ) -> float:
        """Calculate total correction bonus across all matched hallucinations.
        
        Args:
            matches: List of matched (detected, ground_truth) pairs
            detected: All detected claims
            ground_truth: All ground truth claims
            
        Returns:
            Total correction bonus
        """
        total_bonus = 0.0
        
        for det, gt in matches:
            # Only award correction bonus for correctly identified hallucinations
            if (det.label == "hallucinated" and 
                gt.label == "hallucinated" and 
                det.corrected_fact and 
                gt.ground_truth_fact):
                bonus = self._calculate_correction_bonus(
                    det.corrected_fact,
                    gt.ground_truth_fact
                )
                total_bonus += bonus
        
        return total_bonus
    
    def _check_gaming_penalty(
        self,
        detection_output: DetectionOutput,
        ground_truth_claims: List[Claim]
    ) -> float:
        """Check if gaming penalty should be applied.
        
        Gaming penalty is proportional to false positive rate to prevent gaming.
        A high false positive rate indicates the agent is flagging too many factual claims.
        
        Args:
            detection_output: Agent's detection results
            ground_truth_claims: Ground truth claims
            
        Returns:
            Gaming penalty (0.0 or negative value proportional to FP rate)
        """
        if not detection_output.detected_claims or not ground_truth_claims:
            return 0.0
        
        # Match claims to calculate false positives
        matches = self._match_claims(detection_output.detected_claims, ground_truth_claims)
        
        # Count false positives: detected as hallucinated but actually factual
        fp = sum(
            1 for det, gt in matches
            if det.label == "hallucinated" and gt.label in ["factual", "unverifiable"]
        )
        
        # Count total factual claims in ground truth
        total_factual = sum(
            1 for gt in ground_truth_claims
            if gt.label in ["factual", "unverifiable"]
        )
        
        if total_factual == 0:
            return 0.0
        
        # Calculate false positive rate
        fp_rate = fp / max(total_factual, 1)
        
        # Apply proportional penalty if FP rate > 50%
        if fp_rate > 0.5:
            gaming_penalty = -fp_rate * 4.0  # Scales from -2.0 to -4.0
            return gaming_penalty
        
        return 0.0
    
    def _check_passivity_penalty(
        self,
        detection_output: DetectionOutput,
        ground_truth_claims: List[Claim]
    ) -> float:
        """Check if passivity penalty should be applied.
        
        Passivity penalty is applied when <5% of claims are flagged as hallucinated
        in episodes with known hallucinations.
        
        Args:
            detection_output: Agent's detection results
            ground_truth_claims: Ground truth claims
            
        Returns:
            Passivity penalty (0.0 or negative value)
        """
        if not detection_output.detected_claims:
            return 0.0
        
        # Check if there are any hallucinations in ground truth
        has_hallucinations = any(
            claim.label == "hallucinated"
            for claim in ground_truth_claims
        )
        
        if not has_hallucinations:
            return 0.0
        
        # Calculate flagged rate
        flagged_count = sum(
            1 for det in detection_output.detected_claims
            if det.label == "hallucinated"
        )
        
        flagged_rate = flagged_count / len(detection_output.detected_claims)
        
        if flagged_rate < self.PASSIVITY_THRESHOLD:
            return self.PASSIVITY_PENALTY
        
        return 0.0
