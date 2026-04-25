"""
HaluEval dataset parser.

This module parses the HaluEval dataset format and converts it to Episode objects.
HaluEval contains QA, summarization, and dialog samples with hallucinated answers.
"""

import json
from typing import Dict, Any, List
from dataclasses import dataclass
from src.environment.core import Episode, Claim
from src.utils.claim_extraction import extract_claims


def parse_halueval_entry(entry: Dict[str, Any], episode_id: str) -> Episode:
    """
    Parse a single HaluEval dataset entry into an Episode.
    
    Expected format:
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "hallucinated_answer": "The capital of France is London.",
        "label": "hallucinated"
    }
    
    Args:
        entry: Dictionary containing HaluEval fields
        episode_id: Unique identifier for the episode
        
    Returns:
        Episode object with extracted claims and labels
    """
    # Extract fields
    question = entry.get("question", "")
    correct_answer = entry.get("answer", "")
    hallucinated_answer = entry.get("hallucinated_answer", "")
    label = entry.get("label", "hallucinated")
    
    # Use the hallucinated answer as the generated response
    generated_response = hallucinated_answer
    
    # Extract claims from the generated response
    claim_texts = extract_claims(generated_response)
    
    # Create claims with labels
    claims = []
    for claim_text in claim_texts:
        # Determine if this claim is hallucinated or factual
        # For HaluEval, if the overall label is "hallucinated", we mark claims as hallucinated
        # In a more sophisticated version, we would do claim-level matching
        if label == "hallucinated":
            claim_label = "hallucinated"
            ground_truth_fact = correct_answer if correct_answer else None
        else:
            claim_label = "factual"
            ground_truth_fact = None
        
        claims.append(Claim(
            claim_text=claim_text,
            label=claim_label,
            ground_truth_fact=ground_truth_fact
        ))
    
    # If no claims were extracted, create at least one claim from the response
    if not claims:
        claims.append(Claim(
            claim_text=generated_response.strip(),
            label="hallucinated" if label == "hallucinated" else "factual",
            ground_truth_fact=correct_answer if label == "hallucinated" and correct_answer else None
        ))
    
    # Create episode
    episode = Episode(
        episode_id=episode_id,
        source_dataset="halueval_qa",
        difficulty_level="L1",  # Will be assigned by EpisodeBank
        source_text=question,
        generated_response=generated_response,
        claims=claims,
        metadata={
            "correct_answer": correct_answer,
            "overall_label": label,
            "claim_count": len(claims)
        }
    )
    
    return episode


def parse_halueval_dataset(file_path: str) -> List[Episode]:
    """
    Parse a HaluEval dataset file and return a list of Episodes.
    
    Args:
        file_path: Path to the HaluEval JSON file
        
    Returns:
        List of Episode objects
    """
    episodes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "data" in data:
        entries = data["data"]
    else:
        entries = [data]
    
    for idx, entry in enumerate(entries):
        episode_id = f"halueval_{idx:05d}"
        try:
            episode = parse_halueval_entry(entry, episode_id)
            episodes.append(episode)
        except Exception as e:
            print(f"Warning: Failed to parse HaluEval entry {episode_id}: {e}")
            continue
    
    return episodes
