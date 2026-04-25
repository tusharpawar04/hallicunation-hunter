"""
Wikipedia synthetic dataset parser.

This module parses Wikipedia-based synthetic episodes and converts them to Episode objects.
These episodes contain Wikipedia paragraphs with LLM-generated summaries and fact labels.
"""

import json
from typing import Dict, Any, List
from dataclasses import dataclass
from src.environment.core import Episode, Claim
from src.utils.claim_extraction import extract_claims


def parse_wikipedia_entry(entry: Dict[str, Any], episode_id: str) -> Episode:
    """
    Parse a single Wikipedia synthetic entry into an Episode.
    
    Expected format:
    {
        "paragraph": "The Eiffel Tower is a wrought-iron lattice tower...",
        "summary": "The Eiffel Tower was built in 1889 and stands 400 meters tall.",
        "fact_labels": [
            {"claim": "The Eiffel Tower was built in 1889", "label": "factual"},
            {"claim": "It stands 400 meters tall", "label": "hallucinated", "ground_truth": "It stands 330 meters tall"}
        ]
    }
    
    Args:
        entry: Dictionary containing Wikipedia synthetic fields
        episode_id: Unique identifier for the episode
        
    Returns:
        Episode object with extracted claims and labels
    """
    # Extract fields
    paragraph = entry.get("paragraph", "")
    summary = entry.get("summary", "")
    fact_labels = entry.get("fact_labels", [])
    
    # Use summary as the generated response
    generated_response = summary
    
    # If fact_labels are provided, use them directly
    if fact_labels:
        claims = []
        for fact_label in fact_labels:
            claim_text = fact_label.get("claim", "")
            label = fact_label.get("label", "factual")
            ground_truth = fact_label.get("ground_truth", None)
            
            claims.append(Claim(
                claim_text=claim_text,
                label=label,
                ground_truth_fact=ground_truth
            ))
    else:
        # Extract claims from the summary
        claim_texts = extract_claims(generated_response)
        
        # Create claims (default to factual since we don't have labels)
        claims = []
        for claim_text in claim_texts:
            claims.append(Claim(
                claim_text=claim_text,
                label="factual",
                ground_truth_fact=None
            ))
    
    # If no claims were extracted, create at least one claim from the response
    if not claims:
        claims.append(Claim(
            claim_text=generated_response.strip(),
            label="factual",
            ground_truth_fact=None
        ))
    
    # Create episode
    episode = Episode(
        episode_id=episode_id,
        source_dataset="wikipedia_synthetic",
        difficulty_level="L1",  # Will be assigned by EpisodeBank
        source_text=paragraph,
        generated_response=generated_response,
        claims=claims,
        metadata={
            "topic": entry.get("topic", "unknown"),
            "claim_count": len(claims),
            "has_fact_labels": bool(fact_labels)
        }
    )
    
    return episode


def parse_wikipedia_dataset(file_path: str) -> List[Episode]:
    """
    Parse a Wikipedia synthetic dataset file and return a list of Episodes.
    
    Args:
        file_path: Path to the Wikipedia synthetic JSON file
        
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
        episode_id = f"wikipedia_{idx:05d}"
        try:
            episode = parse_wikipedia_entry(entry, episode_id)
            episodes.append(episode)
        except Exception as e:
            print(f"Warning: Failed to parse Wikipedia entry {episode_id}: {e}")
            continue
    
    return episodes
