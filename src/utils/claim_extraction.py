"""
Claim extraction utilities for decomposing LLM-generated text into individual claims.

This module provides functions to extract factual claims from text using spaCy
for sentence segmentation and dependency parsing.
"""

import spacy
from typing import List

# Global spaCy model instance (loaded lazily)
_nlp = None


def _get_nlp():
    """Get or load the spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not found, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def is_declarative(sent) -> bool:
    """
    Check if a sentence is declarative (not a question or imperative).
    
    Args:
        sent: A spaCy Span object representing a sentence
        
    Returns:
        True if the sentence is declarative, False otherwise
    """
    text = sent.text.strip()
    
    # Filter out questions (end with ?)
    if text.endswith('?'):
        return False
    
    # Filter out imperatives (start with verb in base form)
    # Check if the first token is a verb in imperative mood
    if len(sent) > 0:
        first_token = sent[0]
        # Imperatives typically start with a base verb (VB) or are commands
        if first_token.pos_ == "VERB" and first_token.tag_ == "VB":
            # Check if it's not part of a declarative sentence with auxiliary
            # (e.g., "You should go" vs "Go now")
            if not any(token.dep_ == "nsubj" for token in sent):
                return False
    
    # Filter out very short sentences (likely fragments)
    if len(text.split()) < 3:
        return False
    
    return True


def split_on_conjunctions(sent) -> List[str]:
    """
    Split compound sentences on coordinating conjunctions.
    
    Args:
        sent: A spaCy Span object representing a sentence
        
    Returns:
        List of clause strings
    """
    clauses = []
    current_clause_tokens = []
    
    # Coordinating conjunctions that typically separate independent clauses
    coord_conjunctions = {"and", "but", "or", "yet", "so", "nor"}
    
    for token in sent:
        # Check if this is a coordinating conjunction that separates clauses
        if token.text.lower() in coord_conjunctions and token.dep_ == "cc":
            # Save the current clause if it's not empty
            if current_clause_tokens:
                clause_text = " ".join(t.text for t in current_clause_tokens).strip()
                if clause_text:
                    clauses.append(clause_text)
                current_clause_tokens = []
        else:
            current_clause_tokens.append(token)
    
    # Add the final clause
    if current_clause_tokens:
        clause_text = " ".join(t.text for t in current_clause_tokens).strip()
        if clause_text:
            clauses.append(clause_text)
    
    # If no conjunctions were found, return the original sentence
    if not clauses:
        return [sent.text.strip()]
    
    return clauses


def extract_claims(text: str) -> List[str]:
    """
    Extract individual factual claims from text.
    
    This function uses spaCy for sentence segmentation and dependency parsing
    to identify independent claims. It splits compound sentences at coordinating
    conjunctions and filters out questions and imperatives.
    
    Args:
        text: The text to extract claims from
        
    Returns:
        List of claim strings
        
    Example:
        >>> extract_claims("The Eiffel Tower was built in 1889 and it stands 330 meters tall.")
        ['The Eiffel Tower was built in 1889', 'it stands 330 meters tall']
    """
    if not text or not text.strip():
        return []
    
    nlp = _get_nlp()
    doc = nlp(text)
    
    claims = []
    
    for sent in doc.sents:
        # Check if the sentence is declarative
        if not is_declarative(sent):
            continue
        
        # Split compound sentences on conjunctions
        clauses = split_on_conjunctions(sent)
        
        for clause in clauses:
            # Clean up the clause
            clause = clause.strip()
            
            # Filter out very short clauses (likely fragments)
            if len(clause.split()) >= 3:
                claims.append(clause)
    
    return claims
