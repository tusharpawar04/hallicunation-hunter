"""Client wrapper for Hallucination Hunter RL Environment."""

import requests
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp

from src.api.models import Action, DetectionOutput, DetectedClaim


class HallucinationHunterEnv:
    """Client wrapper for interacting with the Hallucination Hunter API.
    
    This class provides a simple interface for training frameworks to interact
    with the environment through HTTP requests.
    
    Attributes:
        base_url: Base URL of the API server
        session: Requests session for connection pooling
    """
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        """Initialize the environment client.
        
        Args:
            base_url: Base URL of the API server (default: http://localhost:7860)
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def reset(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Initialize a new episode.
        
        Returns:
            Tuple of (observation, info) where:
            - observation: Dict with generated_text and task_instruction
            - info: Dict with episode_id, difficulty_level, source_dataset
            
        Raises:
            requests.RequestException: If the API request fails
        """
        response = self.session.post(f"{self.base_url}/reset")
        response.raise_for_status()
        
        data = response.json()
        return data["observation"], data["info"]
    
    def step(self, detection_output: DetectionOutput) -> Dict[str, Any]:
        """Submit detection output and get reward.
        
        Args:
            detection_output: Agent's detection results
            
        Returns:
            Dictionary containing observation, reward, done, and info
            
        Raises:
            requests.RequestException: If the API request fails
        """
        action = Action(detection_output=detection_output)
        
        response = self.session.post(
            f"{self.base_url}/step",
            json=action.model_dump()
        )
        response.raise_for_status()
        
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Get server health status.
        
        Returns:
            Dictionary with status, episode_count, difficulty_distribution, curriculum_state
            
        Raises:
            requests.RequestException: If the API request fails
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        
        return response.json()
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class HallucinationHunterEnvTRL:
    """TRL-compatible wrapper with support for parallel generations.
    
    This class extends the basic client to support GRPO training with
    8 parallel generations per prompt.
    """
    
    def __init__(self, base_url: str = "http://localhost:7860", num_generations: int = 8):
        """Initialize the TRL-compatible environment client.
        
        Args:
            base_url: Base URL of the API server
            num_generations: Number of parallel generations per prompt (default: 8)
        """
        self.base_url = base_url.rstrip("/")
        self.num_generations = num_generations
        self.session = requests.Session()
    
    def reset_batch(self, batch_size: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Initialize multiple episodes in parallel.
        
        Args:
            batch_size: Number of episodes to initialize
            
        Returns:
            Tuple of (observations, infos) where each is a list of dicts
        """
        observations = []
        infos = []
        
        for _ in range(batch_size):
            response = self.session.post(f"{self.base_url}/reset")
            response.raise_for_status()
            data = response.json()
            observations.append(data["observation"])
            infos.append(data["info"])
        
        return observations, infos
    
    def step_batch(
        self,
        detection_outputs: List[DetectionOutput]
    ) -> List[Dict[str, Any]]:
        """Submit multiple detection outputs and get rewards.
        
        Args:
            detection_outputs: List of detection results from parallel generations
            
        Returns:
            List of step results (observation, reward, done, info)
        """
        results = []
        
        for detection_output in detection_outputs:
            action = Action(detection_output=detection_output)
            response = self.session.post(
                f"{self.base_url}/step",
                json=action.model_dump()
            )
            response.raise_for_status()
            results.append(response.json())
        
        return results
    
    async def reset_batch_async(self, batch_size: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Initialize multiple episodes asynchronously.
        
        Args:
            batch_size: Number of episodes to initialize
            
        Returns:
            Tuple of (observations, infos)
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._reset_async(session)
                for _ in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)
        
        observations = [r["observation"] for r in results]
        infos = [r["info"] for r in results]
        
        return observations, infos
    
    async def _reset_async(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Async helper for reset."""
        async with session.post(f"{self.base_url}/reset") as response:
            response.raise_for_status()
            return await response.json()
    
    def close(self):
        """Close the session."""
        self.session.close()
