"""FastAPI server for Hallucination Hunter RL environment."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import threading
from typing import Dict, Any
import time
from pathlib import Path

from src.environment.core import HallucinationEnvironment
from src.environment.episode_bank import EpisodeBank
from src.environment.curriculum import CurriculumManager
from src.environment.reward import RewardEngine
from src.api.models import Action, Observation, StepResult


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global environment instance (thread-safe)
_env_lock = threading.Lock()
_environment: HallucinationEnvironment = None


def create_app(
    episode_bank: EpisodeBank,
    curriculum_manager: CurriculumManager,
    reward_engine: RewardEngine
) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        episode_bank: Episode storage and sampling system
        curriculum_manager: Difficulty progression manager
        reward_engine: Reward calculation engine
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Hallucination Hunter RL Environment",
        description="OpenEnv-compatible API for training language models to detect hallucinations",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Mount static files for playground
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Initialize global environment
    global _environment
    with _env_lock:
        _environment = HallucinationEnvironment(
            episode_bank=episode_bank,
            curriculum_manager=curriculum_manager,
            reward_engine=reward_engine
        )
    
    @app.get("/")
    async def serve_playground():
        """Serve the playground HTML interface."""
        playground_path = Path(__file__).parent.parent.parent / "static" / "playground.html"
        if playground_path.exists():
            return FileResponse(str(playground_path))
        return {"message": "Hallucination Hunter API", "docs": "/docs"}
    
    @app.post("/reset")
    @limiter.limit("60/minute")
    async def reset(request: Request) -> Dict[str, Any]:
        """Initialize a new episode and return the initial observation.
        
        Returns:
            Dictionary containing:
            - observation: Dict with generated_text and task_instruction
            - info: Dict with episode_id, difficulty_level, source_dataset
        """
        with _env_lock:
            try:
                observation, info = _environment.reset()
                # Convert observation dict to Observation model for consistency
                obs_model = Observation(**observation)
                return {
                    "observation": obs_model.model_dump(),
                    "info": info
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/step")
    @limiter.limit("60/minute")
    async def step(request: Request, action: Action) -> StepResult:
        """Process agent's detection output and return reward and next state.
        
        Args:
            action: Agent's action containing detection output
            
        Returns:
            StepResult with observation, reward, done, and info
        """
        with _env_lock:
            try:
                # step() returns tuple: (observation, reward, done, info)
                observation, reward, done, info = _environment.step(action.detection_output)
                
                # Convert observation dict to Observation model
                obs_model = Observation(**observation)
                
                # Construct StepResult
                return StepResult(
                    observation=obs_model,
                    reward=reward,
                    done=done,
                    info=info
                )
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Get server health status and episode bank statistics.
        
        Returns:
            Dictionary containing:
            - status: Server status ("healthy")
            - episode_count: Total number of episodes
            - difficulty_distribution: Count of episodes per difficulty level
            - curriculum_state: Current curriculum state with enabled levels and rolling averages
        """
        try:
            stats = episode_bank.get_statistics()
            curriculum_state = _environment.get_curriculum_state()
            
            return {
                "status": "healthy",
                "episode_count": stats["episode_count"],
                "difficulty_distribution": stats["difficulty_distribution"],
                "curriculum_state": curriculum_state
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/state")
    async def get_state() -> Dict[str, Any]:
        """Get current environment state (OpenEnv standard endpoint).
        
        Returns:
            Dictionary containing:
            - current_level: Current curriculum difficulty level
            - rolling_reward: Rolling average reward
            - episodes_completed: Total episodes completed
            - level_rewards: Per-level reward breakdown
            - enabled_levels: List of currently enabled difficulty levels
        """
        try:
            curriculum_state = _environment.get_curriculum_state()
            
            return {
                "current_level": curriculum_state.get("current_level", "L1"),
                "rolling_reward": curriculum_state.get("rolling_avg_rewards", {}).get("overall", 0.0),
                "episodes_completed": len(_environment.curriculum_manager.episode_history),
                "level_rewards": curriculum_state.get("rolling_avg_rewards", {}),
                "enabled_levels": curriculum_state.get("enabled_levels", ["L1"])
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def get_environment() -> HallucinationEnvironment:
    """Get the global environment instance (thread-safe).
    
    Returns:
        HallucinationEnvironment instance
    """
    with _env_lock:
        return _environment
