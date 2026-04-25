"""Main application entry point for Hallucination Hunter RL Environment."""

import uvicorn
from pathlib import Path

from src.environment.episode_bank import EpisodeBank
from src.environment.curriculum import CurriculumManager
from src.environment.reward import RewardEngine
from src.api.server import create_app


def main():
    """Initialize and run the FastAPI server."""
    print("=" * 60)
    print("Hallucination Hunter RL Environment")
    print("=" * 60)
    
    # Paths
    episodes_dir = Path(__file__).parent / "data" / "episodes"
    
    # Initialize components
    print("\nInitializing components...")
    
    # Episode Bank
    print("  Loading episode bank...")
    episode_bank = EpisodeBank()
    episode_bank.load_episodes(str(episodes_dir))
    stats = episode_bank.get_statistics()
    print(f"    Loaded {stats['episode_count']} episodes")
    print(f"    Distribution: {stats['difficulty_distribution']}")
    
    # Curriculum Manager
    print("  Initializing curriculum manager...")
    promotion_thresholds = {
        "L1": 3.5,  # Promote to L2 when L1 avg reward > 3.5
        "L2": 4.0,  # Promote to L3 when L2 avg reward > 4.0
        "L3": 5.0   # Promote to L4 when L3 avg reward > 5.0
    }
    curriculum_manager = CurriculumManager(
        promotion_thresholds=promotion_thresholds,
        window_size=50
    )
    print(f"    Enabled levels: {curriculum_manager.get_enabled_levels()}")
    
    # Reward Engine
    print("  Initializing reward engine...")
    reward_engine = RewardEngine()
    print("    Reward engine ready")
    
    # Create FastAPI app
    print("\nCreating FastAPI application...")
    app = create_app(
        episode_bank=episode_bank,
        curriculum_manager=curriculum_manager,
        reward_engine=reward_engine
    )
    
    # Server configuration
    host = "0.0.0.0"
    port = 7860
    
    print("\n" + "=" * 60)
    print(f"Server starting on http://{host}:{port}")
    print("=" * 60)
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print("=" * 60)
    
    # Run server
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
