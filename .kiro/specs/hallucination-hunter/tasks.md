# Implementation Plan: Hallucination Hunter RL Environment

## Overview

This implementation plan breaks down the Hallucination Hunter RL environment into 9 phases following the design document's implementation approach. The system is an OpenEnv-compatible reinforcement learning environment that trains language models to detect hallucinations at the claim level using deterministic rewards, curriculum learning, and anti-gaming penalties.

## Tasks

### Phase 1: Core Environment

- [x] 1. Set up project structure and dependencies
  - Create directory structure: src/environment, src/parsers, src/client, src/api, src/utils, tests/, data/, scripts/, configs/
  - Create requirements.txt with FastAPI, Pydantic, spaCy, NLTK, FuzzyWuzzy, Hypothesis, pytest
  - Initialize Python package with __init__.py files
  - _Requirements: 1.7, 6.1_

- [x] 2. Implement core data models
  - [x] 2.1 Create Episode and Claim dataclasses in src/environment/core.py
    - Implement Claim with claim_text, label, ground_truth_fact fields
    - Implement Episode with episode_id, source_dataset, difficulty_level, source_text, generated_response, claims, metadata fields
    - Add validation for label values in {"factual", "hallucinated", "unverifiable"}
    - Add validation for difficulty_level in {"L1", "L2", "L3", "L4"}
    - _Requirements: 1.6, 1.7, 1.2_
  
  - [x] 2.2 Create DetectionOutput and API models in src/api/models.py
    - Implement DetectedClaim with claim_text, label, reason, corrected_fact fields
    - Implement DetectionOutput with detected_claims list
    - Implement Observation, Action, StepResult Pydantic models
    - _Requirements: 3.5, 3.6, 6.2, 6.3_

- [x] 3. Implement EpisodeBank class
  - [x] 3.1 Create EpisodeBank in src/environment/episode_bank.py
    - Implement load_episodes() to load JSON files from data directory
    - Implement sample_episode() with difficulty level filtering
    - Implement get_episode_by_id() for specific episode retrieval
    - Implement get_statistics() for episode count and difficulty distribution
    - Implement _assign_difficulty() heuristic based on claim count and complexity
    - _Requirements: 1.1, 1.2, 1.7_
  
  - [ ]* 3.2 Write unit tests for EpisodeBank
    - Test episode loading from JSON files
    - Test difficulty assignment heuristics
    - Test sampling with different enabled difficulty levels
    - Test episode structure validation
    - _Requirements: 1.1, 1.2, 1.7_

- [x] 4. Implement RewardEngine class
  - [x] 4.1 Create RewardEngine in src/environment/reward.py
    - Implement calculate_reward() with full reward formula
    - Implement _match_claims() using fuzzy string matching (FuzzyWuzzy ratio > 70%)
    - Implement _calculate_correction_bonus() using keyword overlap (Jaccard similarity)
    - Implement _check_gaming_penalty() for >80% flagged rate
    - Implement _check_passivity_penalty() for <5% flagged rate with hallucinations
    - Apply base rewards: TP=+3.0, FP=-2.0, FN=-1.5, TN=+0.5
    - Apply calibration bonus: +1.0 if precision > 0.6 AND recall > 0.6
    - Apply difficulty multipliers: L1=1.0x, L2=1.5x, L3=2.0x, L4=2.5x
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.1, 5.2, 5.4_
  
  - [ ]* 4.2 Write unit tests for RewardEngine
    - Test base reward calculation for each confusion matrix component
    - Test correction bonus with varying keyword overlap
    - Test calibration bonus threshold (precision and recall > 0.6)
    - Test difficulty multipliers for all levels
    - Test gaming penalty (>80% flagged)
    - Test passivity penalty (<5% flagged with hallucinations)
    - Test anti-gaming property: flag-all scores lower than flag-none
    - _Requirements: 4.1-4.8, 5.1, 5.2, 5.4, 10.1-10.8_

- [x] 5. Implement CurriculumManager class
  - [x] 5.1 Create CurriculumManager in src/environment/curriculum.py
    - Implement __init__() with promotion_thresholds and window_size=50
    - Implement record_reward() to track rewards per difficulty level
    - Implement get_enabled_levels() returning currently enabled levels
    - Implement check_promotion() to enable next level when rolling avg exceeds threshold
    - Implement get_rolling_avg() for rolling average calculation (window size 50)
    - Initialize with only L1 enabled
    - Track rewards separately per difficulty level
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 5.2 Write unit tests for CurriculumManager
    - Test promotion logic with various reward sequences
    - Test rolling window calculation (window size 50)
    - Test initial state (only L1 enabled)
    - Test boundary conditions (reward exactly at threshold, just below, just above)
    - Test independent tracking per difficulty level
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [x] 6. Implement HallucinationEnvironment core class
  - [x] 6.1 Create HallucinationEnvironment in src/environment/core.py
    - Extend OpenEnv Environment base class
    - Implement reset() to sample episode and return observation with generated_text and task_instruction
    - Implement step() to accept detection output, calculate reward, return StepResult
    - Implement single-turn episode logic (done=True after first step)
    - Integrate EpisodeBank for episode sampling
    - Integrate CurriculumManager for difficulty progression
    - Integrate RewardEngine for reward calculation
    - Return episode metadata in info dict (episode_id, difficulty_level, source_dataset)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 6.2, 6.3, 6.4, 6.5_
  
  - [ ]* 6.2 Write unit tests for HallucinationEnvironment
    - Test reset() returns valid observation structure
    - Test step() returns valid StepResult structure
    - Test single-turn episode completion (done=True after first step)
    - Test episode metadata in info dict
    - Test integration with EpisodeBank, CurriculumManager, RewardEngine
    - _Requirements: 3.1, 6.2, 6.3, 6.4, 6.5_

- [x] 7. Checkpoint - Ensure all core environment tests pass
  - Ensure all tests pass, ask the user if questions arise.

### Phase 2: Data Preprocessing

- [x] 8. Implement claim extraction utilities
  - [x] 8.1 Create claim extraction in src/utils/claim_extraction.py
    - Implement extract_claims() using spaCy sentence segmentation
    - Implement split_on_conjunctions() for compound sentences
    - Implement is_declarative() to filter questions and imperatives
    - Download and configure spaCy en_core_web_sm model
    - _Requirements: 9.4, 9.5_
  
  - [ ]* 8.2 Write unit tests for claim extraction
    - Test sentence segmentation with various text structures
    - Test compound sentence splitting
    - Test declarative statement filtering
    - _Requirements: 9.4, 9.5_

- [x] 9. Implement dataset parsers
  - [x] 9.1 Create HaluEval parser in src/parsers/halueval.py
    - Parse JSON format with question, answer, hallucinated_answer, label fields
    - Extract claims from hallucinated_answer using claim extraction
    - Assign ground truth labels to each claim
    - Create Episode objects with source_dataset="halueval_qa"
    - _Requirements: 1.3, 9.1, 9.4, 9.5, 9.6_
  
  - [x] 9.2 Create TruthfulQA parser in src/parsers/truthfulqa.py
    - Parse format with question, best_answer, incorrect_answers fields
    - Generate LLM responses using Mistral-7B or Phi-3
    - Extract claims from generated responses
    - Assign ground truth labels based on best_answer
    - Create Episode objects with source_dataset="truthfulqa"
    - _Requirements: 1.4, 9.2, 9.4, 9.5, 9.6_
  
  - [x] 9.3 Create Wikipedia synthetic parser in src/parsers/wikipedia.py
    - Parse format with paragraph, summary, fact_labels fields
    - Extract claims from LLM-generated summaries
    - Assign ground truth labels from fact_labels
    - Create Episode objects with source_dataset="wikipedia_synthetic"
    - _Requirements: 1.5, 9.3, 9.4, 9.5, 9.6_
  
  - [ ]* 9.4 Write unit tests for parsers
    - Test HaluEval parser with valid and invalid inputs
    - Test TruthfulQA parser with valid and invalid inputs
    - Test Wikipedia parser with valid and invalid inputs
    - Test claim extraction produces at least one claim per episode
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_

- [x] 10. Preprocess datasets to create episode bank
  - [x] 10.1 Create preprocessing script in scripts/preprocess_datasets.py
    - Load raw HaluEval dataset and parse to episodes
    - Load raw TruthfulQA dataset and parse to episodes
    - Generate Wikipedia synthetic episodes
    - Assign difficulty levels using EpisodeBank._assign_difficulty()
    - Save episodes as JSON files in data/episodes/{source_dataset}/
    - Validate total episode count >= 1000
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  
  - [x] 10.2 Run preprocessing and validate episode bank
    - Execute preprocessing script
    - Verify episode count >= 1000
    - Verify difficulty distribution is reasonable (L1: ~25%, L2: ~35%, L3: ~25%, L4: ~15%)
    - Verify all episodes have required fields
    - _Requirements: 1.1, 1.2, 1.7_

- [x] 11. Checkpoint - Ensure episode bank is complete
  - Ensure all tests pass, ask the user if questions arise.

### Phase 3: API Server

- [x] 12. Implement FastAPI server
  - [x] 12.1 Create API server in src/api/server.py
    - Implement POST /reset endpoint returning observation and info
    - Implement POST /step endpoint accepting action, returning reward, observation, done, info
    - Implement GET /health endpoint returning status, episode_count, difficulty_distribution, curriculum_state
    - Implement GET /docs endpoint with auto-generated API documentation
    - Add request/response validation using Pydantic models
    - _Requirements: 6.1, 6.2, 6.3, 11.2, 11.3_
  
  - [x] 12.2 Implement rate limiting middleware
    - Add rate limiting to prevent abuse (60 requests per minute)
    - Return HTTP 429 with Retry-After header when limit exceeded
    - _Requirements: 11.5_
  
  - [x] 12.3 Implement concurrent request handling
    - Add thread-safe episode sampling with locks
    - Add thread-safe curriculum updates with locks
    - Support multiple concurrent training clients
    - _Requirements: 11.4_
  
  - [ ]* 12.4 Write API integration tests
    - Test /reset endpoint with various configurations
    - Test /step endpoint with valid and invalid detection outputs
    - Test /health endpoint returns correct statistics
    - Test concurrent requests from multiple clients
    - Test rate limiting behavior under load
    - _Requirements: 6.1, 6.2, 6.3, 11.2, 11.4, 11.5_

- [x] 13. Create deployment entry point
  - [x] 13.1 Create app.py entry point
    - Initialize EpisodeBank and load episodes from data/episodes
    - Initialize CurriculumManager with promotion thresholds from configs/curriculum.yaml
    - Initialize RewardEngine
    - Initialize HallucinationEnvironment
    - Create FastAPI app with create_app()
    - Configure Uvicorn server on port 7860
    - _Requirements: 6.1, 11.1, 11.6_

- [x] 14. Checkpoint - Ensure API server runs locally
  - Ensure all tests pass, ask the user if questions arise.

### Phase 4: Client Wrapper

- [x] 15. Implement environment client
  - [x] 15.1 Create HallucinationHunterEnv in src/client/env_client.py
    - Extend OpenEnv EnvClient base class
    - Implement WebSocket connection to FastAPI server
    - Implement reset() method calling POST /reset
    - Implement step() method calling POST /step
    - Implement synchronous and asynchronous interfaces
    - Serialize actions and deserialize observations
    - _Requirements: 6.1, 6.2, 6.3, 7.3_
  
  - [ ]* 15.2 Write client integration tests
    - Test client connection to server
    - Test reset() returns valid observation
    - Test step() returns valid StepResult
    - Test synchronous and asynchronous interfaces
    - Test error handling for connection failures
    - _Requirements: 6.1, 6.2, 6.3, 7.3_

- [x] 16. Add TRL compatibility layer
  - [x] 16.1 Implement TRL compatibility in src/client/env_client.py
    - Add support for 8 parallel generations per prompt
    - Implement batch reset() for multiple episodes
    - Implement batch step() for multiple detection outputs
    - Ensure compatibility with HuggingFace TRL GRPOTrainer
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 17. Checkpoint - Ensure client connects to server
  - Ensure all tests pass, ask the user if questions arise.

### Phase 5: Metrics and Logging

- [x] 18. Implement metrics calculation
  - [x] 18.1 Create metrics utilities in src/utils/metrics.py
    - Implement calculate_precision() as TP/(TP+FP)
    - Implement calculate_recall() as TP/(TP+FN)
    - Implement calculate_f1() as 2*precision*recall/(precision+recall)
    - Implement calculate_confusion_matrix() from matched claims
    - _Requirements: 8.1, 8.2_
  
  - [ ]* 18.2 Write unit tests for metrics
    - Test precision calculation with various confusion matrices
    - Test recall calculation with various confusion matrices
    - Test F1 calculation
    - Test confusion matrix calculation from matched claims
    - _Requirements: 8.1, 8.2_

- [x] 19. Implement metrics logging
  - [x] 19.1 Create metrics logger in src/utils/metrics.py
    - Implement EpisodeMetrics dataclass with all fields
    - Implement log_episode() to record episode-level metrics
    - Implement track_cumulative_reward() over training steps
    - Implement track_reward_per_difficulty() separately for each level
    - Implement calculate_rolling_average() with configurable window size
    - Save metrics to JSON files with timestamps
    - _Requirements: 7.5, 8.3, 8.4, 8.6_
  
  - [ ]* 19.2 Write unit tests for metrics logging
    - Test episode logging with all fields
    - Test cumulative reward tracking
    - Test reward tracking per difficulty level
    - Test rolling average calculation with different window sizes
    - _Requirements: 7.5, 8.3, 8.4, 8.6_

- [x] 20. Implement visualization data export
  - [x] 20.1 Create export utilities in src/utils/metrics.py
    - Implement export_time_series() for reward per difficulty level
    - Implement export_precision_recall_curves() over training steps
    - Implement export_cumulative_reward() over training steps
    - Export in JSON format compatible with matplotlib and plotly
    - _Requirements: 8.5, 12.1, 12.2, 12.6_
  
  - [ ]* 20.2 Write unit tests for export
    - Test time series export format
    - Test precision/recall curve export format
    - Test cumulative reward export format
    - Test JSON format is parseable by plotting libraries
    - _Requirements: 8.5, 12.1, 12.2, 12.6_

- [x] 21. Implement comparison mode
  - [x] 21.1 Create comparison utilities in src/utils/metrics.py
    - Implement evaluate_model() to run model on test episodes
    - Implement compare_models() to evaluate baseline and trained models
    - Implement calculate_aggregate_metrics() with baseline_reward, trained_reward, baseline_precision, baseline_recall, trained_precision, trained_recall
    - Implement generate_detection_diff() highlighting label and corrected_fact differences
    - _Requirements: 12.3, 12.4, 12.5_
  
  - [ ]* 21.2 Write unit tests for comparison mode
    - Test model evaluation on test episodes
    - Test comparison between baseline and trained models
    - Test aggregate metrics calculation
    - Test detection diff generation
    - _Requirements: 12.3, 12.4, 12.5_

- [x] 22. Checkpoint - Ensure metrics logging works
  - Ensure all tests pass, ask the user if questions arise.

### Phase 6: Property-Based Testing

- [ ] 23. Set up property-based testing framework
  - [ ] 23.1 Create Hypothesis strategies in tests/property/strategies.py
    - Create strategy for generating random Episodes
    - Create strategy for generating random DetectionOutputs
    - Create strategy for generating random reward sequences
    - Create strategy for generating random confusion matrices
    - Configure Hypothesis with min_iterations=100
    - _Requirements: All requirements (property testing validates all)_

- [ ] 24. Implement property tests for episode and curriculum (Properties 1-8)
  - [ ]* 24.1 Write property tests in tests/property/test_properties_1_10.py
    - **Property 1: Episode Structure Completeness** - Validates: Requirements 1.7
    - **Property 2: Difficulty Level Validity** - Validates: Requirements 1.2
    - **Property 3: Claim Decomposition Completeness** - Validates: Requirements 1.6, 9.5, 9.6
    - **Property 4: Curriculum Promotion Threshold** - Validates: Requirements 2.2
    - **Property 5: Rolling Window Size Invariant** - Validates: Requirements 2.5
    - **Property 6: Curriculum Sampling Constraint** - Validates: Requirements 2.4
    - **Property 7: Initial Curriculum State** - Validates: Requirements 2.1
    - **Property 8: Independent Difficulty Tracking** - Validates: Requirements 2.3, 8.4

- [ ] 25. Implement property tests for rewards (Properties 9-16)
  - [ ]* 25.1 Write property tests in tests/property/test_properties_1_10.py (continued)
    - **Property 9: Base Reward Calculation** - Validates: Requirements 4.1, 4.2, 4.3, 4.4
    - **Property 10: Correction Bonus Monotonicity** - Validates: Requirements 4.5
  
  - [ ]* 25.2 Write property tests in tests/property/test_properties_11_20.py
    - **Property 11: Calibration Bonus Threshold** - Validates: Requirements 4.6
    - **Property 12: Difficulty Multiplier Application** - Validates: Requirements 4.7
    - **Property 13: Anti-Gaming Dominance** - Validates: Requirements 4.8
    - **Property 14: Gaming Penalty Threshold** - Validates: Requirements 5.1
    - **Property 15: Passivity Penalty Threshold** - Validates: Requirements 5.2
    - **Property 16: Selective Flagging Optimality** - Validates: Requirements 5.4

- [ ] 26. Implement property tests for API and parsers (Properties 17-19)
  - [ ]* 26.1 Write property tests in tests/property/test_properties_11_20.py (continued)
    - **Property 17: Reset Observation Structure** - Validates: Requirements 3.1, 6.2
    - **Property 18: Task Instruction Completeness** - Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6
    - **Property 19: Dataset Parser Validity** - Validates: Requirements 9.1, 9.2, 9.3, 9.4

- [ ] 27. Implement property tests for episodes and metrics (Properties 20-27)
  - [ ]* 27.1 Write property tests in tests/property/test_properties_11_20.py (continued)
    - **Property 20: Step Response Structure** - Validates: Requirements 6.3
  
  - [ ]* 27.2 Write property tests in tests/property/test_properties_21_32.py
    - **Property 21: Single-Turn Episode Completion** - Validates: Requirements 6.4, 6.5
    - **Property 22: Independent Scoring for Parallel Generations** - Validates: Requirements 7.2
    - **Property 23: Episode Interaction Logging** - Validates: Requirements 7.5
    - **Property 24: Precision and Recall Calculation** - Validates: Requirements 8.1, 8.2
    - **Property 25: Cumulative Reward Tracking** - Validates: Requirements 8.3
    - **Property 26: Rolling Average Calculation** - Validates: Requirements 8.6
    - **Property 27: Metrics Export Format** - Validates: Requirements 8.5, 12.6

- [ ] 28. Implement property tests for health and visualization (Properties 28-32)
  - [ ]* 28.1 Write property tests in tests/property/test_properties_21_32.py (continued)
    - **Property 28: Health Endpoint Response** - Validates: Requirements 11.2
    - **Property 29: Rate Limiting Enforcement** - Validates: Requirements 11.5
    - **Property 30: Visualization Data Completeness** - Validates: Requirements 12.1, 12.2
    - **Property 31: Comparison Mode Metrics** - Validates: Requirements 12.5
    - **Property 32: Detection Output Diff Generation** - Validates: Requirements 12.4

- [ ] 29. Run all property tests and verify coverage
  - [ ] 29.1 Execute all property tests with 100+ iterations
    - Run tests/property/test_properties_1_10.py
    - Run tests/property/test_properties_11_20.py
    - Run tests/property/test_properties_21_32.py
    - Verify all 32 properties pass
    - Verify 100+ iterations per property

- [ ] 30. Checkpoint - Ensure all property tests pass
  - Ensure all tests pass, ask the user if questions arise.

### Phase 7: Training Integration

- [x] 31. Implement training script
  - [x] 31.1 Create training script in scripts/train_agent.py
    - Load Qwen2.5-7B-Instruct with Unsloth 4-bit quantization
    - Add LoRA adapters (r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    - Initialize HallucinationHunterEnv client
    - Configure GRPOConfig with num_generations=8, learning_rate=1e-5, max_steps=1000
    - Create GRPOTrainer with model, tokenizer, config, env
    - Implement checkpoint saving every 100 steps
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 31.2 Test training script with small episode subset
    - Run training for 10 steps on 50 episodes
    - Verify 8 parallel generations per prompt
    - Verify reward signals are passed to TRL correctly
    - Verify checkpoints are saved
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 32. Run full training
  - [ ] 32.1 Execute training for 1000 steps
    - Start FastAPI server locally
    - Run training script for 1000 steps on full episode bank
    - Monitor curriculum progression from L1 to L4
    - Monitor reward curves per difficulty level
    - Save final model to ./final_model
    - _Requirements: 2.1, 2.2, 2.4, 7.1, 7.2, 7.3, 7.4_
  
  - [ ] 32.2 Verify training results
    - Verify curriculum progressed through all difficulty levels
    - Verify reward increased over training
    - Verify final model saved successfully
    - _Requirements: 2.1, 2.2, 2.4_

- [x] 33. Checkpoint - Ensure training completes successfully
  - Ensure all tests pass, ask the user if questions arise.

### Phase 8: Deployment

- [x] 34. Create deployment configuration
  - [x] 34.1 Create Dockerfile
    - Use python:3.10-slim base image
    - Install system dependencies (build-essential)
    - Install Python dependencies from requirements.txt
    - Download spaCy en_core_web_sm model
    - Copy application code (src/, data/, configs/, app.py)
    - Expose port 7860
    - Set CMD to run Uvicorn server
    - _Requirements: 11.1, 11.6_
  
  - [x] 34.2 Create HuggingFace Spaces configuration
    - Create configs/deployment.yaml with title, emoji, sdk, hardware specs
    - Configure environment variables (EPISODE_BANK_PATH, LOG_LEVEL, RATE_LIMIT_PER_MINUTE, MAX_CONCURRENT_REQUESTS)
    - Configure hardware: 4 CPU, 16GB memory, 10GB storage
    - _Requirements: 11.1, 11.6_

- [ ] 35. Deploy to HuggingFace Spaces
  - [ ] 35.1 Deploy application
    - Push code to HuggingFace Spaces repository
    - Verify Docker build succeeds
    - Verify server starts successfully
    - Verify episode bank loads from persistent storage
    - _Requirements: 11.1, 11.6_
  
  - [ ]* 35.2 Run smoke tests on deployed instance
    - Test /health endpoint returns 200 status
    - Test /docs endpoint serves API documentation
    - Verify episode count >= 1000
    - Verify all difficulty levels have episodes
    - Test rate limiting is enabled
    - _Requirements: 11.1, 11.2, 11.3, 11.5, 11.6_

- [ ] 36. Performance testing
  - [ ]* 36.1 Run load tests
    - Test 100 concurrent clients making reset/step requests
    - Verify response times remain under 500ms at p95
    - Verify no memory leaks over 10,000 episodes
    - Verify rate limiting prevents server overload
    - _Requirements: 11.4, 11.5_

- [ ] 37. Checkpoint - Ensure deployment is stable
  - Ensure all tests pass, ask the user if questions arise.

### Phase 9: Evaluation and Visualization

- [x] 38. Evaluate baseline model
  - [x] 38.1 Create evaluation script in scripts/evaluate.py
    - Load 100 test episodes from episode bank
    - Evaluate untrained Qwen2.5-7B-Instruct on test episodes
    - Calculate baseline metrics: reward, precision, recall
    - Save baseline results to JSON
    - _Requirements: 12.3, 12.5_

- [x] 39. Evaluate trained model
  - [x] 39.1 Evaluate trained model on test episodes
    - Load trained model from ./final_model
    - Evaluate on same 100 test episodes
    - Calculate trained metrics: reward, precision, recall
    - Save trained results to JSON
    - _Requirements: 12.3, 12.5_
  
  - [x] 39.2 Verify training improvements
    - Verify trained reward > baseline reward
    - Verify trained precision > baseline precision
    - Verify trained recall > baseline recall
    - Target: trained reward > 3.0, precision > 0.7, recall > 0.7
    - _Requirements: 12.3, 12.5_

- [x] 40. Generate visualizations
  - [x] 40.1 Create visualization script in scripts/visualize.py
    - Generate reward curves per difficulty level (L1, L2, L3, L4)
    - Generate precision curve over training steps
    - Generate recall curve over training steps
    - Generate cumulative reward curve over training steps
    - Generate before-and-after detection examples with highlighted differences
    - Save all plots as PNG files
    - _Requirements: 12.1, 12.2, 12.4, 12.6_
  
  - [x] 40.2 Generate comparison report
    - Calculate aggregate metrics: baseline_reward, trained_reward, baseline_precision, baseline_recall, trained_precision, trained_recall
    - Generate comparison table
    - Generate detection diff examples (5-10 episodes)
    - Export all data in JSON format
    - _Requirements: 12.3, 12.4, 12.5, 12.6_

- [x] 41. Create README documentation
  - [x] 41.1 Write README.md
    - Add project overview and key features
    - Add installation instructions
    - Add usage examples (starting server, training, evaluation)
    - Add API documentation links
    - Add training results and visualizations
    - Add comparison metrics (baseline vs trained)
    - Add deployment instructions for HuggingFace Spaces
    - _Requirements: 11.3, 12.1, 12.2, 12.3, 12.5_

- [x] 42. Final checkpoint - Ensure all deliverables are complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at the end of each phase
- Property tests validate all 32 correctness properties with 100+ iterations each
- The implementation uses Python 3.10+ with FastAPI, Pydantic, spaCy, TRL, and Hypothesis
- Training uses Qwen2.5-7B-Instruct with Unsloth 4-bit quantization and LoRA fine-tuning
- Deployment targets HuggingFace Spaces with Docker containerization
- Expected training improvements: reward > 3.0, precision > 0.7, recall > 0.7
