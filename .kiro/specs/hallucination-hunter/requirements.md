# Requirements Document

## Introduction

The Hallucination Hunter is an OpenEnv-compatible reinforcement learning training environment that trains language models to detect hallucinations in LLM-generated outputs at the individual claim level. The system provides a deterministic reward signal without human labeling, implements anti-gaming penalties, and uses curriculum learning to progressively improve detection capabilities across four difficulty levels.

## Glossary

- **Environment**: The OpenEnv-compatible RL training system that manages episodes, rewards, and curriculum progression
- **Agent**: The language model being trained to detect hallucinations
- **Episode**: A single training instance containing an LLM-generated text, ground truth, and metadata
- **Claim**: An individual factual assertion extracted from LLM-generated text
- **Hallucination**: A claim that contradicts or is unsupported by ground truth information
- **Reward_Engine**: The component that calculates deterministic scores based on detection accuracy
- **Episode_Bank**: The dataset storage and sampling system containing labeled training instances
- **Curriculum_Manager**: The component that controls difficulty progression based on performance thresholds
- **Detection_Output**: The Agent's structured response containing claim labels and corrections
- **Difficulty_Level**: A classification from L1 to L4 indicating episode complexity

## Requirements

### Requirement 1: Episode Dataset Management

**User Story:** As a training system, I want to manage a diverse dataset of labeled episodes, so that the Agent can learn from varied hallucination patterns.

#### Acceptance Criteria

1. THE Episode_Bank SHALL store at least 1000 labeled episodes from multiple sources
2. WHEN an episode is stored, THE Episode_Bank SHALL assign a Difficulty_Level from L1 to L4
3. THE Episode_Bank SHALL include episodes from HaluEval dataset with QA, summarization, and dialog samples
4. THE Episode_Bank SHALL include episodes from TruthfulQA with Mistral-7B or Phi-3 generated responses
5. THE Episode_Bank SHALL include episodes from Wikipedia paragraphs with LLM-generated summaries
6. WHEN an episode is created, THE Episode_Bank SHALL decompose the response into individual claims with separate labels
7. FOR ALL episodes, THE Episode_Bank SHALL store the source text, generated response, ground truth, claim decomposition, and hallucination labels

### Requirement 2: Episode Sampling and Curriculum

**User Story:** As a curriculum system, I want to progressively increase difficulty based on performance, so that the Agent learns systematically from simple to complex cases.

#### Acceptance Criteria

1. WHEN training begins, THE Curriculum_Manager SHALL sample only L1 episodes
2. WHEN the rolling average reward for a Difficulty_Level exceeds the promotion threshold, THE Curriculum_Manager SHALL enable sampling from the next Difficulty_Level
3. THE Curriculum_Manager SHALL track reward statistics separately for each Difficulty_Level
4. WHEN sampling an episode, THE Environment SHALL select from currently enabled Difficulty_Levels
5. THE Curriculum_Manager SHALL maintain a rolling window of the most recent 50 episode rewards for promotion decisions

### Requirement 3: Agent Task Interface

**User Story:** As an Agent, I want to receive clear task instructions and output format specifications, so that I can perform hallucination detection correctly.

#### Acceptance Criteria

1. WHEN an episode begins, THE Environment SHALL provide the Agent with the LLM-generated text to analyze
2. THE Environment SHALL instruct the Agent to identify each factual claim in the text
3. THE Environment SHALL instruct the Agent to label each claim as factual, hallucinated, or unverifiable
4. THE Environment SHALL instruct the Agent to provide the correct fact for each hallucinated claim
5. THE Environment SHALL require the Detection_Output in JSON format with claims, labels, and corrections
6. THE Environment SHALL specify the JSON schema with fields for claim_text, label, reason, and corrected_fact

### Requirement 4: Deterministic Reward Calculation

**User Story:** As a training system, I want to provide deterministic rewards without human labeling, so that training is reproducible and scalable.

#### Acceptance Criteria

1. WHEN the Agent correctly identifies a hallucinated claim, THE Reward_Engine SHALL award 3.0 points
2. WHEN the Agent incorrectly flags a factual claim as hallucinated, THE Reward_Engine SHALL deduct 2.0 points
3. WHEN the Agent fails to identify a hallucinated claim, THE Reward_Engine SHALL deduct 1.5 points
4. WHEN the Agent correctly identifies a factual claim as factual, THE Reward_Engine SHALL award 0.5 points
5. WHEN the Agent provides a correction with keyword overlap to the ground truth, THE Reward_Engine SHALL add a correction bonus proportional to overlap
6. WHEN the Agent achieves both precision and recall above 0.6 for an episode, THE Reward_Engine SHALL add a calibration bonus of 1.0 points
7. THE Reward_Engine SHALL multiply the base reward by the Difficulty_Level multiplier (L1: 1.0x, L2: 1.5x, L3: 2.0x, L4: 2.5x)
8. FOR ALL episodes, flagging all claims as hallucinated SHALL produce a lower total reward than flagging no claims

### Requirement 5: Anti-Gaming Penalty Structure

**User Story:** As a training system, I want to prevent reward gaming strategies, so that the Agent learns genuine detection skills.

#### Acceptance Criteria

1. WHEN the Agent flags more than 80% of claims as hallucinated, THE Reward_Engine SHALL apply a gaming penalty of -5.0 points
2. WHEN the Agent flags fewer than 5% of claims as hallucinated in episodes with known hallucinations, THE Reward_Engine SHALL apply a passivity penalty of -3.0 points
3. THE Reward_Engine SHALL ensure that the optimal strategy requires balanced precision and recall
4. FOR ALL episodes with mixed factual and hallucinated claims, the maximum reward SHALL require selective flagging

### Requirement 6: OpenEnv API Compliance

**User Story:** As an external training framework, I want to interact with the Environment through standard OpenEnv APIs, so that I can use existing RL tooling.

#### Acceptance Criteria

1. THE Environment SHALL expose a FastAPI server with OpenEnv-compatible endpoints
2. THE Environment SHALL implement a reset endpoint that returns an initial observation and episode metadata
3. THE Environment SHALL implement a step endpoint that accepts Detection_Output and returns reward, next observation, done flag, and info
4. WHEN an episode completes, THE Environment SHALL set the done flag to true
5. THE Environment SHALL support single-turn episodes where one detection task equals one step
6. THE Environment SHALL return episode metadata including Difficulty_Level and source dataset in the info dictionary

### Requirement 7: Training Pipeline Integration

**User Story:** As a model trainer, I want to train the Agent using GRPO with multiple generations per prompt, so that I can optimize detection performance.

#### Acceptance Criteria

1. THE Environment SHALL support 8 parallel generations per prompt for GRPO sampling
2. WHEN multiple generations are submitted for the same episode, THE Reward_Engine SHALL score each generation independently
3. THE Environment SHALL provide a client wrapper for API calls compatible with HuggingFace TRL
4. THE Environment SHALL support Qwen2.5-7B-Instruct with Unsloth 4-bit and LoRA fine-tuning
5. THE Environment SHALL log all episode interactions with timestamps, rewards, and Agent outputs for analysis

### Requirement 8: Performance Metrics Tracking

**User Story:** As a researcher, I want to track detailed performance metrics over training, so that I can measure improvement and diagnose issues.

#### Acceptance Criteria

1. THE Environment SHALL calculate and log precision for each episode based on true positives and false positives
2. THE Environment SHALL calculate and log recall for each episode based on true positives and false negatives
3. THE Environment SHALL track cumulative reward over training steps
4. THE Environment SHALL track reward separately for each Difficulty_Level
5. THE Environment SHALL export metrics in a format compatible with plotting libraries
6. THE Environment SHALL calculate rolling averages over configurable window sizes for smoothed metrics

### Requirement 9: Episode Format Parsing

**User Story:** As a preprocessing system, I want to parse episodes from multiple dataset formats, so that I can build a unified Episode_Bank.

#### Acceptance Criteria

1. THE Environment SHALL parse HaluEval dataset JSON format with question, answer, hallucinated_answer, and label fields
2. THE Environment SHALL parse TruthfulQA dataset format with question, best_answer, and incorrect_answers fields
3. THE Environment SHALL parse Wikipedia-based synthetic episodes with paragraph, summary, and fact_labels fields
4. WHEN parsing an episode, THE Environment SHALL extract individual claims using sentence segmentation and dependency parsing
5. WHEN parsing an episode, THE Environment SHALL assign ground truth labels to each extracted claim
6. THE Environment SHALL validate that each parsed episode contains at least one claim with a label

### Requirement 10: Reward Function Unit Testing

**User Story:** As a developer, I want comprehensive unit tests for the reward function, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE test suite SHALL verify that true positive detection awards exactly 3.0 base points
2. THE test suite SHALL verify that false positive detection deducts exactly 2.0 points
3. THE test suite SHALL verify that false negative detection deducts exactly 1.5 points
4. THE test suite SHALL verify that true negative detection awards exactly 0.5 points
5. THE test suite SHALL verify that Difficulty_Level multipliers are applied correctly
6. THE test suite SHALL verify that flagging all claims scores lower than flagging no claims
7. THE test suite SHALL verify that calibration bonus is awarded only when both precision and recall exceed 0.6
8. THE test suite SHALL verify that correction bonus increases with keyword overlap to ground truth

### Requirement 11: Deployment and Accessibility

**User Story:** As a hackathon participant, I want to deploy the Environment to a public endpoint, so that judges and users can interact with the system.

#### Acceptance Criteria

1. THE Environment SHALL be deployable to HuggingFace Spaces as a FastAPI application
2. THE Environment SHALL expose a health check endpoint that returns status and episode count
3. THE Environment SHALL include API documentation accessible via the /docs endpoint
4. THE Environment SHALL handle concurrent requests from multiple training clients
5. THE Environment SHALL implement rate limiting to prevent abuse
6. WHEN deployed, THE Environment SHALL load the Episode_Bank from persistent storage

### Requirement 12: Demonstration and Visualization

**User Story:** As a presenter, I want to generate visualizations and comparisons, so that I can demonstrate training effectiveness.

#### Acceptance Criteria

1. THE Environment SHALL export reward curves per Difficulty_Level as time series data
2. THE Environment SHALL export precision and recall over training steps as time series data
3. THE Environment SHALL provide a comparison mode that evaluates both untrained and trained models on the same episodes
4. THE Environment SHALL generate before-and-after detection examples with highlighted differences
5. THE Environment SHALL calculate aggregate metrics including baseline reward, trained reward, baseline precision, baseline recall, trained precision, and trained recall
6. THE Environment SHALL export all visualization data in JSON format compatible with plotting libraries
