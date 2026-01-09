# Resonant Vectors

A reinforcement learning system that uses vector similarity search and memory-based decision making. The project implements an "EngramBrain" that stores experiences in a vector database and uses nearest-neighbor search to make decisions based on similar past experiences.

This project is testing the idea that something like Sheldrakes "morphic resonance", or a memory of nature, can explain the emergence of complex instinctive behaviours. Such behaviours cannot easily be explained through genetics. However, if animal behaviour is affected by the previous behavior of similar animals, then successful behaviours will tend to create more historical instances of that behaviour, making it easier to "resonate" with an animal in the present moment.

## Overview

Resonant Vectors explores a novel approach to reinforcement learning where:

- **Engrams** (memory traces) store state-action-outcome triplets
- **Vector similarity search** finds relevant past experiences
- **Resonator vectors** encode state information for similarity matching
- **Vector databases** (FAISS or Milvus) provide fast nearest neighbor search

The system is trained on the LunarLander-v2 environment from Gymnasium, learning to land a spacecraft by recalling and learning from similar past situations.

## Architecture

### Core Components

- **`EngramBrain`**: The decision-making engine that queries similar experiences and generates actions
- **`EngramStore`**: Manages storage and retrieval of engrams (supports both Milvus and FAISS backends)
- **`Trainer`**: Orchestrates training on Gymnasium environments with metrics tracking
- **`WeightedResonatorFactory`**: Converts input states to resonator vectors with configurable component weights

### Vector Store Backends

The system supports two vector store implementations:

- **Milvus**: Persistent vector database with Docker deployment (good for long-running experiments)
- **FAISS**: In-memory vector store (fast, no external dependencies, perfect for development/testing)

Configure via `VECTOR_STORE_TYPE` in `settings.py` (default: `"faiss"`).

### How It Works

1. **Observation**: The agent receives a state observation from the environment
2. **Resonator Creation**: The state is converted to a resonator vector (with optional component weighting)
3. **Similarity Search**: The vector store finds the nearest engrams to the resonator vector
4. **Action Selection**: Actions are scored based on similar past experiences and their outcomes (with optional decay and trial success multipliers)
5. **Learning**: After each trial, new engrams are created and stored with their outcomes (with optional sampling)

## Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Virtual environment (recommended)

### Installation

1. **Create a virtual environment**:

```bash
python -m venv venv
```

2. **Activate the virtual environment**:

```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### Install and Run Milvus (Optional)

**Note**: Milvus is optional. The default configuration uses FAISS (in-memory), which requires no setup.

To use Milvus as the vector store backend:

```bash
docker compose up -d
```

Verify Milvus is running:

```bash
docker compose ps
```

Milvus will be available at `localhost:19530`. The project includes a `docker-compose.yml` file with Milvus, etcd, and MinIO configured.

## Usage

### Training a Single Agent

Train an agent with default settings:

```python
from Trainer import Trainer

trainer = Trainer("lander3", clear_collection=True)
trainer.train(1000)  # Run 1000 trials
```

Parameters:

- `instance_name`: Unique name for the vector store collection
- `clear_collection`: Whether to reset the collection before training

### Running the Main Script

The `main.py` file runs the trainer:

```bash
python main.py
```

To continue training without clearing the collection:

```bash
python main.py --no-clear
```

The script handles graceful shutdown (Ctrl+C) and automatically saves metrics.

## Configuration

Edit `settings.py` to customize behavior. Key settings include:

### Basic Settings

- `STATE_VECTOR_SIZE`: Dimension of state vectors (default: 8)
- `OUTPUT_VECTOR_SIZE`: Number of possible actions (default: 4)
- `NOISE_START`: Initial noise at episode start (default: 0.3)
- `NOISE_END`: Target noise at episode end (default: 0.05)
- `NOISE_DECAY_RATE`: Controls noise decay speed - higher = faster decay (default: 3.0)
- `NOISE`: Backward compatibility alias for `NOISE_START` (default: 0.3)
- `MIN_RESULTS`: Minimum number of similar engrams to retrieve (default: 400)
- `MAX_TRIAL_LENGTH`: Maximum steps per trial (default: 400)
- `TRIALS_PER_EXPERIMENT`: Number of trials per training run (default: 500)

**Noise Decay**: Noise starts at `NOISE_START` at the beginning of each episode and exponentially decays toward `NOISE_END` as the episode progresses. This encourages exploration early in episodes while allowing more exploitation later.

### Vector Store

- `VECTOR_STORE_TYPE`: Backend selection - `"milvus"` or `"faiss"` (default: `"faiss"`)
- `DROP_COLLECTION`: Drop collection on initialization (default: False)
- `VECTOR_SAVE_RATE`: Fraction of vectors to randomly sample and save, 0.0-1.0 (default: 0.2)
- `DELETE_OLDEST_BEFORE_INSERT`: Maintain constant collection size by deleting oldest records (default: False)

### Learning & Action Selection

- `PROBABILISTIC_CHOICE`: Use probabilistic action selection vs. argmax (default: True)
- `READ_ONLY`: Disable learning/engram storage (default: False)
- `VECTOR_COMPONENT_WEIGHTS`: Weighting for each state component in similarity calculations (8 weights)

### Hit Points System

- `USE_HIT_POINTS`: Enable hit points system (default: True)
- `HIT_POINTS`: Initial hit points (default: 500)
- `METABOLIC_COST`: Energy cost per step (default: 0.2)

### Advanced Features

- `PANIC_ENABLED`: Enable panic mode with increased noise (default: False)
- `DECAY_ENABLED`: Enable decay-based ranking by insertion ID (default: False)
- `TRIAL_SUCCESS_MULTIPLIER_SCALE`: Weight trial success when scoring vectors (default: 0)

### Display

- `DISPLAY`: Show the environment visualization (default: False)
- `SHOW_ACTION_OUTPUT`: Display action selection details (default: False)

See `settings.py` for complete documentation of all configuration options.

## Project Structure

```
.
├── main.py                    # Entry point - runs trainer with graceful shutdown
├── EngramBrain.py             # Core decision-making system
├── engram.py                  # Engram data structure and Milvus store implementation
├── faiss_store.py             # FAISS-based in-memory vector store
├── Trainer.py                 # Training orchestration with metrics tracking
├── WeightedResonatorFactory.py # State-to-resonator conversion with weights
├── IResonatorFactory.py       # Interface for resonator factories
├── settings.py                # Configuration parameters
├── hello_milvus.py            # Milvus connection test script
├── metrics_plotter.py         # Visualization of training metrics
├── analyze_trials.py          # Analysis tools for experiment results
├── weight_optimizer.py        # Optimization utilities
├── min_results_optimizer.py   # MIN_RESULTS parameter optimization
├── check_ids.py               # Utility for checking insertion IDs
├── gym/                       # Custom gym environments
│   ├── lander_environment.py
│   └── lander.py
├── experiments/               # Experiment output directories
│   └── [experiment_id]/       # Contains metrics, plots, and settings
├── tests/                     # Test suite
│   ├── test_engram_brain.py
│   ├── test_engram_store.py
│   ├── test_engram.py
│   ├── test_resonator_factory.py
│   ├── test_trainer.py
│   └── test_utils.py
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Milvus, etcd, and MinIO configuration
└── volumes/                   # Persistent storage for Docker services
```

## Key Concepts

### Engrams

An **engram** represents a memory trace containing:

- `vector`: The state/resonator vector at the time of the experience
- `action`: The action taken
- `outcome`: The reward/outcome of that action (normalized to -1 to 1)

### Resonators

A **resonator** is a transformed version of the input state, optimized for similarity matching. The `WeightedResonatorFactory` appends success metrics to the state vector.

### Similarity Search

The system uses L2 (Euclidean) distance to find the most similar past experiences:

- **FAISS**: Uses `IndexFlatL2` for exact nearest neighbor search
- **Milvus**: Uses `IVF_FLAT` index which provides a balance between search speed and accuracy

## Dependencies

Key dependencies include:

- `faiss-cpu`: FAISS vector similarity search (for FAISS backend)
- `pymilvus`: Milvus Python client (for Milvus backend)
- `gymnasium`: Reinforcement learning environments
- `numpy`: Numerical computations
- `pandas`: Data analysis and metrics
- `box2d-py`: Physics engine for LunarLander
- `pytest`: Testing framework

See `requirements.txt` for the complete list.

## Testing

Run the test suite:

```bash
pytest
```

Run specific test files:

```bash
pytest tests/test_engram_brain.py
pytest tests/test_engram_store.py
```

## Notes

- The system learns online during training, storing engrams after each trial
- Success metrics are normalized and used to weight engram outcomes
- The feedback queue batches updates for efficiency
- Multiple trainers can use separate collections for parallel training
- FAISS backend is in-memory only - data is lost when the process exits
- Milvus backend persists data across runs
- Vector sampling (`VECTOR_SAVE_RATE`) can reduce memory usage while maintaining performance

## License

[Add your license here]
