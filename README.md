# Resonant Vectors

A reinforcement learning system that uses vector similarity search and memory-based decision making. The project implements an "EngramBrain" that stores experiences in a Milvus vector database and uses nearest-neighbor search to make decisions based on similar past experiences.

This project is testing the idea that something like Sheldrakes "morphic resonance", or a memory of nature, can explain the emergence of complex instinctive behaviours. Such behaviours cannot easily be explained through genetics. However, if animal behaviour is affected by the previous behavior of similar animals, then successful behaviours will tend to create more historical instances of that behaviour, making it easier to "resonate" with an animal in the present moment.

## Overview

Resonant Vectors explores a novel approach to reinforcement learning where:
- **Engrams** (memory traces) store state-action-outcome triplets
- **Vector similarity search** finds relevant past experiences
- **Resonator vectors** encode state information for similarity matching
- **Milvus** provides fast approximate nearest neighbor search

The system is trained on the LunarLander-v2 environment from Gymnasium, learning to land a spacecraft by recalling and learning from similar past situations.

## Architecture

### Core Components

- **`EngramBrain`**: The decision-making engine that queries similar experiences and generates actions
- **`EngramStore`**: Manages storage and retrieval of engrams in Milvus
- **`Trainer`**: Orchestrates training on Gymnasium environments
- **`Breeder`**: Genetic algorithm for evolving resonator dimension weights
- **`WeightedResonatorFactory`**: Converts input states to resonator vectors using learned weights

### How It Works

1. **Observation**: The agent receives a state observation from the environment
2. **Resonator Creation**: The state is converted to a resonator vector (with optional dimension weighting)
3. **Similarity Search**: Milvus finds the nearest engrams to the resonator vector
4. **Action Selection**: Actions are scored based on similar past experiences and their outcomes
5. **Learning**: After each trial, new engrams are created and stored with their outcomes

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

### Install and Run Milvus

Milvus is required as the vector database backend:

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -O docker-compose.yml

docker compose up -d
```

Verify Milvus is running:
```bash
docker compose ps
```

Milvus will be available at `localhost:19530`.

## Usage

### Training a Single Agent

Train an agent with default settings:

```python
from Trainer import Trainer

trainer = Trainer("lander3", [1, 1, 1, 1, 1, 1, 1, 1], clear_collection=True)
trainer.train(1000)  # Run 1000 trials
```

Parameters:
- `instance_name`: Unique name for the Milvus collection
- `dimension_weights`: Weights for each state dimension (8 values for LunarLander)
- `clear_collection`: Whether to reset the collection before training

### Genetic Algorithm Breeding

Evolve optimal dimension weights using the Breeder:

```python
from Breeder import Breeder

breeder = Breeder(population_size=10, genome_size=8)
breeder.run()  # Run one generation
```

The `Breeder` evaluates each genome by training a `Trainer` and selects the best performers for breeding.

### Running the Main Script

The `main.py` file contains example usage:

```bash
python main.py
```

Currently configured to run the trainer, but can be modified to run the breeder instead.

## Configuration

Edit `settings.py` to customize behavior:

- `STATE_VECTOR_SIZE`: Dimension of state vectors (default: 9)
- `OUTPUT_VECTOR_SIZE`: Number of possible actions (default: 4)
- `NOISE`: Random noise added to action selection (default: 0.1)
- `MIN_RESULTS`: Minimum number of similar engrams to retrieve (default: 300)
- `MAX_TRIAL_LENGTH`: Maximum steps per trial (default: 400)
- `METABOLIC_COST`: Energy cost per step when using hit points (default: 0.2)
- `DISPLAY`: Show the environment visualization (default: True)
- `PROBABILISTIC_CHOICE`: Use probabilistic action selection vs. argmax (default: False)
- `READ_ONLY`: Disable learning/engram storage (default: False)
- `DROP_COLLECTION`: Drop collection on initialization (default: True)

## Project Structure

```
.
├── main.py                    # Entry point with example usage
├── EngramBrain.py             # Core decision-making system
├── engram.py                  # Engram data structure and Milvus store
├── Trainer.py                 # Training orchestration
├── Breeder.py                 # Genetic algorithm for hyperparameter evolution
├── WeightedResonatorFactory.py # State-to-resonator conversion
├── IResonatorFactory.py       # Interface for resonator factories
├── settings.py                # Configuration parameters
├── hello_milvus.py            # Milvus connection test script
├── gym/                       # Custom gym environments (if any)
│   ├── lander_environment.py
│   └── lander.py
├── requirements.txt           # Python dependencies
└── docker-compose.yml         # Milvus configuration
```

## Key Concepts

### Engrams

An **engram** represents a memory trace containing:
- `vector`: The state/resonator vector at the time of the experience
- `action`: The action taken
- `outcome`: The reward/outcome of that action (normalized to -1 to 1)

### Resonators

A **resonator** is a transformed version of the input state, optimized for similarity matching. The `WeightedResonatorFactory` applies learned weights to different state dimensions and appends success metrics.

### Similarity Search

The system uses L2 (Euclidean) distance in Milvus to find the most similar past experiences. The IVF_FLAT index provides a balance between search speed and accuracy.

## Dependencies

Key dependencies include:
- `pymilvus`: Milvus Python client
- `gymnasium`: Reinforcement learning environments
- `numpy`: Numerical computations
- `box2d-py`: Physics engine for LunarLander

See `requirements.txt` for the complete list.

## Notes

- The system learns online during training, storing engrams after each trial
- Success metrics are normalized and used to weight engram outcomes
- The feedback queue batches updates for efficiency
- Multiple trainers can use separate Milvus collections for parallel training

## License

[Add your license here]
