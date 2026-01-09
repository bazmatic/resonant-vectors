import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple
from engram import Engram, EngramField
from settings import STATE_VECTOR_SIZE


@pytest.fixture
def mock_milvus_collection():
    """Mock Milvus collection with common methods."""
    collection = MagicMock()
    collection.num_entities = 0
    collection.load = MagicMock()
    collection.insert = MagicMock()
    collection.search = MagicMock(return_value=[[], []])
    collection.create_index = MagicMock()
    return collection


@pytest.fixture
def mock_milvus_utility():
    """Mock Milvus utility functions."""
    utility = MagicMock()
    utility.has_collection = MagicMock(return_value=False)
    utility.drop_collection = MagicMock()
    return utility


@pytest.fixture
def mock_milvus_connections():
    """Mock Milvus connections."""
    connections = MagicMock()
    connections.connect = MagicMock()
    return connections


@pytest.fixture
def mock_gym_environment():
    """Mock gymnasium environment."""
    env = MagicMock()
    env.reset = MagicMock(return_value=(np.array([0.0] * 8), {}))
    env.step = MagicMock(return_value=(np.array([0.0] * 8), 0.0, False, False, {}))
    env.close = MagicMock()
    return env


@pytest.fixture
def sample_vector():
    """Sample state vector for testing."""
    # Generate a vector of STATE_VECTOR_SIZE elements
    return [0.1 * (i + 1) % 1.0 for i in range(STATE_VECTOR_SIZE)]


@pytest.fixture
def sample_engram(sample_vector):
    """Sample Engram for testing."""
    return Engram(
        vector=sample_vector,
        action=1,
        outcome=0.5,
        trial_final_success=0.8
    )


@pytest.fixture
def sample_engrams(sample_vector):
    """Multiple sample Engrams for testing."""
    return [
        Engram(vector=[0.1 + 0.0] * STATE_VECTOR_SIZE, action=0, outcome=0.8),
        Engram(vector=[0.2 + 0.0] * STATE_VECTOR_SIZE, action=1, outcome=0.5),
        Engram(vector=[0.0 + 0.0] * STATE_VECTOR_SIZE, action=2, outcome=-0.3),
        Engram(vector=[0.3 + 0.0] * STATE_VECTOR_SIZE, action=3, outcome=0.2),
    ]


@pytest.fixture
def sample_milvus_record():
    """Mock Milvus record object."""
    record = MagicMock()
    record.fields = {
        EngramField.vector: [0.1 * (i + 1) % 1.0 for i in range(STATE_VECTOR_SIZE)],
        EngramField.action: 1,
        EngramField.outcome: 0.5,
        EngramField.trial_final_success: 0.8,
    }
    record.distance = 0.123
    record.entity = MagicMock()
    record.entity.get = MagicMock(return_value=1)
    return record


@pytest.fixture
def mock_resonator_factory():
    """Mock resonator factory."""
    factory = MagicMock()
    factory.make_resonator = MagicMock(side_effect=lambda input, success: input)  # Return only input, not append success
    return factory


@pytest.fixture
def mock_settings():
    """Mock settings constants."""
    with patch('settings.STATE_VECTOR_SIZE', STATE_VECTOR_SIZE), \
         patch('settings.OUTPUT_VECTOR_SIZE', 4), \
         patch('settings.PAST_HISTORY_STEPS', 8), \
         patch('settings.NOISE_START', 0.1), \
         patch('settings.NOISE_END', 0.1), \
         patch('settings.NOISE_DECAY_RATE', 3.0), \
         patch('settings.NOISE', 0.1), \
         patch('settings.MIN_RESULTS', 300), \
         patch('settings.DECAY_ENABLED', False), \
         patch('settings.DECAY_FUNCTION', 'gauss'), \
         patch('settings.DECAY_OFFSET_IDS', 40000), \
         patch('settings.DECAY_SCALE_IDS', 200000), \
         patch('settings.DECAY_VALUE', 0.9), \
         patch('settings.TRIAL_SUCCESS_MULTIPLIER_SCALE', 0.5), \
         patch('settings.PANIC_MAX_NOISE', 1.0):
        yield

