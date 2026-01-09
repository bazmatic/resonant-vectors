import pytest
import numpy as np
from unittest.mock import MagicMock
from engram import Engram, EngramField


class TestEngram:
    """Tests for Engram class."""
    
    def test_init_with_all_fields(self):
        """Test Engram initialization with all fields."""
        vector = [0.1, 0.2, 0.3]
        action = 2
        outcome = 0.5
        trial_final_success = 0.8
        
        engram = Engram(
            vector=vector,
            action=action,
            outcome=outcome,
            trial_final_success=trial_final_success
        )
        
        assert engram.vector == vector
        assert engram.action == action
        assert engram.outcome == outcome
        assert engram.trial_final_success == trial_final_success
    
    def test_init_with_defaults(self):
        """Test Engram initialization with default values."""
        vector = [0.1, 0.2, 0.3]
        action = 1
        outcome = -0.5
        
        engram = Engram(vector=vector, action=action, outcome=outcome)
        
        assert engram.vector == vector
        assert engram.action == action
        assert engram.outcome == outcome
        assert engram.trial_final_success == 0.0
    
    def test_from_record_with_all_fields(self, sample_milvus_record):
        """Test creating Engram from Milvus record with all fields."""
        engram = Engram.from_record(sample_milvus_record)
        
        assert engram.vector == sample_milvus_record.fields[EngramField.vector]
        assert engram.action == sample_milvus_record.fields[EngramField.action]
        assert engram.outcome == sample_milvus_record.fields[EngramField.outcome]
        assert engram.trial_final_success == sample_milvus_record.fields[EngramField.trial_final_success]
    
    def test_from_record_with_missing_fields(self):
        """Test creating Engram from record with missing optional fields."""
        record = MagicMock()
        record.fields = {
            EngramField.vector: [0.1, 0.2, 0.3],
            EngramField.action: 1,
            EngramField.outcome: 0.5,
        }
        
        engram = Engram.from_record(record)
        
        assert engram.vector == [0.1, 0.2, 0.3]
        assert engram.action == 1
        assert engram.outcome == 0.5
        assert engram.trial_final_success == 0.0
    
    def test_random_creates_valid_engram(self):
        """Test random engram generation with correct dimensions and ranges."""
        input_size = 8
        action_options = 4
        
        engram = Engram.random(input_size, action_options)
        
        assert len(engram.vector) == input_size
        assert all(0.0 <= v <= 1.0 for v in engram.vector)
        assert 0 <= engram.action < action_options
        assert -1.0 <= engram.outcome <= 1.0
    
    def test_random_action_bounds(self):
        """Test that random action is within valid bounds."""
        input_size = 5
        action_options = 4
        
        # Generate multiple random engrams to test bounds
        for _ in range(100):
            engram = Engram.random(input_size, action_options)
            assert 0 <= engram.action < action_options
    
    def test_random_outcome_range(self):
        """Test that random outcome is within [-1, 1] range."""
        input_size = 5
        action_options = 4
        
        # Generate multiple random engrams to test range
        for _ in range(100):
            engram = Engram.random(input_size, action_options)
            assert -1.0 <= engram.outcome <= 1.0
    
    def test_random_vector_dimensions(self):
        """Test that random vector has correct dimensions."""
        for input_size in [1, 5, 8, 100]:
            engram = Engram.random(input_size, 4)
            assert len(engram.vector) == input_size

