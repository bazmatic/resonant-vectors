import pytest
from Trainer import normalise_observation


class TestNormaliseObservation:
    """Tests for normalise_observation utility function."""
    
    def test_normalise_observation_all_fields(self):
        """Test that all observation fields are normalized correctly to [-1, 1] range."""
        observation = [2.5, 2.5, 10.0, 10.0, 3.1415927, 10.0, 1.0, 1.0]
        
        result = normalise_observation(observation)
        
        assert result[0] == pytest.approx(1.0)  # 2.5 / 2.5
        assert result[1] == pytest.approx(1.0)  # 2.5 / 2.5
        assert result[2] == pytest.approx(1.0)  # 10.0 / 10.0
        assert result[3] == pytest.approx(1.0)  # 10.0 / 10.0
        assert result[4] == pytest.approx(1.0)  # 3.1415927 / 3.1415927
        assert result[5] == pytest.approx(1.0)  # 10.0 / 10.0
        assert result[6] == 1.0  # Leg contact: 1.0 → 1.0
        assert result[7] == 1.0  # Leg contact: 1.0 → 1.0
    
    def test_normalise_observation_negative_values(self):
        """Test normalization with negative values to [-1, 1] range."""
        observation = [-2.5, -2.5, -10.0, -10.0, -3.1415927, -10.0, 0.0, 0.0]
        
        result = normalise_observation(observation)
        
        assert result[0] == pytest.approx(-1.0)  # -2.5 / 2.5
        assert result[1] == pytest.approx(-1.0)  # -2.5 / 2.5
        assert result[2] == pytest.approx(-1.0)  # -10.0 / 10.0
        assert result[3] == pytest.approx(-1.0)  # -10.0 / 10.0
        assert result[4] == pytest.approx(-1.0)  # -3.1415927 / 3.1415927
        assert result[5] == pytest.approx(-1.0)  # -10.0 / 10.0
        assert result[6] == -1.0  # Leg contact: 0.0 → -1.0
        assert result[7] == -1.0  # Leg contact: 0.0 → -1.0
    
    def test_normalise_observation_zero_values(self):
        """Test normalization with zero values to [-1, 1] range."""
        observation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        result = normalise_observation(observation)
        
        assert result[0] == pytest.approx(0.0)  # 0.0 / 2.5
        assert result[1] == pytest.approx(0.0)  # 0.0 / 2.5
        assert result[2] == pytest.approx(0.0)  # 0.0 / 10.0
        assert result[3] == pytest.approx(0.0)  # 0.0 / 10.0
        assert result[4] == pytest.approx(0.0)  # 0.0 / 3.1415927
        assert result[5] == pytest.approx(0.0)  # 0.0 / 10.0
        assert result[6] == -1.0  # Leg contact: 0.0 → -1.0
        assert result[7] == 1.0  # Leg contact: 1.0 → 1.0
    
    def test_normalise_observation_modifies_in_place(self):
        """Test that normalise_observation modifies the list in place."""
        observation = [1.5, 1.5, 5.0, 5.0, 3.1415927, 5.0, 1.0, 1.0]
        original_id = id(observation)
        
        result = normalise_observation(observation)
        
        assert id(result) == original_id
        assert result is observation
    
    def test_normalise_observation_fractional_values(self):
        """Test normalization with fractional values to [-1, 1] range."""
        observation = [1.25, 1.25, 5.0, 5.0, 1.57079635, 5.0, 0.0, 1.0]
        
        result = normalise_observation(observation)
        
        assert result[0] == pytest.approx(0.5)  # 1.25 / 2.5
        assert result[1] == pytest.approx(0.5)  # 1.25 / 2.5
        assert result[2] == pytest.approx(0.5)  # 5.0 / 10.0
        assert result[3] == pytest.approx(0.5)  # 5.0 / 10.0
        assert result[4] == pytest.approx(0.5)  # 1.57079635 / 3.1415927
        assert result[5] == pytest.approx(0.5)  # 5.0 / 10.0
        assert result[6] == -1.0  # Leg contact: 0.0 → -1.0
        assert result[7] == 1.0  # Leg contact: 1.0 → 1.0
    
    def test_normalise_observation_clamping(self):
        """Test that values exceeding typical ranges are clamped to [-1, 1]."""
        observation = [5.0, -5.0, 20.0, -20.0, 6.2831854, -20.0, 1.0, 0.0]
        
        result = normalise_observation(observation)
        
        # All values should be clamped to [-1, 1] range
        assert result[0] == pytest.approx(1.0)  # 5.0 / 2.5 = 2.0, clamped to 1.0
        assert result[1] == pytest.approx(-1.0)  # -5.0 / 2.5 = -2.0, clamped to -1.0
        assert result[2] == pytest.approx(1.0)  # 20.0 / 10.0 = 2.0, clamped to 1.0
        assert result[3] == pytest.approx(-1.0)  # -20.0 / 10.0 = -2.0, clamped to -1.0
        assert result[4] == pytest.approx(1.0)  # 6.2831854 / 3.1415927 = 2.0, clamped to 1.0
        assert result[5] == pytest.approx(-1.0)  # -20.0 / 10.0 = -2.0, clamped to -1.0
        assert result[6] == 1.0  # Leg contact: 1.0 → 1.0
        assert result[7] == -1.0  # Leg contact: 0.0 → -1.0

