import pytest
import numpy as np
from WeightedResonatorFactory import WeightedResonatorFactory
from settings import STATE_VECTOR_SIZE


class TestWeightedResonatorFactory:
    """Tests for WeightedResonatorFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Create WeightedResonatorFactory instance with default weights."""
        return WeightedResonatorFactory()
    
    def test_make_resonator_returns_only_input(self, factory):
        """Test that make_resonator returns only the input state (success not included in distance calculations)."""
        input_vector = np.array([0.1 * (i + 1) % 1.0 for i in range(STATE_VECTOR_SIZE)])
        success = 0.5
        
        result = factory.make_resonator(input_vector, success)
        
        # Should return weighted input, not append success
        assert len(result) == len(input_vector)
    
    def test_make_resonator_correct_dimensions(self):
        """Test that make_resonator produces correct output dimensions with matching weights."""
        input_sizes = [1, 5, 8, 21, 100]
        
        for input_size in input_sizes:
            # Create factory with weights matching input size
            weights = [1.0] * input_size
            factory = WeightedResonatorFactory(weights=weights)
            input_vector = np.random.random(input_size)
            success = 0.5
            
            result = factory.make_resonator(input_vector, success)
            
            # Should return same dimensions as input (success not appended)
            assert len(result) == input_size
            # With all weights = 1.0, result should equal input
            np.testing.assert_array_equal(result, input_vector)
    
    def test_make_resonator_zero_success(self):
        """Test make_resonator with zero success value (success is ignored)."""
        input_vector = np.array([0.1, 0.2, 0.3])
        weights = [1.0, 1.0, 1.0]
        factory = WeightedResonatorFactory(weights=weights)
        success = 0.0
        
        result = factory.make_resonator(input_vector, success)
        
        # Should return only input, regardless of success value
        assert len(result) == 3
        np.testing.assert_array_equal(result, input_vector)
    
    def test_make_resonator_negative_success(self):
        """Test make_resonator with negative success value (success is ignored)."""
        input_vector = np.array([0.1, 0.2, 0.3])
        weights = [1.0, 1.0, 1.0]
        factory = WeightedResonatorFactory(weights=weights)
        success = -0.5
        
        result = factory.make_resonator(input_vector, success)
        
        # Should return only input, regardless of success value
        assert len(result) == 3
        np.testing.assert_array_equal(result, input_vector)
    
    def test_make_resonator_positive_success(self):
        """Test make_resonator with positive success value (success is ignored)."""
        input_vector = np.array([0.1, 0.2, 0.3])
        weights = [1.0, 1.0, 1.0]
        factory = WeightedResonatorFactory(weights=weights)
        success = 0.8
        
        result = factory.make_resonator(input_vector, success)
        
        # Should return only input, regardless of success value
        assert len(result) == 3
        np.testing.assert_array_equal(result, input_vector)
    
    def test_make_resonator_extreme_values(self):
        """Test make_resonator with extreme success values (success is ignored)."""
        input_vector = np.array([0.1, 0.2, 0.3])
        weights = [1.0, 1.0, 1.0]
        factory = WeightedResonatorFactory(weights=weights)
        
        # Test with -1.0 - should return only input
        result = factory.make_resonator(input_vector, -1.0)
        np.testing.assert_array_equal(result, input_vector)
        
        # Test with 1.0 - should return only input
        result = factory.make_resonator(input_vector, 1.0)
        np.testing.assert_array_equal(result, input_vector)
    
    def test_make_resonator_preserves_input(self):
        """Test that make_resonator preserves input vector values when weights are all 1.0."""
        input_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        factory = WeightedResonatorFactory(weights=weights)
        success = 0.7
        
        result = factory.make_resonator(input_vector, success)
        
        # Should return input exactly as-is when weights are all 1.0 (backward compatibility)
        np.testing.assert_array_equal(result, input_vector)
    
    def test_make_resonator_applies_weights(self):
        """Test that make_resonator applies component weights correctly."""
        input_vector = np.array([1.0, 2.0, 3.0, 4.0])
        weights = [2.0, 0.5, 1.0, 3.0]
        factory = WeightedResonatorFactory(weights=weights)
        success = 0.5
        
        result = factory.make_resonator(input_vector, success)
        
        # Should multiply each component by its weight
        expected = np.array([2.0, 1.0, 3.0, 12.0])  # [1.0*2.0, 2.0*0.5, 3.0*1.0, 4.0*3.0]
        np.testing.assert_array_equal(result, expected)
    
    def test_make_resonator_custom_weights(self):
        """Test make_resonator with custom weights for different components."""
        input_vector = np.array([0.5, 0.5, 0.5, 0.5])
        weights = [0.0, 1.0, 2.0, 0.5]
        factory = WeightedResonatorFactory(weights=weights)
        success = 0.3
        
        result = factory.make_resonator(input_vector, success)
        
        # Component 0 should be zeroed, component 2 should be doubled, etc.
        expected = np.array([0.0, 0.5, 1.0, 0.25])
        np.testing.assert_array_equal(result, expected)
    
    def test_make_resonator_backward_compatibility_all_ones(self):
        """Test that default weights (all 1.0) maintain backward compatibility."""
        input_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        factory = WeightedResonatorFactory(weights=weights)
        success = 0.5
        
        result = factory.make_resonator(input_vector, success)
        
        # With all weights = 1.0, should return input unchanged
        np.testing.assert_array_equal(result, input_vector)

