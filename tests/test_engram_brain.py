import pytest
import numpy as np
from unittest.mock import MagicMock, Mock, patch
from engram import Engram, EngramStore
from EngramBrain import EngramBrain
from IResonatorFactory import IResonatorFactory


class TestEngramBrain:
    """Tests for EngramBrain class."""
    
    @pytest.fixture
    def mock_store(self):
        """Mock EngramStore."""
        store = MagicMock(spec=EngramStore)
        store.nearest = MagicMock(return_value=[])
        store.insert = MagicMock()
        store.get_count = MagicMock(return_value=0)
        return store
    
    @pytest.fixture
    def mock_factory(self):
        """Mock resonator factory."""
        factory = MagicMock(spec=IResonatorFactory)
        factory.make_resonator = MagicMock(side_effect=lambda input, success: input)  # Return only input, not append success
        return factory
    
    @pytest.fixture
    def brain(self, mock_store, mock_factory):
        """Create EngramBrain instance for testing."""
        return EngramBrain(input_size=8, output_size=4, engram_store=mock_store, resonator_factory=mock_factory)
    
    def test_init(self, brain, mock_store, mock_factory):
        """Test EngramBrain initialization."""
        assert brain.input_size == 8
        assert brain.output_size == 4
        assert brain.engram_store == mock_store
        assert brain.resonator_factory == mock_factory
    
    def test_input_to_resonator(self, brain, mock_factory):
        """Test that input_to_resonator delegates to factory."""
        input_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        success = 0.5
        
        result = brain.input_to_resonator(input_vector, success)
        
        mock_factory.make_resonator.assert_called_once_with(input_vector, success)
        # Factory now returns only input (success not included in distance calculations)
        np.testing.assert_array_equal(result, input_vector)
    
    def test_get_resonating_engrams(self, brain, mock_store):
        """Test that get_resonating_engrams queries store correctly."""
        resonator = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # 8D state vector
        min_results = 300
        
        result = brain.get_resonating_engrams(resonator, min_results)
        
        # nearest is called with the numpy array directly (it may convert internally)
        mock_store.nearest.assert_called_once()
        call_args = mock_store.nearest.call_args[0]
        # Check that first arg is the resonator (as array or list)
        assert len(call_args) >= 2
        assert call_args[1] == min_results
        assert result == []
    
    def test_score_engrams_without_trial_metadata(self, brain):
        """Test scoring engrams without trial metadata."""
        engram1 = Engram(vector=[0.1] * 8, action=0, outcome=0.8)
        engram2 = Engram(vector=[0.2] * 8, action=1, outcome=0.5)
        engram3 = Engram(vector=[0.3] * 8, action=2, outcome=-0.3)
        
        resonating_engrams = [
            (engram1, 0.1),
            (engram2, 0.2),
            (engram3, 0.3),
        ]
        
        scored = brain.score_engrams(resonating_engrams)
        
        assert len(scored) == 3
        # Should be sorted by score descending
        # Format is now (engram, score, distance)
        assert scored[0][1] == 0.8  # engram1 score
        assert scored[0][2] == 0.1  # engram1 distance
        assert scored[1][1] == 0.5  # engram2 score
        assert scored[1][2] == 0.2  # engram2 distance
        assert scored[2][1] == -0.3  # engram3 score
        assert scored[2][2] == 0.3  # engram3 distance
    
    def test_score_engrams_with_trial_metadata(self, brain):
        """Test scoring engrams with trial success multiplier."""
        with patch('EngramBrain.TRIAL_SUCCESS_MULTIPLIER_SCALE', 0.5):
            engram1 = Engram(
                vector=[0.1] * 8, 
                action=0, 
                outcome=0.8,
                trial_final_success=1.0  # Best trial
            )
            engram2 = Engram(
                vector=[0.2] * 8, 
                action=1, 
                outcome=0.5,
                trial_final_success=-1.0  # Worst trial
            )
            engram3 = Engram(
                vector=[0.3] * 8, 
                action=2, 
                outcome=0.4,
                trial_final_success=0.0  # No trial metadata
            )
            
            resonating_engrams = [
                (engram1, 0.1),
                (engram2, 0.2),
                (engram3, 0.3),
            ]
            
            scored = brain.score_engrams(resonating_engrams)
            
            assert len(scored) == 3
            # Format is now (engram, score, distance)
            # engram1: 0.8 * (1.0 + 1.0 * 0.5) = 0.8 * 1.5 = 1.2
            assert scored[0][1] == pytest.approx(1.2)  # score
            assert scored[0][2] == 0.1  # distance
            # engram2: 0.5 * (1.0 + (-1.0) * 0.5) = 0.5 * 0.5 = 0.25
            # engram3: 0.4 (no multiplier) - comes before engram2 since 0.4 > 0.25
            assert scored[1][1] == pytest.approx(0.4)  # engram3 score
            assert scored[1][2] == 0.3  # engram3 distance
            assert scored[2][1] == pytest.approx(0.25)  # engram2 score
            assert scored[2][2] == 0.2  # engram2 distance
    
    def test_make_output_empty_results(self, brain):
        """Test output generation with empty results (random fallback)."""
        scored_engrams = []
        
        with patch('numpy.random.random') as mock_random:
            mock_random.return_value = np.array([0.1, 0.2, 0.3, 0.4])
            output = brain.make_output([0.0] * 8, scored_engrams)
        
        assert len(output) == brain.output_size
        # make_output returns a list (tolist()), so compare as list
        # Convert to list if it's a numpy array
        if isinstance(output, np.ndarray):
            output = output.tolist()
        assert output == [0.1, 0.2, 0.3, 0.4]
    
    def test_make_output_with_scored_engrams(self, brain):
        """Test output generation with scored engrams."""
        engram1 = Engram(vector=[0.1] * 8, action=0, outcome=0.8)
        engram2 = Engram(vector=[0.2] * 8, action=0, outcome=0.6)
        engram3 = Engram(vector=[0.3] * 8, action=1, outcome=0.4)
        engram4 = Engram(vector=[0.4] * 8, action=2, outcome=-0.2)
        
        # Format is now (engram, score, distance)
        # Using equal distances for simple average test
        scored_engrams = [
            (engram1, 0.8, 0.1),
            (engram2, 0.6, 0.1),
            (engram3, 0.4, 0.1),
            (engram4, -0.2, 0.1),
        ]
        
        with patch('numpy.random.normal', return_value=np.array([0.0, 0.0, 0.0, 0.0])):
            with patch('settings.NOISE_START', 0.0), patch('settings.NOISE_END', 0.0):
                output = brain.make_output([0.0] * 8, scored_engrams, episode_progress=0.0)
        
        assert len(output) == brain.output_size
        # Action 0: weighted average with equal distances = (0.8 + 0.6) / 2 = 0.7
        assert output[0] == pytest.approx(0.7)
        # Action 1: 0.4 / 1 = 0.4
        assert output[1] == pytest.approx(0.4)
        # Action 2: -0.2 / 1 = -0.2
        assert output[2] == pytest.approx(-0.2)
        # Action 3: 0 (no engrams)
        assert output[3] == pytest.approx(0.0)
    
    def test_make_output_panic_factor_noise_scaling(self, brain):
        """Test panic factor noise scaling."""
        engram = Engram(vector=[0.1] * 8, action=0, outcome=0.5)
        scored_engrams = [(engram, 0.5, 0.1)]  # Format: (engram, score, distance)
        
        with patch('EngramBrain.NOISE_START', 0.1), patch('EngramBrain.NOISE_END', 0.1):
            with patch('EngramBrain.PANIC_MAX_NOISE', 1.0):
                # No panic (panic_factor = 0.0), episode_progress = 0.0 means noise = NOISE_START
                with patch('numpy.random.normal') as mock_normal:
                    mock_normal.return_value = np.array([0.0, 0.0, 0.0, 0.0])
                    output_no_panic = brain.make_output([0.0] * 8, scored_engrams, panic_factor=0.0, episode_progress=0.0)
                    # Should use episode_noise = NOISE_START = 0.1 (since episode_progress=0.0)
                    assert mock_normal.call_args[0][1] == 0.1
                
                # Full panic (panic_factor = 1.0)
                with patch('numpy.random.normal') as mock_normal:
                    mock_normal.return_value = np.array([0.0, 0.0, 0.0, 0.0])
                    output_panic = brain.make_output([0.0] * 8, scored_engrams, panic_factor=1.0, episode_progress=0.0)
                    # With panic_factor=1.0, noise = episode_noise * (1.0 + 1.0 * (1.0/0.1 - 1.0)) = 0.1 * 10.0 = 1.0
                    assert mock_normal.call_args[0][1] == pytest.approx(1.0)
    
    def test_make_output_distance_weighting(self, brain):
        """Test that make_output weights scores by distance - closer engrams have more influence."""
        # Create two engrams for the same action with same score but different distances
        engram_close = Engram(vector=[0.1] * 8, action=0, outcome=0.5)
        engram_far = Engram(vector=[0.2] * 8, action=0, outcome=0.5)
        
        # Close engram: distance = 0.1, weight = 1/(1+0.1) = 0.909
        # Far engram: distance = 1.0, weight = 1/(1+1.0) = 0.5
        scored_engrams = [
            (engram_close, 0.5, 0.1),  # Close engram
            (engram_far, 0.5, 1.0),   # Far engram
        ]
        
        with patch('numpy.random.normal', return_value=np.array([0.0, 0.0, 0.0, 0.0])):
            with patch('settings.NOISE_START', 0.0), patch('settings.NOISE_END', 0.0):
                output = brain.make_output([0.0] * 8, scored_engrams, episode_progress=0.0)
        
        # The weighted average should favor the closer engram
        # weight_close = 1/(1+0.1) = 0.909, weight_far = 1/(1+1.0) = 0.5
        # weighted_score = (0.5 * 0.909 + 0.5 * 0.5) / (0.909 + 0.5) = 0.7045 / 1.409 = 0.5
        # Actually, since both scores are the same (0.5), the result should still be 0.5
        # But let's test with different scores to see the distance weighting effect
        
        # Test with different scores to verify distance weighting
        engram_close_high = Engram(vector=[0.1] * 8, action=0, outcome=0.8)
        engram_far_low = Engram(vector=[0.2] * 8, action=0, outcome=0.2)
        
        scored_engrams_diff = [
            (engram_close_high, 0.8, 0.1),  # Close engram with high score
            (engram_far_low, 0.2, 1.0),     # Far engram with low score
        ]
        
        with patch('numpy.random.normal', return_value=np.array([0.0, 0.0, 0.0, 0.0])):
            with patch('settings.NOISE_START', 0.0), patch('settings.NOISE_END', 0.0):
                output_diff = brain.make_output([0.0] * 8, scored_engrams_diff, episode_progress=0.0)
        
        # Weighted average should be closer to the close engram's score
        # weight_close = 1/(1+0.1) = 0.909, weight_far = 1/(1+1.0) = 0.5
        # weighted_score = (0.8 * 0.909 + 0.2 * 0.5) / (0.909 + 0.5) = 0.8272 / 1.409 = 0.587
        expected_weighted = (0.8 * (1.0 / 1.1) + 0.2 * (1.0 / 2.0)) / ((1.0 / 1.1) + (1.0 / 2.0))
        assert output_diff[0] == pytest.approx(expected_weighted, abs=0.01)
        
        # Verify that the close engram has more influence than the far one
        # The result should be closer to 0.8 (close engram) than to 0.2 (far engram)
        assert output_diff[0] > 0.5, "Close engram should have more influence"
        assert output_diff[0] < 0.8, "Result should be weighted average, not just close engram"
    
    def test_apply_feedback(self, brain, mock_store, mock_factory):
        """Test that apply_feedback creates and inserts engram correctly."""
        input_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        action = 2
        outcome = 0.5
        success = 0.8
        trial_final_success = 0.9
        
        brain.apply_feedback(
            input_vector, action, outcome, success,
            trial_final_success
        )
        
        # Verify engram was inserted with correct metadata
        # Note: apply_feedback now stores only the input state, not the resonator
        mock_store.insert.assert_called_once()
        call_args = mock_store.insert.call_args
        inserted_engram = call_args[0][0]
        assert inserted_engram.action == action
        assert inserted_engram.outcome == outcome
        # Vector should be the input state (8D), not resonator with success
        assert len(inserted_engram.vector) == 8
        np.testing.assert_array_equal(inserted_engram.vector, input_vector)
        assert call_args[0][1] == trial_final_success
    
    def test_decide_without_distance_info(self, brain, mock_store, mock_factory):
        """Test decide method without distance info."""
        input_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        success = 0.5
        
        engram1 = Engram(vector=[0.1] * 8, action=0, outcome=0.8)
        engram2 = Engram(vector=[0.2] * 8, action=1, outcome=0.5)
        mock_store.nearest.return_value = [
            (engram1, 0.1),
            (engram2, 0.2),
        ]
        
        with patch('settings.MIN_RESULTS', 300):
            with patch('numpy.random.normal', return_value=np.array([0.0, 0.0, 0.0, 0.0])):
                with patch('settings.NOISE_START', 0.0), patch('settings.NOISE_END', 0.0):
                    output = brain.decide(input_vector, success, return_distance_info=False, episode_progress=0.0)
        
        assert isinstance(output, list)
        assert len(output) == brain.output_size
    
    def test_decide_with_distance_info(self, brain, mock_store, mock_factory):
        """Test decide method with distance info."""
        input_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        success = 0.5
        
        engram1 = Engram(vector=[0.1] * 8, action=0, outcome=0.8)
        engram2 = Engram(vector=[0.2] * 8, action=1, outcome=0.5)
        mock_store.nearest.return_value = [
            (engram1, 0.1),
            (engram2, 0.2),
        ]
        
        with patch('settings.MIN_RESULTS', 300):
            with patch('numpy.random.normal', return_value=np.array([0.0, 0.0, 0.0, 0.0])):
                with patch('settings.NOISE_START', 0.0), patch('settings.NOISE_END', 0.0):
                    output, avg_distance = brain.decide(input_vector, success, return_distance_info=True, episode_progress=0.0)
        
        assert isinstance(output, list)
        assert len(output) == brain.output_size
        assert avg_distance == pytest.approx(0.15)  # (0.1 + 0.2) / 2
    
    def test_decide_with_empty_results_distance_info(self, brain, mock_store):
        """Test decide with empty results returns infinite distance."""
        input_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        success = 0.5
        
        mock_store.nearest.return_value = []
        
        with patch('settings.MIN_RESULTS', 300):
            with patch('numpy.random.random', return_value=np.array([0.1, 0.2, 0.3, 0.4])):
                output, avg_distance = brain.decide(input_vector, success, return_distance_info=True)
        
        assert avg_distance == float('inf')

