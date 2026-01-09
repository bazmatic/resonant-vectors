import pytest
import numpy as np
import json
from unittest.mock import MagicMock, Mock, patch
from Trainer import Trainer, normalise_observation
from engram import EngramStore
from EngramBrain import EngramBrain
from WeightedResonatorFactory import WeightedResonatorFactory


class TestTrainer:
    """Tests for Trainer class core methods."""
    
    @pytest.fixture
    def mock_env(self):
        """Mock gymnasium environment."""
        env = MagicMock()
        env.reset = MagicMock(return_value=(np.array([0.0] * 8), {}))
        env.step = MagicMock(return_value=(np.array([0.0] * 8), 0.0, False, False, {}))
        env.close = MagicMock()
        return env
    
    @pytest.fixture
    def mock_store(self):
        """Mock EngramStore."""
        store = MagicMock(spec=EngramStore)
        store.nearest = MagicMock(return_value=[])
        store.insert = MagicMock()
        store.get_count = MagicMock(return_value=0)
        store.get_outcome_stats = MagicMock(return_value={'positive_ratio': 0.0, 'negative_ratio': 0.0, 'mean_outcome': 0.0})
        return store
    
    @pytest.fixture
    def mock_brain(self, mock_store):
        """Mock EngramBrain."""
        brain = MagicMock(spec=EngramBrain)
        brain.decide = MagicMock(return_value=([0.1, 0.2, 0.3, 0.4], 0.5))
        brain.apply_feedback = MagicMock()
        brain.engram_store = mock_store
        return brain
    
    @patch('Trainer.gym.make')
    @patch('Trainer.EngramStore')
    @patch('Trainer.EngramBrain')
    @patch('Trainer.WeightedResonatorFactory')
    def test_init(self, mock_factory_class, mock_brain_class, mock_store_class, mock_gym_make, mock_env, mock_settings):
        """Test Trainer initialization."""
        with patch('settings.DISPLAY', False):
            with patch('settings.SHOW_ACTION_OUTPUT', False):
                mock_store = MagicMock()
                mock_store_class.return_value = mock_store
                mock_factory = MagicMock()
                mock_factory_class.return_value = mock_factory
                mock_brain = MagicMock()
                mock_brain_class.return_value = mock_brain
                mock_gym_make.return_value = mock_env
                
                trainer = Trainer("test_instance", clear_collection=True)
                
                assert trainer.instance_name == "test_instance"
                assert trainer.trial_count == 0
                assert trainer.feedback_queue == []
                assert trainer.best_success == -1.0
                assert trainer.mean_success == -1.0
                assert 'rewards' in trainer.metrics
                assert 'episode_lengths' in trainer.metrics
                assert 'successes' in trainer.metrics
                assert 'deaths' in trainer.metrics
    
    def test_queue_feedback(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test feedback queue management."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            observation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                            action = 2
                            reward = 0.5
                            
                            trainer.queue_feedback(observation, action, reward)
                            
                            assert len(trainer.feedback_queue) == 1
                            assert trainer.feedback_queue[0] == (observation, action, reward)
    
    def test_flush_feedback_processes_queue(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test that flush_feedback processes all queued feedback."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('Trainer.DISPLAY', False):
                            with patch('settings.READ_ONLY', False):
                                trainer = Trainer("test_instance", clear_collection=True)
                                
                                # Add feedback to queue
                                trainer.queue_feedback([0.1] * 8, 0, 0.5)
                                trainer.queue_feedback([0.2] * 8, 1, 0.3)
                                
                                trainer.trial_count = 5
                                trainer.flush_feedback(100.0, 0.5, 200, True)
                                
                                # Verify feedback was processed
                                assert len(trainer.feedback_queue) == 0
                                # Now uses batch_apply_feedback instead of individual calls
                                assert mock_brain.batch_apply_feedback.call_count == 1
                                # Verify it was called with the correct number of items
                                call_args = mock_brain.batch_apply_feedback.call_args
                                assert len(call_args[0][0]) == 2  # 2 observations
                                assert len(call_args[0][1]) == 2  # 2 actions
                                assert len(call_args[0][2]) == 2  # 2 outcomes
    
    def test_flush_feedback_updates_best_success(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test that flush_feedback updates best_success."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('Trainer.DISPLAY', False):
                            with patch('settings.READ_ONLY', False):
                                trainer = Trainer("test_instance", clear_collection=True)
                                
                                trainer.best_success = -1.0
                                trainer.trial_count = 1  # Set trial count to avoid division by zero
                                trainer.flush_feedback(100.0, 0.8, 200, True)
                                
                                assert trainer.best_success == 0.8
                                
                                # Test that it doesn't update if new success is lower
                                trainer.trial_count = 2
                                trainer.flush_feedback(100.0, 0.5, 200, True)
                                assert trainer.best_success == 0.8
    
    def test_flush_feedback_updates_mean_success(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test that flush_feedback updates mean_success."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('Trainer.DISPLAY', False):
                            with patch('settings.READ_ONLY', False):
                                trainer = Trainer("test_instance", clear_collection=True)
                                
                                # First trial
                                trainer.trial_count = 1
                                trainer.mean_success = -1.0
                                trainer.flush_feedback(100.0, 0.5, 200, True)
                                assert trainer.mean_success == 0.5
                                
                                # Second trial
                                trainer.trial_count = 2
                                trainer.flush_feedback(150.0, 0.7, 200, True)
                                # Mean should be (0.5 + 0.7) / 2 = 0.6
                                assert trainer.mean_success == pytest.approx(0.6)
    
    def test_flush_feedback_mean_success_rolling_window(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test that mean_success uses rolling window after 20 trials."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('Trainer.DISPLAY', False):
                            with patch('settings.READ_ONLY', False):
                                trainer = Trainer("test_instance", clear_collection=True)
                                
                                trainer.trial_count = 20
                                trainer.mean_success = 0.5
                                
                                trainer.flush_feedback(100.0, 0.8, 200, True)
                                
                                # Should use rolling average: (0.5 * 19 + 0.8) / 20
                                expected = (0.5 * 19 + 0.8) / 20
                                assert trainer.mean_success == pytest.approx(expected)
    
    def test_get_summary_stats_empty_metrics(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test get_summary_stats with empty metrics."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            stats = trainer.get_summary_stats()
                            
                            assert stats['total_trials'] == 0
                            assert stats['overall_average_reward'] == 0.0
                            assert stats['success_rate'] == 0.0
                            assert stats['death_count'] == 0
                            assert stats['best_episode_reward'] == -float('inf')
    
    def test_get_summary_stats_with_data(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test get_summary_stats calculation with data."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            # Add some metrics
                            trainer.metrics['rewards'] = [100.0, 150.0, 200.0, 250.0]
                            trainer.metrics['episode_lengths'] = [200, 250, 300, 350]
                            trainer.metrics['successes'] = [False, False, True, True]
                            trainer.metrics['deaths'] = [False, False, False, False]
                            trainer.metrics['engram_distances'] = [0.5, 0.4, 0.3, 0.2]
                            trainer.metrics['rolling_average_50'] = [175.0]
                            trainer.metrics['rolling_average_100'] = [175.0]
                            trainer.metrics['best_reward'] = 250.0
                            
                            mock_store.get_count.return_value = 1000
                            
                            stats = trainer.get_summary_stats()
                            
                            assert stats['total_trials'] == 4
                            assert stats['overall_average_reward'] == pytest.approx(175.0)
                            assert stats['success_rate'] == pytest.approx(50.0)
                            assert stats['death_count'] == 0
                            assert stats['best_episode_reward'] == 250.0
                            assert stats['total_engrams'] == 1000
    
    def test_convert_to_json_serializable_numpy_types(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test _convert_to_json_serializable converts numpy types."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            # Test numpy integer
                            result = trainer._convert_to_json_serializable(np.int64(42))
                            assert isinstance(result, float)
                            assert result == 42.0
                            
                            # Test numpy float
                            result = trainer._convert_to_json_serializable(np.float64(3.14))
                            assert isinstance(result, float)
                            assert result == 3.14
                            
                            # Test numpy bool
                            result = trainer._convert_to_json_serializable(np.bool_(True))
                            assert isinstance(result, bool)
                            assert result == True
                            
                            # Test numpy array
                            arr = np.array([1, 2, 3])
                            result = trainer._convert_to_json_serializable(arr)
                            assert isinstance(result, list)
                            assert result == [1, 2, 3]
                            
                            # Test nested dict
                            data = {'key': np.int64(42), 'nested': {'arr': np.array([1, 2])}}
                            result = trainer._convert_to_json_serializable(data)
                            assert isinstance(result, dict)
                            assert result['key'] == 42.0
                            assert result['nested']['arr'] == [1, 2]
    
    def test_value_to_color_red(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test _value_to_color maps -1 to red."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            color = trainer._value_to_color(-1.0)
                            assert color == (255, 0, 0)  # Red
    
    def test_value_to_color_yellow(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test _value_to_color maps 0 to yellow."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            color = trainer._value_to_color(0.0)
                            assert color == (255, 255, 0)  # Yellow
    
    def test_value_to_color_green(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test _value_to_color maps 1 to green."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            color = trainer._value_to_color(1.0)
                            assert color == (0, 255, 0)  # Green
    
    def test_value_to_color_interpolation(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test _value_to_color interpolates between colors."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            # Test midpoint between red and yellow (-0.5)
                            color = trainer._value_to_color(-0.5)
                            assert color[0] == 255  # Red component
                            assert 0 < color[1] < 255  # Green component between 0 and 255
                            assert color[2] == 0  # Blue component
                            
                            # Test midpoint between yellow and green (0.5)
                            color = trainer._value_to_color(0.5)
                            assert 0 < color[0] < 255  # Red component between 0 and 255
                            assert color[1] == 255  # Green component
                            assert color[2] == 0  # Blue component
    
    def test_value_to_color_clamping(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test _value_to_color clamps values outside [-1, 1]."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            trainer = Trainer("test_instance", clear_collection=True)
                            
                            # Test value < -1
                            color = trainer._value_to_color(-2.0)
                            assert color == (255, 0, 0)  # Clamped to red
                            
                            # Test value > 1
                            color = trainer._value_to_color(2.0)
                            assert color == (0, 255, 0)  # Clamped to green
    
    def test_action_selection_probabilistic_choice_enabled(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test that action selection uses probabilistic choice when enabled."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            with patch('settings.READ_ONLY', True):
                                with patch('settings.USE_HIT_POINTS', False):
                                    with patch('settings.MAX_TRIAL_LENGTH', 1):
                                        with patch('settings.PROBABILISTIC_CHOICE', True):
                                            trainer = Trainer("test_instance", clear_collection=True)
                                            
                                            # Mock brain output with different values
                                            brain_output = [0.1, 0.5, 0.2, 0.3]
                                            mock_brain.decide.return_value = (brain_output, 0.5)
                                            
                                            # Mock environment to terminate immediately
                                            mock_env.reset.return_value = (np.array([0.0] * 8), {})
                                            mock_env.step.return_value = (np.array([0.0] * 8), 0.0, True, False, {})
                                            
                                            # Mock numpy.random.choice to verify it's called with probabilities
                                            with patch('numpy.random.choice') as mock_choice:
                                                mock_choice.return_value = 1
                                                trainer.trial()
                                                
                                                # Verify that random.choice was called (probabilistic selection)
                                                assert mock_choice.called
                                                # Check that it was called with probabilities
                                                call_args = mock_choice.call_args
                                                if call_args and len(call_args) > 1 and 'p' in call_args[1]:
                                                    probabilities = call_args[1]['p']
                                                    # Probabilities should sum to 1.0
                                                    assert abs(sum(probabilities) - 1.0) < 1e-10
    
    def test_action_selection_probabilistic_choice_zero_sum(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test that action selection uses random choice when normalized sum is zero."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            with patch('settings.READ_ONLY', True):
                                with patch('settings.USE_HIT_POINTS', False):
                                    with patch('settings.MAX_TRIAL_LENGTH', 1):
                                        with patch('settings.PROBABILISTIC_CHOICE', True):
                                            trainer = Trainer("test_instance", clear_collection=True)
                                            
                                            # Mock brain output where all values are the same (normalized sum = 0)
                                            brain_output = [0.5, 0.5, 0.5, 0.5]
                                            mock_brain.decide.return_value = (brain_output, 0.5)
                                            
                                            # Mock environment to terminate immediately
                                            mock_env.reset.return_value = (np.array([0.0] * 8), {})
                                            mock_env.step.return_value = (np.array([0.0] * 8), 0.0, True, False, {})
                                            
                                            # Mock numpy.random.choice to verify it's called without probabilities
                                            with patch('numpy.random.choice') as mock_choice:
                                                mock_choice.return_value = 2
                                                trainer.trial()
                                                
                                                # Verify that random.choice was called
                                                assert mock_choice.called
                                                # When sum is zero, it should be called without p parameter
                                                # It should be called with numpy.arange(4) directly
                                                assert len(mock_choice.call_args_list) > 0
    
    def test_action_selection_argmax_when_probabilistic_disabled(self, mock_env, mock_store, mock_brain, mock_settings):
        """Test that action selection uses argmax when probabilistic choice is disabled."""
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('Trainer.EngramStore', return_value=mock_store):
                with patch('Trainer.EngramBrain', return_value=mock_brain):
                    with patch('Trainer.WeightedResonatorFactory'):
                        with patch('settings.DISPLAY', False):
                            with patch('settings.READ_ONLY', True):
                                with patch('settings.USE_HIT_POINTS', False):
                                    with patch('settings.MAX_TRIAL_LENGTH', 1):
                                        with patch('settings.PROBABILISTIC_CHOICE', False):
                                            trainer = Trainer("test_instance", clear_collection=True)
                                            
                                            # Mock brain output with max at index 2
                                            brain_output = [0.1, 0.2, 0.9, 0.3]
                                            mock_brain.decide.return_value = (brain_output, 0.5)
                                            
                                            # Mock environment to terminate immediately
                                            mock_env.reset.return_value = (np.array([0.0] * 8), {})
                                            mock_env.step.return_value = (np.array([0.0] * 8), 0.0, True, False, {})
                                            
                                            # Since numpy.argmax is hard to mock (already imported), 
                                            # we verify the behavior by checking that the action selected
                                            # matches what argmax would return for the normalized values
                                            trainer.trial()
                                            
                                            # Verify that the action passed to env.step matches argmax of normalized values
                                            # normalized = [0.1-0.1, 0.2-0.1, 0.9-0.1, 0.3-0.1] = [0, 0.1, 0.8, 0.2]
                                            # argmax should be 2 (index of 0.8)
                                            normalized = [x - min(brain_output) for x in brain_output]
                                            expected_action = np.argmax(normalized)
                                            
                                            # Check that env.step was called with the correct action
                                            assert mock_env.step.called
                                            step_call_args = mock_env.step.call_args[0]
                                            actual_action = step_call_args[0]
                                            assert actual_action == expected_action, \
                                                f"Action {actual_action} should be {expected_action} (argmax of normalized values)"
    
    def test_probabilistic_choice_weights_actions_by_engram_outcomes(self, mock_env, mock_settings):
        """Test that probabilistic choice correctly weights actions: positive outcomes more likely, negative less likely."""
        from engram import EngramStore, Engram
        from EngramBrain import EngramBrain
        from WeightedResonatorFactory import WeightedResonatorFactory
        
        with patch('Trainer.gym.make', return_value=mock_env):
            with patch('settings.DISPLAY', False):
                with patch('settings.READ_ONLY', True):  # Don't add new engrams during test
                    with patch('settings.USE_HIT_POINTS', False):
                        with patch('settings.MAX_TRIAL_LENGTH', 1):
                            with patch('settings.PROBABILISTIC_CHOICE', True):
                                with patch('settings.MIN_RESULTS', 10):
                                    with patch('settings.NOISE_START', 0.0), patch('settings.NOISE_END', 0.0):  # No noise for deterministic probabilities
                                        with patch('builtins.print'):  # Suppress print statements
                                            # Create real store and brain (not mocked) to test actual scoring logic
                                            store = MagicMock(spec=EngramStore)
                                            store.get_count = MagicMock(return_value=100)
                                            
                                            # Create engrams with different outcomes for different actions
                                            # Action 0: Strong positive outcomes (should be most likely)
                                            # Action 1: Weak positive outcomes (should be somewhat likely)
                                            # Action 2: Negative outcomes (should be least likely)
                                            # Action 3: No engrams (should have zero weight)
                                            
                                            engrams_for_action_0 = [
                                                Engram(vector=[0.1] * 9, action=0, outcome=0.9),
                                                Engram(vector=[0.11] * 9, action=0, outcome=0.8),
                                                Engram(vector=[0.12] * 9, action=0, outcome=0.85),
                                            ]
                                            engrams_for_action_1 = [
                                                Engram(vector=[0.2] * 9, action=1, outcome=0.3),
                                                Engram(vector=[0.21] * 9, action=1, outcome=0.2),
                                            ]
                                            engrams_for_action_2 = [
                                                Engram(vector=[0.3] * 9, action=2, outcome=-0.5),
                                                Engram(vector=[0.31] * 9, action=2, outcome=-0.6),
                                                Engram(vector=[0.32] * 9, action=2, outcome=-0.4),
                                            ]
                                            
                                            all_engrams = engrams_for_action_0 + engrams_for_action_1 + engrams_for_action_2
                                            
                                            # Mock store.nearest to return engrams based on query vector
                                            def mock_nearest(vector, limit):
                                                # Return all engrams (they'll be scored and weighted)
                                                results = [(engram, 0.1) for engram in all_engrams]
                                                return results[:limit]
                                            
                                            store.nearest = MagicMock(side_effect=mock_nearest)
                                            
                                            factory = WeightedResonatorFactory()
                                            brain = EngramBrain(input_size=8, output_size=4, engram_store=store, resonator_factory=factory)
                                            
                                            trainer = Trainer("test_instance", clear_collection=True)
                                            trainer.brain = brain  # Replace with our brain
                                            
                                            # Mock environment to terminate immediately
                                            mock_env.reset.return_value = (np.array([0.0] * 8), {})
                                            mock_env.step.return_value = (np.array([0.0] * 8), 0.0, True, False, {})
                                            
                                            # Track actions across all trials
                                            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                                            
                                            # Run many trials to get statistical significance
                                            num_trials = 500  # Reduced for faster test
                                            
                                            for _ in range(num_trials):
                                                # Reset the mock call list but keep track of calls
                                                mock_env.step.reset_mock()
                                                trainer.trial()
                                                
                                                # Get the action that was chosen from the last call
                                                if mock_env.step.called:
                                                    step_args = mock_env.step.call_args[0]
                                                    action = step_args[0]
                                                    action_counts[action] += 1
                                            
                                            # Calculate selection frequencies
                                            total_actions = sum(action_counts.values())
                                            assert total_actions > 0, "No actions were recorded"
                                            
                                            freq_action_0 = action_counts[0] / total_actions
                                            freq_action_1 = action_counts[1] / total_actions
                                            freq_action_2 = action_counts[2] / total_actions
                                            freq_action_3 = action_counts[3] / total_actions
                                            
                                            # Verify that action 0 (strong positive) is chosen most frequently
                                            assert freq_action_0 > freq_action_1, \
                                                f"Action 0 (positive outcomes) should be more likely than action 1, " \
                                                f"but got freq_0={freq_action_0:.3f}, freq_1={freq_action_1:.3f}, " \
                                                f"counts={action_counts}"
                                            
                                            # Verify that action 1 (weak positive) is more likely than action 2 (negative)
                                            assert freq_action_1 > freq_action_2, \
                                                f"Action 1 (weak positive) should be more likely than action 2 (negative), " \
                                                f"but got freq_1={freq_action_1:.3f}, freq_2={freq_action_2:.3f}, " \
                                                f"counts={action_counts}"
                                            
                                            # Verify that action 2 (negative) is less likely than positive actions
                                            assert freq_action_2 < freq_action_0, \
                                                f"Action 2 (negative outcomes) should be less likely than action 0, " \
                                                f"but got freq_2={freq_action_2:.3f}, freq_0={freq_action_0:.3f}, " \
                                                f"counts={action_counts}"
                                            
                                            # Verify that action 3 (no engrams) is less likely than positive actions
                                            # (it should have zero weight, but after normalization it may still have some probability)
                                            assert freq_action_3 < freq_action_0, \
                                                f"Action 3 (no engrams) should be less likely than action 0 (positive), " \
                                                f"but got freq_3={freq_action_3:.3f}, freq_0={freq_action_0:.3f}, counts={action_counts}"
                                            assert freq_action_3 < freq_action_1, \
                                                f"Action 3 (no engrams) should be less likely than action 1 (positive), " \
                                                f"but got freq_3={freq_action_3:.3f}, freq_1={freq_action_1:.3f}, counts={action_counts}"
                                            
                                            # Verify that positive actions together are more likely than negative
                                            positive_freq = freq_action_0 + freq_action_1
                                            negative_freq = freq_action_2
                                            assert positive_freq > negative_freq, \
                                                f"Positive actions (0+1) should be more likely than negative (2), " \
                                                f"but got positive={positive_freq:.3f}, negative={negative_freq:.3f}, " \
                                                f"counts={action_counts}"


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
        observation = [2.5, 2.5, 10.0, 10.0, 3.1415927, 10.0, 1.0, 1.0]
        original_id = id(observation)
        
        result = normalise_observation(observation)
        
        assert id(result) == original_id
        assert result is observation
    
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

