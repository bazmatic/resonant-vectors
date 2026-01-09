
import gymnasium as gym
from EngramBrain import EngramBrain
from engram import EngramStore
import numpy
from WeightedResonatorFactory import WeightedResonatorFactory
import settings
from settings import DISPLAY, READ_ONLY, USE_HIT_POINTS, HIT_POINTS, MAX_TRIAL_LENGTH, METABOLIC_COST, SHOW_ACTION_OUTPUT, PANIC_ENABLED, PANIC_MAX_NOISE, VECTOR_COMPONENT_WEIGHTS, PAST_HISTORY_STEPS
import json
from typing import Dict, List, Any
import pygame

class Trainer:
    # constructor
    def __init__(self, instance_name: str, clear_collection: bool = True):
        self.trial_count = 0
        self.clear_collection = clear_collection
        self.instance_name = instance_name
        # Create the resonator factor and store
        self.resonator_factory = WeightedResonatorFactory(weights=VECTOR_COMPONENT_WEIGHTS)
        self.engram_store = EngramStore(instance_name, clear_collection)
        # Create the brain (21 = 8 current obs + 8 past obs avg + 4 past action dist + 1 past reward avg)
        self.brain = EngramBrain(21, 4, self.engram_store, self.resonator_factory)
        if DISPLAY == True:
            render_mode = "human"
        else:
            render_mode = None
        self.env = gym.make("LunarLander-v3", render_mode=render_mode)
        self.feedback_queue = []
        self.best_success = -1.0
        self.mean_success = -1.0
        
        # Initialize metrics tracking
        self.metrics: Dict[str, List[Any]] = {
            'rewards': [],
            'episode_lengths': [],
            'successes': [],  # boolean list, reward >= 200
            'deaths': [],  # boolean list, hit points ran out
            'engram_distances': [],  # average distance per episode
            'engram_counts': [],
            'action_distributions': [],  # list of dicts, one per episode
            'best_reward': -float('inf'),
            'rolling_average_50': [],
            'rolling_average_100': []
        }
        
        # Initialize pygame window for action output display if enabled
        self.action_output_surface = None
        self.action_output_background = None
        self.action_output_clock = None
        self.action_output_font = None
        # Action labels: 0=Nothing, 1=Left, 2=Main, 3=Right
        self.action_labels = ['N', 'L', 'M', 'R']
        if DISPLAY == True and SHOW_ACTION_OUTPUT == True:
            pygame.init()
            # Window size: 200x50 for 4 squares (40x40 each with spacing)
            window_width = 200
            window_height = 50
            # Use double buffering to reduce flicker
            self.action_output_surface = pygame.display.set_mode((window_width, window_height), pygame.DOUBLEBUF)
            pygame.display.set_caption("Action Output")
            # Create a background surface to draw to
            self.action_output_background = pygame.Surface((window_width, window_height))
            # Initialize font for labels
            self.action_output_font = pygame.font.Font(None, 36)
            self.action_output_clock = pygame.time.Clock()

    def reset(self):
        self.env.reset()

    # Run a number of trials and return the average reward
    def train(self, trials: int, report_interval: int = None, save_interval: int = None) -> float:
        """
        Train the agent for a specified number of trials.
        
        Args:
            trials: Number of trials to run
            report_interval: If specified, print summary statistics every N trials (default: None, no periodic reporting)
            save_interval: If specified, save metrics backup every N trials (default: None, no periodic saving)
        """
        print(f"Training {self.instance_name}")
        # Create the environment
        
        total_reward = 0.0
        for trial_num in range(trials):
            total_reward += self.trial()
            
            # Periodic reporting
            if report_interval is not None and (trial_num + 1) % report_interval == 0:
                stats = self.get_summary_stats()
                print(f"\n=== Progress Report (Trial {trial_num + 1}/{trials}) ===")
                print(f"Overall Average Reward: {stats['overall_average_reward']:.2f}")
                print(f"Success Rate: {stats['success_rate']:.1f}%")
                print(f"Death Count: {stats['death_count']} ({stats['death_rate']:.1f}%)")
                print(f"Best Episode Reward: {stats['best_episode_reward']:.2f}")
                print(f"Rolling Average (last 50): {stats['current_rolling_average_50']:.2f}")
                print(f"Rolling Average (last 100): {stats['current_rolling_average_100']:.2f}")
                print(f"Total Engrams: {stats['total_engrams']}")
                print(f"Average Engram Distance: {stats['average_engram_distance']:.4f}")
                print("=" * 50 + "\n")
            
            # Periodic saving (backup)
            if save_interval is not None and (trial_num + 1) % save_interval == 0:
                backup_filename = f"training_metrics_backup_trial_{trial_num + 1}.json"
                self.save_metrics(backup_filename)
                print(f"Backup metrics saved to {backup_filename}\n")

        self.env.close()
        
        # Cleanup pygame window if it was created
        if self.action_output_surface is not None:
            pygame.quit()
        
        result = total_reward / trials
        print(f">>>>>>> Average reward: {result}")
        return result


    def trial(self):
        # Run one trial
        self.trial_count += 1
        # print(f"Trial: {self.trial_count} ========================")
        self.env.reset()

        hit_points = HIT_POINTS
        total_reward = 0.0
        observation, _ = self.env.reset()
        quit = False
        death_occurred = False
        
        # Track metrics for this episode
        episode_actions = []
        episode_distances = []
        
        # History buffer for past context (observation, action, reward) tuples
        history_buffer: List[tuple] = []

        for time_step in range(MAX_TRIAL_LENGTH):
            
            # Calculate episode progress (0.0 at start, approaches 1.0 at end)
            episode_progress = time_step / MAX_TRIAL_LENGTH
            
            # Calculate panic factor based on current hit points (if panic is enabled)
            panic_factor = 0.0
            if USE_HIT_POINTS == True and PANIC_ENABLED == True:
                # Calculate panic factor: 0.0 at full hit points, 1.0 at zero hit points
                panic_factor = max(0.0, min(1.0, 1.0 - (hit_points / HIT_POINTS)))
            
            # Compute past averages and extend observation with historical context
            past_averages = self._compute_past_averages(history_buffer)
            extended_observation = list(observation) + past_averages
            
            # Get brain output with distance info for metrics
            brain_output, distance = self.brain.decide(extended_observation, self.mean_success + 0.05, return_distance_info=True, panic_factor=panic_factor, episode_progress=episode_progress)
            episode_distances.append(distance)
            
            # Draw action output if enabled
            if DISPLAY == True and SHOW_ACTION_OUTPUT == True:
                self._draw_action_output(brain_output)
            
            normalised = [x - min(brain_output) for x in brain_output]
            
            # Ensure normalised has exactly 4 elements for LunarLander (4 actions)
            if len(normalised) != 4:
                normalised = normalised[:4] if len(normalised) > 4 else normalised + [0.0] * (4 - len(normalised))

            # Select action with highest score (noise already provides exploration)
            action = numpy.argmax(normalised)
            
            # Track action
            episode_actions.append(action)

            raw_observation, reward, terminated, truncated, info = self.env.step(action)
            observation = normalise_observation(raw_observation)

            total_reward = total_reward + reward
            
            # Update history buffer with the step data (using base 8-element observation)
            history_buffer.append((list(observation), action, reward))
            # Keep only the last N steps
            if len(history_buffer) > PAST_HISTORY_STEPS:
                history_buffer.pop(0)
            
            if READ_ONLY == False:
                #print(f"Reward: {reward}")
                # Queue feedback with the extended observation (21 elements)
                self.queue_feedback(extended_observation, action, reward)
                #self.brain.apply_feedback(observation, action, reward)

            if USE_HIT_POINTS == True:
                hit_points += reward
                hit_points -= METABOLIC_COST
                if hit_points < 0:
                    print(f"** DEATH at step {time_step + 1}: Hit points exhausted (hit_points: {hit_points:.2f})\n")
                    death_occurred = True
                    quit = True

            if terminated or truncated:
                quit = True

            if quit:
                break
        
        # Set reward to maximally negative value if death occurred
        if death_occurred:
            total_reward = -300.0
        
        # Calculate normalized success for trial metadata
        normalized_success = total_reward
        if normalized_success < -300:
            normalized_success = -300
        elif normalized_success > 300:
            normalized_success = 300
        normalized_success = normalized_success / 300
        
        # Record metrics for this episode
        episode_length = time_step + 1
        is_success = bool(total_reward >= 200)  # Ensure Python bool, not numpy bool
        
        # Pass trial metadata to flush_feedback so it can be stored with engrams
        self.flush_feedback(total_reward, normalized_success, episode_length, is_success)
        # Filter out infinite distances before averaging
        valid_distances = [d for d in episode_distances if d != float('inf')]
        avg_distance = sum(valid_distances) / len(valid_distances) if valid_distances else float('inf')
        engram_count = self.brain.engram_store.get_count()
        
        # Calculate action distribution
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for a in episode_actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        total_actions = len(episode_actions)
        action_distribution = {k: v / total_actions if total_actions > 0 else 0.0 for k, v in action_counts.items()}
        
        # Update metrics
        self.metrics['rewards'].append(total_reward)
        self.metrics['episode_lengths'].append(episode_length)
        self.metrics['successes'].append(is_success)
        self.metrics['deaths'].append(death_occurred)
        self.metrics['engram_distances'].append(avg_distance)
        self.metrics['engram_counts'].append(engram_count)
        self.metrics['action_distributions'].append(action_distribution)
        
        # Update best reward
        if total_reward > self.metrics['best_reward']:
            self.metrics['best_reward'] = total_reward
        
        # Calculate rolling averages
        if len(self.metrics['rewards']) >= 50:
            recent_50 = self.metrics['rewards'][-50:]
            self.metrics['rolling_average_50'].append(sum(recent_50) / len(recent_50))
        else:
            self.metrics['rolling_average_50'].append(sum(self.metrics['rewards']) / len(self.metrics['rewards']) if self.metrics['rewards'] else 0.0)
        
        if len(self.metrics['rewards']) >= 100:
            recent_100 = self.metrics['rewards'][-100:]
            self.metrics['rolling_average_100'].append(sum(recent_100) / len(recent_100))
        else:
            self.metrics['rolling_average_100'].append(sum(self.metrics['rewards']) / len(self.metrics['rewards']) if self.metrics['rewards'] else 0.0)

        # print(f"*** Length of trial: {episode_length}")
        # Calculate base noise based on trial progress (not episode progress within trial)
        # This ensures noise monotonically decreases as training progresses
        # Use trial count normalized by a reasonable expectation of total trials
        # This gives a consistent noise level regardless of individual trial length
        from settings import TRIALS_PER_EXPERIMENT
        trial_progress = min(1.0, self.trial_count / TRIALS_PER_EXPERIMENT) if TRIALS_PER_EXPERIMENT > 0 else 0.0
        
        # Calculate and display base noise (without panic factor) based on trial progress
        base_noise = EngramBrain.calculate_base_noise(trial_progress)
        
        # Show positive numbers as green for total_reward
        GREEN = '\033[92m'
        RESET = '\033[0m'
        reward_str = f"{total_reward:.1f}"
        if total_reward > 0:
            reward_str = f"{GREEN}{reward_str}{RESET}"
        print(f"*** Trial {self.trial_count} | Rolling Average: {self.metrics['rolling_average_100'][-1]:.1f} | Total reward: {reward_str} | Noise: {base_noise:.3f}")
        # if death_occurred:
        #     print(f"*** DEATH: Hit points exhausted")


        return total_reward
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute and return summary statistics of training metrics."""
        if not self.metrics['rewards']:
            return {
                'total_trials': 0,
                'overall_average_reward': 0.0,
                'success_rate': 0.0,
                'death_count': 0,
                'death_rate': 0.0,
                'best_episode_reward': -float('inf'),
                'current_rolling_average_50': 0.0,
                'current_rolling_average_100': 0.0,
                'total_engrams': 0,
                'average_engram_distance': 0.0,
                'mean_episode_length': 0.0
            }
        
        total_trials = len(self.metrics['rewards'])
        overall_avg = sum(self.metrics['rewards']) / total_trials
        success_count = sum(self.metrics['successes'])
        success_rate = (success_count / total_trials) * 100.0 if total_trials > 0 else 0.0
        death_count = sum(self.metrics['deaths'])
        death_rate = (death_count / total_trials) * 100.0 if total_trials > 0 else 0.0
        
        current_rolling_50 = self.metrics['rolling_average_50'][-1] if self.metrics['rolling_average_50'] else 0.0
        current_rolling_100 = self.metrics['rolling_average_100'][-1] if self.metrics['rolling_average_100'] else 0.0
        
        total_engrams = self.brain.engram_store.get_count()
        
        # Average engram distance from recent episodes (last 20)
        recent_distances = self.metrics['engram_distances'][-20:] if len(self.metrics['engram_distances']) >= 20 else self.metrics['engram_distances']
        # Filter out infinite distances before averaging
        valid_distances = [d for d in recent_distances if d != float('inf')]
        avg_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 0.0
        
        mean_episode_length = sum(self.metrics['episode_lengths']) / total_trials if total_trials > 0 else 0.0
        
        return {
            'total_trials': total_trials,
            'overall_average_reward': overall_avg,
            'success_rate': success_rate,
            'death_count': death_count,
            'death_rate': death_rate,
            'best_episode_reward': self.metrics['best_reward'],
            'current_rolling_average_50': current_rolling_50,
            'current_rolling_average_100': current_rolling_100,
            'total_engrams': total_engrams,
            'average_engram_distance': avg_distance,
            'mean_episode_length': mean_episode_length
        }
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, (numpy.integer, numpy.floating)):
            return float(obj)
        elif isinstance(obj, (numpy.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def save_metrics(self, filename: str) -> None:
        """Export all metrics to a JSON file."""
        # Get outcome stats from engram store
        outcome_stats = self.brain.engram_store.get_outcome_stats()
        
        # Collect all settings from settings module
        settings_dict = {
            'STATE_VECTOR_SIZE': settings.STATE_VECTOR_SIZE,
            'OUTPUT_VECTOR_SIZE': settings.OUTPUT_VECTOR_SIZE,
            'PAST_HISTORY_STEPS': settings.PAST_HISTORY_STEPS,
            'NOISE_START': settings.NOISE_START,
            'NOISE_END': settings.NOISE_END,
            'NOISE_DECAY_RATE': settings.NOISE_DECAY_RATE,
            'NOISE': settings.NOISE,  # Backward compatibility
            'MIN_RESULTS': settings.MIN_RESULTS,
            'READ_ONLY': settings.READ_ONLY,
            'DROP_COLLECTION': settings.DROP_COLLECTION,
            'USE_HIT_POINTS': settings.USE_HIT_POINTS,
            'HIT_POINTS': settings.HIT_POINTS,
            'MAX_TRIAL_LENGTH': settings.MAX_TRIAL_LENGTH,
            'METABOLIC_COST': settings.METABOLIC_COST,
            'PANIC_ENABLED': settings.PANIC_ENABLED,
            'PANIC_MAX_NOISE': settings.PANIC_MAX_NOISE,
            'DISPLAY': settings.DISPLAY,
            'SHOW_ACTION_OUTPUT': settings.SHOW_ACTION_OUTPUT,
            'DECAY_ENABLED': settings.DECAY_ENABLED,
            'DECAY_FUNCTION': settings.DECAY_FUNCTION,
            'DECAY_OFFSET_IDS': settings.DECAY_OFFSET_IDS,
            'DECAY_SCALE_IDS': settings.DECAY_SCALE_IDS,
            'DECAY_VALUE': settings.DECAY_VALUE,
            'TRIAL_SUCCESS_MULTIPLIER_SCALE': settings.TRIAL_SUCCESS_MULTIPLIER_SCALE,
            'VECTOR_COMPONENT_WEIGHTS': settings.VECTOR_COMPONENT_WEIGHTS,
            'VECTOR_SAVE_RATE': settings.VECTOR_SAVE_RATE
        }
        
        # Prepare data for export
        export_data = {
            'instance_name': self.instance_name,
            'settings': settings_dict,
            'summary_stats': self.get_summary_stats(),
            'outcome_stats': outcome_stats,
            'metrics': {
                'rewards': self.metrics['rewards'],
                'episode_lengths': self.metrics['episode_lengths'],
                'successes': self.metrics['successes'],
                'deaths': self.metrics['deaths'],
                'engram_distances': self.metrics['engram_distances'],
                'engram_counts': self.metrics['engram_counts'],
                'action_distributions': self.metrics['action_distributions'],
                'best_reward': self.metrics['best_reward'],
                'rolling_average_50': self.metrics['rolling_average_50'],
                'rolling_average_100': self.metrics['rolling_average_100']
            }
        }
        
        # Convert numpy types to JSON-serializable types
        export_data = self._convert_to_json_serializable(export_data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Metrics saved to {filename}")
    
    def queue_feedback(self, observation: list[float], action: int, reward: float):
        self.feedback_queue.append((observation, action, reward))
    
    def flush_feedback(self, total_reward: float, normalized_success: float, episode_length: int, is_success: bool):
        if READ_ONLY == False:
            # Use normalized_success (already normalized)
            success = normalized_success
            if success > self.best_success:
                self.best_success = success
            
            # Calculate mean success based on the last 20 trials
            if self.mean_success == -1.0:
                self.mean_success = success
            else:
                if self.trial_count < 20:
                    self.mean_success = (self.mean_success * (self.trial_count - 1) + success) / self.trial_count
                else:
                    self.mean_success = (self.mean_success * 19 + success) / 20

            # print(f"SUCCESS: {success}")
            # print(f"MEAN SUCCESS: {self.mean_success}")

            # Batch insert all engrams from the queue for better performance
            if len(self.feedback_queue) > 0:
                observations, actions, rewards = zip(*self.feedback_queue)
                self.brain.batch_apply_feedback(
                    list(observations), 
                    list(actions), 
                    list(rewards), 
                    success, 
                    normalized_success
                )
            self.feedback_queue.clear()
    
    def _compute_past_averages(self, history_buffer: List[tuple]) -> List[float]:
        """
        Compute rolling averages of past observations, actions, and rewards.
        
        Args:
            history_buffer: List of (observation, action, reward) tuples from recent steps
            
        Returns:
            13-element list containing:
            - [0-7]: Average of each observation component over last N steps
            - [8-11]: Action distribution (proportion of each action 0-3)
            - [12]: Average reward over last N steps
        """
        if len(history_buffer) == 0:
            # No history yet - return zeros
            return [0.0] * 13
        
        # Compute average of observations (8 components)
        obs_sums = [0.0] * 8
        for obs, _, _ in history_buffer:
            for i in range(8):
                obs_sums[i] += obs[i]
        obs_avgs = [s / len(history_buffer) for s in obs_sums]
        
        # Compute action distribution (4 values)
        action_counts = [0, 0, 0, 0]
        for _, action, _ in history_buffer:
            action_counts[action] += 1
        action_dist = [c / len(history_buffer) for c in action_counts]
        
        # Compute average reward (1 value)
        reward_sum = sum(reward for _, _, reward in history_buffer)
        reward_avg = reward_sum / len(history_buffer)
        
        # Combine all: 8 obs avgs + 4 action dist + 1 reward avg = 13 values
        return obs_avgs + action_dist + [reward_avg]
    
    def _value_to_color(self, value: float) -> tuple[int, int, int]:
        """
        Map a value in the range [-1, 1] to an RGB color.
        -1 = red (255, 0, 0)
        0 = yellow (255, 255, 0)
        +1 = green (0, 255, 0)
        """
        # Clamp value to [-1, 1]
        value = max(-1.0, min(1.0, value))
        
        if value <= 0:
            # Interpolate from red to yellow (value goes from -1 to 0)
            # When value = -1, we want red (255, 0, 0)
            # When value = 0, we want yellow (255, 255, 0)
            t = (value + 1.0)  # Maps -1->0 to 0->1
            r = 255
            g = int(255 * t)
            b = 0
        else:
            # Interpolate from yellow to green (value goes from 0 to 1)
            # When value = 0, we want yellow (255, 255, 0)
            # When value = 1, we want green (0, 255, 0)
            t = value  # Maps 0->1 to 0->1
            r = int(255 * (1.0 - t))
            g = 255
            b = 0
        
        return (r, g, b)
    
    def _draw_action_output(self, brain_output: list[float]) -> None:
        """
        Draw action output values as colored squares in the pygame window.
        Normalizes values to [-1, 1] range using min-max normalization.
        """
        if self.action_output_surface is None or self.action_output_background is None:
            return
        
        # Normalize brain_output values to [-1, 1] range
        if len(brain_output) == 0:
            return
        
        min_val = min(brain_output)
        max_val = max(brain_output)
        
        if max_val == min_val:
            # All values are the same, set all to 0 (yellow)
            normalized_values = [0.0] * len(brain_output)
        else:
            # Min-max normalization to [-1, 1]
            normalized_values = [2.0 * (val - min_val) / (max_val - min_val) - 1.0 for val in brain_output]
        
        # Clear the background surface with black
        self.action_output_background.fill((0, 0, 0))
        
        # Draw squares for each action on the background surface
        square_size = 40
        spacing = 10
        start_x = spacing
        start_y = 5
        
        for i, normalized_value in enumerate(normalized_values):
            color = self._value_to_color(normalized_value)
            x = start_x + i * (square_size + spacing)
            y = start_y
            pygame.draw.rect(self.action_output_background, color, (x, y, square_size, square_size))
            
            # Draw label on the square
            if self.action_output_font is not None and i < len(self.action_labels):
                label_text = self.action_labels[i]
                # Use white text for visibility on colored backgrounds
                text_surface = self.action_output_font.render(label_text, True, (255, 255, 255))
                # Center the text in the square
                text_x = x + (square_size - text_surface.get_width()) // 2
                text_y = y + (square_size - text_surface.get_height()) // 2
                self.action_output_background.blit(text_surface, (text_x, text_y))
        
        # Blit the background surface to the display surface
        self.action_output_surface.blit(self.action_output_background, (0, 0))
        
        # Update the display using flip() for double buffering
        pygame.display.flip()
        
        # Limit frame rate to reduce flickering (30 FPS should be smooth)
        if self.action_output_clock is not None:
            self.action_output_clock.tick(30)
        
        # Handle pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass  # Don't quit, just ignore


def normalise_observation(observation: list[float]) -> list[float]:
    # Normalize all components to [-1, 1] range with proper scaling and clamping
    # Position (x, y): divide by 2.5 (max absolute value: ±2.5)
    observation[0] = max(-1.0, min(1.0, observation[0] / 2.5))
    observation[1] = max(-1.0, min(1.0, observation[1] / 2.5))
    # Velocities (vx, vy): divide by 10 (max absolute value: ±10)
    observation[2] = max(-1.0, min(1.0, observation[2] / 10.0))
    observation[3] = max(-1.0, min(1.0, observation[3] / 10.0))
    # Angle: divide by π (max absolute value: ±π)
    observation[4] = max(-1.0, min(1.0, observation[4] / 3.1415927))
    # Angular velocity: divide by 10 (max absolute value: ±10)
    observation[5] = max(-1.0, min(1.0, observation[5] / 10.0))
    # Leg contact booleans: map 0→-1, 1→1
    observation[6] = -1.0 if observation[6] == 0.0 else 1.0
    observation[7] = -1.0 if observation[7] == 0.0 else 1.0
    return observation