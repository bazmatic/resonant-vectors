
import gymnasium as gym
from EngramBrain import EngramBrain
from engram import EngramStore
import numpy
from WeightedResonatorFactory import WeightedResonatorFactory
from settings import DISPLAY, READ_ONLY, USE_HIT_POINTS, HIT_POINTS, MAX_TRIAL_LENGTH, METABOLIC_COST, PROBABILISTIC_CHOICE
import json
from typing import Dict, List, Any

class Trainer:
    # constructor
    def __init__(self, instance_name: str, dimension_weights: list[float], clear_collection: bool = True):
        self.trial_count = 0
        self.clear_collection = clear_collection
        self.instance_name = instance_name
        # Create the resonator factor and brain      
        self.resonator_factory = WeightedResonatorFactory(dimension_weights)
        self.brain = EngramBrain(9, 4, EngramStore(instance_name, clear_collection), self.resonator_factory)
        if DISPLAY == True:
            render_mode = "human"
        else:
            render_mode = None
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)
        self.feedback_queue = []
        self.best_success = -1.0
        self.mean_success = -1.0
        
        # Initialize metrics tracking
        self.metrics: Dict[str, List[Any]] = {
            'rewards': [],
            'episode_lengths': [],
            'successes': [],  # boolean list, reward >= 200
            'engram_distances': [],  # average distance per episode
            'engram_counts': [],
            'action_distributions': [],  # list of dicts, one per episode
            'best_reward': -float('inf'),
            'rolling_average_50': [],
            'rolling_average_100': []
        }

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
        result = total_reward / trials
        print(f">>>>>>> Average reward: {result}")
        return result


    def trial(self):
        # Run one trial
        self.trial_count += 1
        print(f"Trial: {self.trial_count} ========================")
        self.env.reset()

        hit_points = HIT_POINTS
        total_reward = 0.0
        observation, _ = self.env.reset()
        quit = False
        
        # Track metrics for this episode
        episode_actions = []
        episode_distances = []

        for time_step in range(MAX_TRIAL_LENGTH):
            
            # Get brain output with distance info for metrics
            brain_output, distance = self.brain.decide(observation, self.mean_success + 0.05, return_distance_info=True)
            episode_distances.append(distance)
            
            normalised = [x - min(brain_output) for x in brain_output]

            if PROBABILISTIC_CHOICE == True:    
                output_sum = sum(normalised)
                if output_sum == 0.0:
                    action = numpy.random.choice(numpy.arange(4))
                else:                     
                    probabilities = [x / output_sum for x in normalised]
                    # Then choose one
                    action = numpy.random.choice(numpy.arange(4), p=probabilities)

            else:
                action = numpy.argmax(normalised)
            
            # Track action
            episode_actions.append(action)

            raw_observation, reward, terminated, truncated, info = self.env.step(action)
            observation = normalise_observation(raw_observation)

            total_reward = total_reward + reward
            
            if READ_ONLY == False:
                #print(f"Reward: {reward}")
                # add time to observation array
                self.queue_feedback(observation, action, reward)
                #self.brain.apply_feedback(observation, action, reward)

            if USE_HIT_POINTS == True:
                hit_points += reward
                hit_points -= METABOLIC_COST
                if hit_points < 0:
                    print("** DEAD\n")
                    quit = True

            if terminated or truncated:
                quit = True

            if quit:
                break
        
        self.flush_feedback(total_reward)
        
        # Record metrics for this episode
        episode_length = time_step + 1
        is_success = bool(total_reward >= 200)  # Ensure Python bool, not numpy bool
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

        print(f"*** Length of trial: {episode_length}")
        print(f"*** Total reward: {total_reward}")


        return total_reward
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute and return summary statistics of training metrics."""
        if not self.metrics['rewards']:
            return {
                'total_trials': 0,
                'overall_average_reward': 0.0,
                'success_rate': 0.0,
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
        
        # Prepare data for export
        export_data = {
            'instance_name': self.instance_name,
            'summary_stats': self.get_summary_stats(),
            'outcome_stats': outcome_stats,
            'metrics': {
                'rewards': self.metrics['rewards'],
                'episode_lengths': self.metrics['episode_lengths'],
                'successes': self.metrics['successes'],
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
    
    def flush_feedback(self, success: float):
        if READ_ONLY == False:
       
            # Normalise success from -200 to 200 to -1 to 1
            if success < -300:
                success = -300
            elif success > 300:
                success = 300          
            success = success / 300
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
            #self.mean_success = (self.mean_success * (self.trial_count - 1) + success) / self.trial_count

            print(f"SUCCESS: {success}")
            print(f"MEAN SUCCESS: {self.mean_success}")

             # For each engram in the queue
            for observation, action, reward in self.feedback_queue:
                self.brain.apply_feedback(observation, action, reward, success)
            self.feedback_queue.clear()


def normalise_observation(observation: list[float]) -> list[float]:
    observation[0] = observation[0] / 1.5
    observation[1] = observation[1] / 1.5
    observation[2] = observation[2] / 5.0
    observation[3] = observation[3] / 5.0
    observation[4] = observation[4] / 3.1415927
    observation[5] = observation[5] / 5.0
    observation[6] = 0
    observation[7] = 0
    return observation