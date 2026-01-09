"""
Utility script for visualizing training metrics.
Loads metrics from JSON files exported by Trainer.save_metrics().
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional


def load_metrics(filename: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_learning_curve(metrics: Dict[str, Any], window_size: int = 50, save_path: Optional[str] = None):
    """
    Plot learning curve showing reward over time with rolling average.
    
    Args:
        metrics: Metrics dictionary loaded from JSON
        window_size: Size of rolling average window
        save_path: Optional path to save the figure
    """
    rewards = metrics['metrics']['rewards']
    trials = list(range(1, len(rewards) + 1))
    
    # Calculate rolling average
    rolling_avg = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        window = rewards[start_idx:i+1]
        rolling_avg.append(sum(window) / len(window))
    
    plt.figure(figsize=(10, 6))
    plt.plot(trials, rewards, alpha=0.3, color='lightblue', label='Episode Reward')
    plt.plot(trials, rolling_avg, color='blue', linewidth=2, label=f'Rolling Average ({window_size} episodes)')
    plt.axhline(y=200, color='green', linestyle='--', label='Solved Threshold (200)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to {save_path}")
    else:
        plt.show()


def plot_success_rate(metrics: Dict[str, Any], window_size: int = 100, save_path: Optional[str] = None):
    """
    Plot success rate over time (percentage of episodes achieving >= 200 points).
    
    Args:
        metrics: Metrics dictionary loaded from JSON
        window_size: Size of window for calculating success rate
        save_path: Optional path to save the figure
    """
    successes = metrics['metrics']['successes']
    trials = list(range(1, len(successes) + 1))
    
    # Calculate rolling success rate
    success_rates = []
    for i in range(len(successes)):
        start_idx = max(0, i - window_size + 1)
        window = successes[start_idx:i+1]
        success_rate = (sum(window) / len(window)) * 100.0
        success_rates.append(success_rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(trials, success_rates, color='green', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title(f'Success Rate (rolling window: {window_size} episodes)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Success rate plot saved to {save_path}")
    else:
        plt.show()


def plot_memory_growth(metrics: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot the growth of engrams in memory over time.
    
    Args:
        metrics: Metrics dictionary loaded from JSON
        save_path: Optional path to save the figure
    """
    engram_counts = metrics['metrics']['engram_counts']
    trials = list(range(1, len(engram_counts) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(trials, engram_counts, color='purple', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Number of Engrams')
    plt.title('Memory Growth Over Time')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Memory growth plot saved to {save_path}")
    else:
        plt.show()


def plot_engram_distances(metrics: Dict[str, Any], window_size: int = 50, save_path: Optional[str] = None):
    """
    Plot average distance to retrieved engrams over time.
    
    Args:
        metrics: Metrics dictionary loaded from JSON
        window_size: Size of rolling average window
        save_path: Optional path to save the figure
    """
    distances = metrics['metrics']['engram_distances']
    # Filter out infinite distances
    valid_distances = [(i, d) for i, d in enumerate(distances) if d != float('inf')]
    
    if not valid_distances:
        print("No valid distance data to plot")
        return
    
    trials = [i + 1 for i, _ in valid_distances]
    dist_values = [d for _, d in valid_distances]
    
    # Calculate rolling average
    rolling_avg = []
    for i in range(len(dist_values)):
        start_idx = max(0, i - window_size + 1)
        window = dist_values[start_idx:i+1]
        rolling_avg.append(sum(window) / len(window))
    
    plt.figure(figsize=(10, 6))
    plt.plot(trials, dist_values, alpha=0.3, color='orange', label='Episode Average Distance')
    plt.plot(trials, rolling_avg, color='red', linewidth=2, label=f'Rolling Average ({window_size} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Distance to Retrieved Engrams')
    plt.title('Engram Similarity Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Engram distances plot saved to {save_path}")
    else:
        plt.show()


def save_settings_file(metrics: Dict[str, Any], output_dir: str):
    """
    Save settings to a text file in the output directory.
    
    Args:
        metrics: Metrics dictionary loaded from JSON
        output_dir: Directory to save the settings file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    settings_file = os.path.join(output_dir, 'settings.txt')
    
    with open(settings_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Training Settings: {metrics.get('instance_name', 'Unknown')}\n")
        f.write("=" * 60 + "\n\n")
        
        if 'settings' in metrics:
            settings = metrics['settings']
            # Group settings by category
            categories = {
                'Core Settings': [
                    'STATE_VECTOR_SIZE', 'OUTPUT_VECTOR_SIZE', 'NOISE_START', 'NOISE_END', 
                    'NOISE_DECAY_RATE', 'NOISE', 'MIN_RESULTS',
                    'READ_ONLY', 'DROP_COLLECTION'
                ],
                'Trial Settings': [
                    'MAX_TRIAL_LENGTH', 'USE_HIT_POINTS', 'HIT_POINTS', 'METABOLIC_COST'
                ],
                'Panic Settings': [
                    'PANIC_ENABLED', 'PANIC_MAX_NOISE'
                ],
                'Action Selection': [
                    'PROBABILISTIC_CHOICE', 'DISPLAY', 'SHOW_ACTION_OUTPUT'
                ],
                'Decay Ranker Settings': [
                    'DECAY_ENABLED', 'DECAY_FUNCTION', 'DECAY_OFFSET_IDS',
                    'DECAY_SCALE_IDS', 'DECAY_VALUE'
                ],
                'Trial Success Multiplier': [
                    'TRIAL_SUCCESS_MULTIPLIER_SCALE'
                ]
            }
            
            for category, keys in categories.items():
                f.write(f"{category}:\n")
                f.write("-" * 60 + "\n")
                for key in keys:
                    if key in settings:
                        value = settings[key]
                        # Format boolean values nicely
                        if isinstance(value, bool):
                            value = 'True' if value else 'False'
                        f.write(f"  {key:30} = {value}\n")
                f.write("\n")
        else:
            f.write("No settings found in metrics file.\n")
    
    print(f"Settings saved to {settings_file}")


def plot_all_metrics(metrics_file: str, output_dir: Optional[str] = None):
    """
    Generate all standard plots from a metrics JSON file.
    
    Args:
        metrics_file: Path to metrics JSON file
        output_dir: Optional directory to save plots (if None, displays plots)
    """
    metrics = load_metrics(metrics_file)
    
    base_name = metrics_file.replace('.json', '') if metrics_file.endswith('.json') else metrics_file
    
    # Save settings file if output directory is specified
    if output_dir:
        save_settings_file(metrics, output_dir)
    
    plots = [
        (plot_learning_curve, 'learning_curve.png'),
        (plot_success_rate, 'success_rate.png'),
        (plot_memory_growth, 'memory_growth.png'),
        (plot_engram_distances, 'engram_distances.png')
    ]
    
    for plot_func, filename in plots:
        save_path = None
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
        
        try:
            plot_func(metrics, save_path=save_path)
        except Exception as e:
            print(f"Error generating {filename}: {e}")


def print_summary(metrics_file: str):
    """Print a text summary of the metrics."""
    metrics = load_metrics(metrics_file)
    stats = metrics['summary_stats']
    outcome_stats = metrics['outcome_stats']
    
    print(f"\n{'='*60}")
    print(f"Training Summary: {metrics['instance_name']}")
    print(f"{'='*60}")
    print(f"Total Trials: {stats['total_trials']}")
    print(f"Overall Average Reward: {stats['overall_average_reward']:.2f}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Best Episode Reward: {stats['best_episode_reward']:.2f}")
    print(f"Rolling Average (last 50): {stats['current_rolling_average_50']:.2f}")
    print(f"Rolling Average (last 100): {stats['current_rolling_average_100']:.2f}")
    print(f"Mean Episode Length: {stats['mean_episode_length']:.1f} steps")
    print(f"\nMemory Statistics:")
    print(f"  Total Engrams: {stats['total_engrams']}")
    print(f"  Average Engram Distance: {stats['average_engram_distance']:.4f}")
    print(f"  Positive Outcome Ratio: {outcome_stats['positive_ratio']:.2%}")
    print(f"  Negative Outcome Ratio: {outcome_stats['negative_ratio']:.2%}")
    print(f"  Mean Outcome: {outcome_stats['mean_outcome']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python metrics_plotter.py <metrics_file.json> [output_dir]")
        print("  If output_dir is provided, plots will be saved there.")
        print("  Otherwise, plots will be displayed interactively.")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print_summary(metrics_file)
    plot_all_metrics(metrics_file, output_dir)


