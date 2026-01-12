#!/usr/bin/env python3
"""
Weight optimization script for finding optimal VECTOR_COMPONENT_WEIGHTS.
Uses random search to explore the weight space and find configurations
that maximize training performance.
"""

import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

# Import project modules
from Trainer import Trainer
import settings


def generate_random_weights(num_components: int = 8) -> List[float]:
    """Generate random weights in range [0.1, 2.5]."""
    return [round(random.uniform(0.1, 2.5), 2) for _ in range(num_components)]


def mutate_weights(weights: List[float], mutation_rate: float = 0.3, mutation_strength: float = 0.4) -> List[float]:
    """Mutate weights by adding small random changes."""
    new_weights = weights.copy()
    for i in range(len(new_weights)):
        if random.random() < mutation_rate:
            change = random.gauss(0, mutation_strength)
            new_weights[i] = max(0.1, min(2.5, new_weights[i] + change))
            new_weights[i] = round(new_weights[i], 2)
    return new_weights


def run_trial_with_weights(weights: List[float], trials_per_run: int = 500) -> Dict[str, Any]:
    """
    Run a training session with the given weights and return performance metrics.
    """
    # Override the settings
    settings.VECTOR_COMPONENT_WEIGHTS = weights
    settings.DEMO_AFTER_TRAINING = False  # Disable demo for optimization runs
    settings.TRIALS_PER_EXPERIMENT = trials_per_run
    
    # Create trainer and run
    trainer = Trainer("optimizer_run", clear_collection=True)
    
    try:
        trainer.train(trials_per_run, report_interval=100)
        stats = trainer.get_summary_stats()
        
        return {
            'weights': weights,
            'success_rate': stats['success_rate'],
            'average_reward': stats['overall_average_reward'],
            'best_reward': stats['best_episode_reward'],
            'rolling_avg_100': stats['current_rolling_average_100'],
            'death_rate': stats['death_rate'],
        }
    except Exception as e:
        print(f"Error during training: {e}")
        return {
            'weights': weights,
            'success_rate': 0.0,
            'average_reward': -300.0,
            'best_reward': -300.0,
            'rolling_avg_100': -300.0,
            'death_rate': 100.0,
            'error': str(e)
        }


def optimize_weights(
    num_iterations: int = 20,
    trials_per_run: int = 500,
    use_evolution: bool = True
):
    """
    Main optimization loop.
    
    Args:
        num_iterations: Number of different weight configurations to try
        trials_per_run: Number of training trials per configuration
        use_evolution: If True, mutate best weights; if False, pure random search
    """
    results = []
    best_result = None
    best_score = -float('inf')
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"optimization_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("WEIGHT OPTIMIZATION")
    print(f"Iterations: {num_iterations}")
    print(f"Trials per run: {trials_per_run}")
    print(f"Strategy: {'Evolutionary' if use_evolution else 'Random'}")
    print("="*70)
    print()
    
    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print("="*70)
        
        # Generate weights
        if iteration == 0 or not use_evolution:
            # First iteration or pure random: generate random weights
            weights = generate_random_weights()
        else:
            if random.random() < 0.3:
                # 30% chance: try completely random weights for diversity
                weights = generate_random_weights()
            else:
                # 70% chance: mutate best weights found so far
                weights = mutate_weights(best_result['weights'])
        
        print(f"Testing weights: {weights}")
        print()
        
        # Run trial
        result = run_trial_with_weights(weights, trials_per_run)
        results.append(result)
        
        # Score is based on rolling average (more stable than single trial metrics)
        score = result['rolling_avg_100']
        
        print(f"\n--- Results ---")
        print(f"Success Rate: {result['success_rate']:.1f}%")
        print(f"Average Reward: {result['average_reward']:.1f}")
        print(f"Rolling Avg (100): {result['rolling_avg_100']:.1f}")
        print(f"Best Reward: {result['best_reward']:.1f}")
        print(f"Death Rate: {result['death_rate']:.1f}%")
        
        if score > best_score:
            best_score = score
            best_result = result
            print(f"\n🌟 NEW BEST! Score: {score:.1f}")
        else:
            print(f"\nCurrent best score: {best_score:.1f}")
        
        # Save intermediate results
        save_results(results, best_result, results_dir)
    
    # Final report
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest weights found:")
    print(f"  {best_result['weights']}")
    print(f"\nBest performance:")
    print(f"  Success Rate: {best_result['success_rate']:.1f}%")
    print(f"  Average Reward: {best_result['average_reward']:.1f}")
    print(f"  Rolling Avg (100): {best_result['rolling_avg_100']:.1f}")
    print(f"\nResults saved to: {results_dir}")
    print()
    print("To use these weights, update settings.py:")
    print(f"  VECTOR_COMPONENT_WEIGHTS = {best_result['weights']}")
    
    return best_result


def save_results(results: List[Dict], best_result: Dict, results_dir: str):
    """Save optimization results to JSON file."""
    output = {
        'best_result': best_result,
        'all_results': results,
        'num_iterations': len(results)
    }
    
    filepath = os.path.join(results_dir, 'optimization_results.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize VECTOR_COMPONENT_WEIGHTS")
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of weight configurations to try (default: 20)')
    parser.add_argument('--trials', type=int, default=500,
                        help='Training trials per configuration (default: 500)')
    parser.add_argument('--random', action='store_true',
                        help='Use pure random search instead of evolutionary')
    
    args = parser.parse_args()
    
    optimize_weights(
        num_iterations=args.iterations,
        trials_per_run=args.trials,
        use_evolution=not args.random
    )
