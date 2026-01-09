"""
MIN_RESULTS Optimization Test Runner

Systematically tests different MIN_RESULTS values to find the optimal
setting for engram retrieval in the resonant vectors system.
"""

import settings
import csv
from typing import List, Dict, Any
from contextlib import contextmanager

# Note: Trainer is imported inside the function after settings are overridden
# to ensure MIN_RESULTS override takes effect

# Optimal weights from weight optimization
OPTIMAL_WEIGHTS = [0.5, 1.4, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0]

# MIN_RESULTS values to test
MIN_RESULTS_VALUES = [10, 100, 200, 300, 400, 500, 1000]


@contextmanager
def temporary_settings(**overrides):
    """
    Context manager to temporarily override settings.
    
    Args:
        **overrides: Settings to override (e.g., READ_ONLY=True, DISPLAY=False, MIN_RESULTS=100)
    
    Example:
        with temporary_settings(READ_ONLY=True, DISPLAY=False, MIN_RESULTS=200):
            # code that uses modified settings
            pass
    """
    original_values = {}
    try:
        for key, value in overrides.items():
            if hasattr(settings, key):
                original_value = getattr(settings, key)
                original_values[key] = original_value
                
                # Special handling for VECTOR_COMPONENT_WEIGHTS (list)
                # Since Trainer imports it at module level, we need to modify in place
                if key == 'VECTOR_COMPONENT_WEIGHTS' and isinstance(original_value, list):
                    # Store a copy for restoration
                    original_values[key] = original_value.copy()
                    # Modify in place
                    original_value.clear()
                    original_value.extend(value)
                else:
                    setattr(settings, key, value)
        yield
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if key == 'VECTOR_COMPONENT_WEIGHTS' and isinstance(original_value, list):
                # Restore list in place
                current = getattr(settings, key)
                current.clear()
                current.extend(original_value)
            else:
                setattr(settings, key, original_value)


def run_min_results_test(min_results: int) -> Dict[str, Any]:
    """
    Run 50 trials with a specific MIN_RESULTS value.
    
    Args:
        min_results: MIN_RESULTS value to test
    
    Returns:
        Dictionary containing summary statistics
    """
    import importlib
    
    # Temporarily override settings for read-only mode, no display, optimal weights, and MIN_RESULTS
    with temporary_settings(
        READ_ONLY=True,
        DISPLAY=False,
        VECTOR_COMPONENT_WEIGHTS=OPTIMAL_WEIGHTS,
        MIN_RESULTS=min_results
    ):
        # Reload modules that import MIN_RESULTS to pick up the new value
        # EngramBrain imports MIN_RESULTS at module level, so we need to reload it
        import EngramBrain
        importlib.reload(EngramBrain)
        
        # Also reload Trainer since it imports EngramBrain
        import Trainer
        importlib.reload(Trainer)
        
        # Now import Trainer class
        from Trainer import Trainer
        
        # Create trainer with clear_collection=False (read-only mode)
        trainer = Trainer("lander3", clear_collection=False)
        
        # Run 50 trials
        trainer.train(trials=50, report_interval=None, save_interval=None)
        
        # Get summary statistics
        stats = trainer.get_summary_stats()
        
        # Clean up
        trainer.env.close()
        if trainer.action_output_surface is not None:
            import pygame
            pygame.quit()
    
    # Add configuration info to stats
    result = {
        'min_results': min_results,
        'avg_reward': stats['overall_average_reward'],
        'success_rate': stats['success_rate'],
        'death_rate': stats['death_rate'],
        'best_reward': stats['best_episode_reward'],
        'rolling_avg_50': stats['current_rolling_average_50'],
        'rolling_avg_100': stats['current_rolling_average_100'],
        'total_engrams': stats['total_engrams'],
        'avg_engram_distance': stats['average_engram_distance']
    }
    
    return result


def run_optimization(output_file: str = "min_results_optimization.csv"):
    """
    Main function to run MIN_RESULTS optimization tests.
    
    Args:
        output_file: Path to CSV file for results
    """
    print("=" * 60)
    print("MIN_RESULTS Optimization Test Runner")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Using optimal weights: {OPTIMAL_WEIGHTS}")
    print(f"Testing MIN_RESULTS values: {MIN_RESULTS_VALUES}")
    print()
    
    print(f"Will test {len(MIN_RESULTS_VALUES)} MIN_RESULTS configurations")
    print(f"Each configuration will run 50 trials")
    print(f"Total trials: {len(MIN_RESULTS_VALUES) * 50}")
    print()
    
    # Run tests for each MIN_RESULTS value
    results = []
    for idx, min_results in enumerate(MIN_RESULTS_VALUES, 1):
        print(f"[{idx}/{len(MIN_RESULTS_VALUES)}] Testing MIN_RESULTS: {min_results}")
        
        try:
            result = run_min_results_test(min_results)
            results.append(result)
            
            print(f"  ✓ Completed - Avg Reward: {result['avg_reward']:.2f}, "
                  f"Success Rate: {result['success_rate']:.1f}%")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Still add a result entry with error info
            results.append({
                'min_results': min_results,
                'avg_reward': None,
                'success_rate': None,
                'death_rate': None,
                'best_reward': None,
                'rolling_avg_50': None,
                'rolling_avg_100': None,
                'total_engrams': None,
                'avg_engram_distance': None,
                'error': str(e)
            })
        print()
    
    # Save results to CSV
    if results:
        fieldnames = [
            'min_results',
            'avg_reward',
            'success_rate',
            'death_rate',
            'best_reward',
            'rolling_avg_50',
            'rolling_avg_100',
            'total_engrams',
            'avg_engram_distance'
        ]
        
        # Add 'error' field if any results have errors
        if any('error' in r for r in results):
            fieldnames.append('error')
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print("=" * 60)
        print(f"Results saved to: {output_file}")
        print(f"Total configurations tested: {len(results)}")
        
        # Show best result
        valid_results = [r for r in results if r.get('avg_reward') is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x['avg_reward'])
            print(f"\nBest MIN_RESULTS: {best['min_results']}")
            print(f"  Avg Reward: {best['avg_reward']:.2f}")
            print(f"  Success Rate: {best['success_rate']:.1f}%")
            print(f"  Avg Engram Distance: {best['avg_engram_distance']:.4f}")
        
        print("=" * 60)
    else:
        print("No results to save.")


if __name__ == "__main__":
    run_optimization()
