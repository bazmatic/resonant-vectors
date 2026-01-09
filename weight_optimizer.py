"""
Weight Optimization Test Runner

Systematically tests different vector component weight configurations
to find optimal weights for the resonant vectors system.
"""

import settings
from Trainer import Trainer
import csv
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager
from collections import defaultdict
import os

# Number of trials to run per weight configuration
TRIALS_PER_WEIGHT_CONFIG = 100

# Target sum for all weight configurations (8.0 = neutral case where all weights = 1.0)
# Normalizing to a fixed sum ensures we explore relative component importance
# rather than absolute scale, since the effective weight is affected by the overall share
TARGET_WEIGHT_SUM = 8.0


def normalize_weights(weights: List[float], target_sum: float = TARGET_WEIGHT_SUM) -> List[float]:
    """
    Normalize weights to a fixed target sum, preserving relative ratios.
    
    This ensures all weight configurations have the same total sum during exploration,
    allowing fair comparison of relative component importance rather than absolute scale.
    
    Args:
        weights: List of weight values to normalize
        target_sum: Target sum for normalized weights (default: TARGET_WEIGHT_SUM)
    
    Returns:
        Normalized weights list that sums to target_sum
    """
    if not weights:
        return weights
    
    current_sum = sum(weights)
    
    # Handle edge case where sum is 0 or very small
    if abs(current_sum) < 1e-10:
        # Return equal weights if sum is effectively zero
        return [target_sum / len(weights)] * len(weights)
    
    # Scale all weights proportionally to achieve target sum
    scale_factor = target_sum / current_sum
    normalized = [w * scale_factor for w in weights]
    
    return normalized


def generate_weight_configs() -> List[Tuple[str, List[float]]]:
    """
    Generate weight configurations to test.
    All configurations are normalized to TARGET_WEIGHT_SUM to ensure fair comparison.
    
    Returns:
        List of tuples: (config_name, weights_list)
    """
    configs = []
    
    # Baseline: current weights from settings, normalized to target sum
    baseline_weights = normalize_weights(list(settings.VECTOR_COMPONENT_WEIGHTS))
    configs.append(("baseline", baseline_weights))
    
    # Component sensitivity: test each component individually
    # Test weights: [0.1, 0.5, 1.0, 1.5, 2.0]
    test_weights = [0.1, 0.5, 1.0, 1.5, 2.0]
    component_names = [
        "x_position",
        "y_position", 
        "vx_velocity",
        "vy_velocity",
        "angle",
        "angular_velocity",
        "leg_contact_1",
        "leg_contact_2"
    ]
    
    for component_idx in range(8):
        for weight_value in test_weights:
            # Create weights with all components at 1.0 except the test component
            weights = [1.0] * 8
            weights[component_idx] = weight_value
            # Normalize to target sum before adding to configs
            weights = normalize_weights(weights)
            config_name = f"{component_names[component_idx]}_{weight_value}"
            configs.append((config_name, weights))
    
    return configs


@contextmanager
def temporary_settings(**overrides):
    """
    Context manager to temporarily override settings.
    
    Args:
        **overrides: Settings to override (e.g., READ_ONLY=True, DISPLAY=False)
    
    Example:
        with temporary_settings(READ_ONLY=True, DISPLAY=False):
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


def analyze_results(csv_file: str) -> Dict[str, Any]:
    """
    Analyze CSV results and extract performance insights.
    
    Args:
        csv_file: Path to CSV file with test results
    
    Returns:
        Dictionary containing analysis data:
        - component_performance: Dict mapping component_index -> weight_value -> metrics
        - best_configs: List of top-performing configurations
        - component_best_weights: Dict mapping component_index -> best weight value
    """
    if not os.path.exists(csv_file):
        return {
            'component_performance': {},
            'best_configs': [],
            'component_best_weights': {},
            'error': 'File not found'
        }
    
    results = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with errors or missing avg_reward
            if 'error' in row and row['error']:
                continue
            if not row.get('avg_reward') or row['avg_reward'] == '':
                continue
            
            try:
                row['avg_reward'] = float(row['avg_reward'])
                row['success_rate'] = float(row.get('success_rate', 0))
                row['weights'] = [float(w.strip()) for w in row['weights'].split(',')]
                results.append(row)
            except (ValueError, KeyError):
                continue
    
    if not results:
        return {
            'component_performance': {},
            'best_configs': [],
            'component_best_weights': {},
            'error': 'No valid results found'
        }
    
    # Component names for reference
    component_names = [
        "x_position", "y_position", "vx_velocity", "vy_velocity",
        "angle", "angular_velocity", "leg_contact_1", "leg_contact_2"
    ]
    
    # Analyze component performance
    # Group by component and weight value
    component_performance = defaultdict(lambda: defaultdict(lambda: {'rewards': [], 'success_rates': [], 'count': 0}))
    
    for result in results:
        weights = result['weights']
        avg_reward = result['avg_reward']
        success_rate = result['success_rate']
        
        # Check if this is a single-component test (all weights are 1.0 except one)
        non_one_indices = [i for i, w in enumerate(weights) if abs(w - 1.0) > 0.01]
        
        if len(non_one_indices) == 1:
            # Single component test
            component_idx = non_one_indices[0]
            weight_value = weights[component_idx]
            component_performance[component_idx][weight_value]['rewards'].append(avg_reward)
            component_performance[component_idx][weight_value]['success_rates'].append(success_rate)
            component_performance[component_idx][weight_value]['count'] += 1
    
    # Calculate averages per component/weight
    component_avg_performance = {}
    for component_idx, weight_data in component_performance.items():
        component_avg_performance[component_idx] = {}
        for weight_value, metrics in weight_data.items():
            if metrics['count'] > 0:
                component_avg_performance[component_idx][weight_value] = {
                    'avg_reward': sum(metrics['rewards']) / len(metrics['rewards']),
                    'avg_success_rate': sum(metrics['success_rates']) / len(metrics['success_rates']),
                    'count': metrics['count']
                }
    
    # Find best weight for each component
    component_best_weights = {}
    for component_idx, weight_data in component_avg_performance.items():
        if weight_data:
            # Find weight with highest average reward
            best_weight = max(weight_data.items(), key=lambda x: x[1]['avg_reward'])
            component_best_weights[component_idx] = {
                'weight': best_weight[0],
                'avg_reward': best_weight[1]['avg_reward'],
                'avg_success_rate': best_weight[1]['avg_success_rate']
            }
    
    # Get top-performing configurations overall
    best_configs = sorted(results, key=lambda x: x['avg_reward'], reverse=True)[:10]
    
    return {
        'component_performance': dict(component_avg_performance),
        'component_best_weights': component_best_weights,
        'best_configs': best_configs,
        'total_configs': len(results),
        'component_names': component_names
    }


def suggest_next_configs(analysis: Dict[str, Any], num_suggestions: int = 10, 
                        tested_configs: set = None) -> List[Tuple[str, List[float]]]:
    """
    Generate suggested weight configurations based on analysis.
    
    Args:
        analysis: Analysis dictionary from analyze_results()
        num_suggestions: Number of configurations to suggest
        tested_configs: Set of weight tuples (as strings) that have already been tested
    
    Returns:
        List of (config_name, weights_list) tuples
    """
    if tested_configs is None:
        tested_configs = set()
    
    suggestions = []
    component_names = analysis.get('component_names', [
        "x_position", "y_position", "vx_velocity", "vy_velocity",
        "angle", "angular_velocity", "leg_contact_1", "leg_contact_2"
    ])
    
    component_best_weights = analysis.get('component_best_weights', {})
    best_configs = analysis.get('best_configs', [])
    
    # Strategy 1: Combine best-performing individual component weights
    if component_best_weights and len(component_best_weights) >= 2:
        # Create a configuration using best weights from top components
        best_components = sorted(
            component_best_weights.items(),
            key=lambda x: x[1]['avg_reward'],
            reverse=True
        )[:4]  # Top 4 components
        
        if best_components:
            weights = [1.0] * 8
            config_parts = []
            for component_idx, data in best_components:
                weights[component_idx] = data['weight']
                config_parts.append(f"{component_names[component_idx]}_{data['weight']}")
            
            # Normalize to target sum before checking tested_configs
            weights = normalize_weights(weights)
            weight_key = ','.join(map(str, weights))
            if weight_key not in tested_configs:
                suggestions.append(("combined_best_" + "_".join(config_parts[:3]), weights))
                tested_configs.add(weight_key)
    
    # Strategy 2: Refine around best overall configurations
    if best_configs and len(suggestions) < num_suggestions:
        for config in best_configs[:3]:  # Top 3 configs
            base_weights = config['weights']
            # Ensure base_weights is a list (it might be a string from CSV)
            if isinstance(base_weights, str):
                base_weights = [float(w.strip()) for w in base_weights.split(',')]
            
            # Try variations: slightly increase/decrease each component
            for component_idx in range(8):
                for adjustment in [0.1, -0.1, 0.2, -0.2]:
                    new_weights = base_weights.copy()
                    new_weights[component_idx] = max(0.1, min(3.0, new_weights[component_idx] + adjustment))
                    # Normalize to target sum before checking tested_configs
                    new_weights = normalize_weights(new_weights)
                    
                    weight_key = ','.join(map(str, new_weights))
                    if weight_key not in tested_configs and len(suggestions) < num_suggestions:
                        config_name = f"refine_{config['config_name']}_{component_names[component_idx]}_{adjustment:+.1f}"
                        suggestions.append((config_name, new_weights))
                        tested_configs.add(weight_key)
    
    # Strategy 3: Multi-component variations around best individual weights
    if component_best_weights and len(suggestions) < num_suggestions:
        # Get top 3 best components
        top_components = sorted(
            component_best_weights.items(),
            key=lambda x: x[1]['avg_reward'],
            reverse=True
        )[:3]
        
        if len(top_components) >= 2:
            # Try combinations of 2-3 top components
            for i, (idx1, data1) in enumerate(top_components):
                for j, (idx2, data2) in enumerate(top_components[i+1:], i+1):
                    weights = [1.0] * 8
                    weights[idx1] = data1['weight']
                    weights[idx2] = data2['weight']
                    # Normalize to target sum before checking tested_configs
                    weights = normalize_weights(weights)
                    
                    weight_key = ','.join(map(str, weights))
                    if weight_key not in tested_configs and len(suggestions) < num_suggestions:
                        config_name = f"pair_{component_names[idx1]}_{data1['weight']}_{component_names[idx2]}_{data2['weight']}"
                        suggestions.append((config_name, weights))
                        tested_configs.add(weight_key)
                    
                    # Try with third component if available
                    if len(top_components) >= 3 and len(suggestions) < num_suggestions:
                        idx3, data3 = top_components[2]
                        weights3 = weights.copy()
                        weights3[idx3] = data3['weight']
                        # Normalize to target sum before checking tested_configs
                        weights3 = normalize_weights(weights3)
                        weight_key3 = ','.join(map(str, weights3))
                        if weight_key3 not in tested_configs:
                            config_name3 = f"triple_{component_names[idx1]}_{data1['weight']}_{component_names[idx2]}_{data2['weight']}_{component_names[idx3]}_{data3['weight']}"
                            suggestions.append((config_name3, weights3))
                            tested_configs.add(weight_key3)
    
    # Strategy 4: Fill remaining slots with promising single-component refinements
    if component_best_weights and len(suggestions) < num_suggestions:
        for component_idx, data in sorted(
            component_best_weights.items(),
            key=lambda x: x[1]['avg_reward'],
            reverse=True
        ):
            best_weight = data['weight']
            # Try weights around the best weight
            for offset in [0.2, 0.3, -0.2, -0.3, 0.4, -0.4]:
                if len(suggestions) >= num_suggestions:
                    break
                new_weight = max(0.1, min(3.0, best_weight + offset))
                weights = [1.0] * 8
                weights[component_idx] = new_weight
                # Normalize to target sum before checking tested_configs
                weights = normalize_weights(weights)
                
                weight_key = ','.join(map(str, weights))
                if weight_key not in tested_configs:
                    config_name = f"refine_{component_names[component_idx]}_{new_weight:.1f}"
                    suggestions.append((config_name, weights))
                    tested_configs.add(weight_key)
    
    return suggestions[:num_suggestions]


def run_weight_test(weights: List[float], config_name: str, iteration: int = 0, 
                   config_type: str = "systematic", suggestion_basis: str = None) -> Dict[str, Any]:
    """
    Run trials with a specific weight configuration.
    
    Args:
        weights: List of 8 weight values for vector components
        config_name: Identifier for this configuration
        iteration: Iteration/round number (0 = initial systematic, 1+ = refinement)
        config_type: Type of configuration ("systematic" or "suggested")
        suggestion_basis: Optional description of what analysis led to this suggestion
    
    Returns:
        Dictionary containing summary statistics
    """
    # Temporarily override settings for read-only mode and no display
    with temporary_settings(READ_ONLY=True, DISPLAY=False, VECTOR_COMPONENT_WEIGHTS=weights):
        # Create trainer with clear_collection=False (read-only mode)
        trainer = Trainer("lander3", clear_collection=False)
        
        # Run trials per weight configuration
        trainer.train(trials=TRIALS_PER_WEIGHT_CONFIG, report_interval=None, save_interval=None)
        
        # Get summary statistics
        stats = trainer.get_summary_stats()
        
        # Clean up
        trainer.env.close()
        if trainer.action_output_surface is not None:
            import pygame
            pygame.quit()
    
    # Add configuration info to stats
    result = {
        'config_name': config_name,
        'weights': ','.join(map(str, weights)),
        'iteration': iteration,
        'config_type': config_type,
        'avg_reward': stats['overall_average_reward'],
        'success_rate': stats['success_rate'],
        'death_rate': stats['death_rate'],
        'best_reward': stats['best_episode_reward'],
        'rolling_avg_50': stats['current_rolling_average_50'],
        'rolling_avg_100': stats['current_rolling_average_100'],
        'total_engrams': stats['total_engrams'],
        'avg_engram_distance': stats['average_engram_distance']
    }
    
    if suggestion_basis:
        result['suggestion_basis'] = suggestion_basis
    
    return result


def run_optimization(output_file: str = "weight_optimization_results.csv"):
    """
    Main function to run weight optimization tests.
    
    Args:
        output_file: Path to CSV file for results
    """
    print("=" * 60)
    print("Weight Optimization Test Runner")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print()
    
    # Generate weight configurations
    configs = generate_weight_configs()
    print(f"Generated {len(configs)} weight configurations to test")
    print(f"Each configuration will run {TRIALS_PER_WEIGHT_CONFIG} trials")
    print(f"Total trials: {len(configs) * TRIALS_PER_WEIGHT_CONFIG}")
    print()
    
    # Run tests for each configuration
    results = []
    for idx, (config_name, weights) in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] Testing configuration: {config_name}")
        print(f"  Weights: {weights}")
        
        try:
            result = run_weight_test(weights, config_name, iteration=0, config_type="systematic")
            results.append(result)
            
            print(f"  ✓ Completed - Avg Reward: {result['avg_reward']:.2f}, "
                  f"Success Rate: {result['success_rate']:.1f}%")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Still add a result entry with error info
            results.append({
                'config_name': config_name,
                'weights': ','.join(map(str, weights)),
                'iteration': 0,
                'config_type': 'systematic',
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
        _save_results(results, output_file, append=False)
        
        print("=" * 60)
        print(f"Results saved to: {output_file}")
        print(f"Total configurations tested: {len(results)}")
        print("=" * 60)
    else:
        print("No results to save.")


def run_optimization_iterative(output_file: str = "weight_optimization_results.csv", 
                              max_iterations: int = 3, 
                              suggestions_per_iteration: int = 10):
    """
    Run optimization with iterative refinement.
    
    Args:
        output_file: Path to CSV file for results
        max_iterations: Maximum number of refinement iterations (0 = systematic only)
        suggestions_per_iteration: Number of suggested configs to test per iteration
    """
    print("=" * 60)
    print("Weight Optimization Test Runner (Iterative Mode)")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Max iterations: {max_iterations}")
    print(f"Suggestions per iteration: {suggestions_per_iteration}")
    print()
    
    all_results = []
    
    # Iteration 0: Run initial systematic tests
    print("=" * 60)
    print("ITERATION 0: Initial Systematic Tests")
    print("=" * 60)
    print()
    
    configs = generate_weight_configs()
    print(f"Generated {len(configs)} weight configurations to test")
    print(f"Each configuration will run {TRIALS_PER_WEIGHT_CONFIG} trials")
    print()
    
    iteration_results = []
    for idx, (config_name, weights) in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] Testing configuration: {config_name}")
        print(f"  Weights: {weights}")
        
        try:
            result = run_weight_test(weights, config_name, iteration=0, config_type="systematic")
            iteration_results.append(result)
            all_results.append(result)
            
            print(f"  ✓ Completed - Avg Reward: {result['avg_reward']:.2f}, "
                  f"Success Rate: {result['success_rate']:.1f}%")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_result = {
                'config_name': config_name,
                'weights': ','.join(map(str, weights)),
                'iteration': 0,
                'config_type': 'systematic',
                'avg_reward': None,
                'success_rate': None,
                'death_rate': None,
                'best_reward': None,
                'rolling_avg_50': None,
                'rolling_avg_100': None,
                'total_engrams': None,
                'avg_engram_distance': None,
                'error': str(e)
            }
            iteration_results.append(error_result)
            all_results.append(error_result)
        print()
    
    # Save initial results
    _save_results(all_results, output_file)
    print(f"Initial results saved to: {output_file}")
    print()
    
    # Iterative refinement
    for iteration in range(1, max_iterations + 1):
        print("=" * 60)
        print(f"ITERATION {iteration}: Refinement")
        print("=" * 60)
        print()
        
        # Analyze current results
        print("Analyzing results...")
        analysis = analyze_results(output_file)
        
        if 'error' in analysis:
            print(f"  ✗ Analysis error: {analysis['error']}")
            print("  Skipping further iterations.")
            break
        
        print(f"  ✓ Analyzed {analysis['total_configs']} configurations")
        
        # Show component insights
        if analysis['component_best_weights']:
            print("\n  Top component weights:")
            for component_idx, data in sorted(
                analysis['component_best_weights'].items(),
                key=lambda x: x[1]['avg_reward'],
                reverse=True
            )[:3]:
                component_name = analysis['component_names'][component_idx]
                print(f"    {component_name}: weight={data['weight']:.2f}, "
                      f"avg_reward={data['avg_reward']:.2f}")
        
        # Generate suggestions
        print(f"\n  Generating {suggestions_per_iteration} suggestions...")
        
        # Get already tested configurations
        tested_configs = set()
        for result in all_results:
            if 'weights' in result and result['weights']:
                tested_configs.add(result['weights'])
        
        suggestions = suggest_next_configs(
            analysis, 
            num_suggestions=suggestions_per_iteration,
            tested_configs=tested_configs
        )
        
        if not suggestions:
            print("  No new suggestions generated. Stopping iterations.")
            break
        
        print(f"  ✓ Generated {len(suggestions)} suggestions")
        print()
        
        # Run suggested configurations
        iteration_results = []
        for idx, (config_name, weights) in enumerate(suggestions, 1):
            print(f"[{idx}/{len(suggestions)}] Testing suggestion: {config_name}")
            print(f"  Weights: {weights}")
            
            try:
                result = run_weight_test(
                    weights, 
                    config_name, 
                    iteration=iteration, 
                    config_type="suggested",
                    suggestion_basis=f"iteration_{iteration}_refinement"
                )
                iteration_results.append(result)
                all_results.append(result)
                
                print(f"  ✓ Completed - Avg Reward: {result['avg_reward']:.2f}, "
                      f"Success Rate: {result['success_rate']:.1f}%")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                error_result = {
                    'config_name': config_name,
                    'weights': ','.join(map(str, weights)),
                    'iteration': iteration,
                    'config_type': 'suggested',
                    'suggestion_basis': f"iteration_{iteration}_refinement",
                    'avg_reward': None,
                    'success_rate': None,
                    'death_rate': None,
                    'best_reward': None,
                    'rolling_avg_50': None,
                    'rolling_avg_100': None,
                    'total_engrams': None,
                    'avg_engram_distance': None,
                    'error': str(e)
                }
                iteration_results.append(error_result)
                all_results.append(error_result)
            print()
        
        # Append results to CSV
        _save_results(iteration_results, output_file, append=True)
        print(f"Iteration {iteration} results appended to: {output_file}")
        print()
        
        # Show best so far
        valid_results = [r for r in all_results if r.get('avg_reward') is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x['avg_reward'])
            print(f"  Best configuration so far: {best['config_name']}")
            print(f"    Avg Reward: {best['avg_reward']:.2f}, Success Rate: {best['success_rate']:.1f}%")
            print()
    
    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    valid_results = [r for r in all_results if r.get('avg_reward') is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['avg_reward'])
        print(f"Best configuration: {best['config_name']}")
        print(f"  Weights: {best['weights']}")
        print(f"  Avg Reward: {best['avg_reward']:.2f}")
        print(f"  Success Rate: {best['success_rate']:.1f}%")
        print(f"  Iteration: {best['iteration']}")
        print(f"  Type: {best['config_type']}")
    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)


def refine_around_config(base_weights: List[float], base_name: str = "refined",
                         step_size: float = 0.1, num_variations: int = 20) -> List[Tuple[str, List[float]]]:
    """
    Generate fine-grained refinements around a specific weight configuration.
    
    Args:
        base_weights: Base weight configuration to refine around
        base_name: Base name for generated configurations
        step_size: Step size for weight adjustments (default: 0.1)
        num_variations: Number of variations to generate
    
    Returns:
        List of (config_name, weights_list) tuples
    """
    suggestions = []
    component_names = [
        "x_position", "y_position", "vx_velocity", "vy_velocity",
        "angle", "angular_velocity", "leg_contact_1", "leg_contact_2"
    ]
    
    # Strategy 1: Single-component fine adjustments
    for component_idx in range(8):
        base_weight = base_weights[component_idx]
        # Try small adjustments around base weight
        for adjustment in [-step_size, step_size, -2*step_size, 2*step_size]:
            new_weights = base_weights.copy()
            new_weight = max(0.1, min(3.0, base_weight + adjustment))
            new_weights[component_idx] = new_weight
            # Normalize to target sum before appending
            new_weights = normalize_weights(new_weights)
            
            config_name = f"{base_name}_{component_names[component_idx]}_{adjustment:+.2f}"
            suggestions.append((config_name, new_weights))
            
            if len(suggestions) >= num_variations:
                break
        if len(suggestions) >= num_variations:
            break
    
    # Strategy 2: Two-component combinations if we need more
    if len(suggestions) < num_variations:
        for i in range(8):
            for j in range(i+1, 8):
                if len(suggestions) >= num_variations:
                    break
                new_weights = base_weights.copy()
                # Small adjustments to both
                new_weights[i] = max(0.1, min(3.0, base_weights[i] + step_size))
                new_weights[j] = max(0.1, min(3.0, base_weights[j] + step_size))
                # Normalize to target sum before appending
                new_weights = normalize_weights(new_weights)
                config_name = f"{base_name}_{component_names[i]}_{component_names[j]}_+{step_size}"
                suggestions.append((config_name, new_weights))
            if len(suggestions) >= num_variations:
                break
    
    return suggestions[:num_variations]


def continue_refinement(csv_file: str = "weight_optimization_results.csv",
                       max_iterations: int = 3,
                       suggestions_per_iteration: int = 15,
                       fine_grained: bool = False):
    """
    Continue refinement from existing results.
    
    Args:
        csv_file: Path to existing CSV results file
        max_iterations: Number of additional iterations to run
        suggestions_per_iteration: Number of suggestions per iteration
        fine_grained: If True, use fine-grained refinement around best config
    """
    print("=" * 60)
    print("Continuing Weight Optimization Refinement")
    print("=" * 60)
    print(f"Reading results from: {csv_file}")
    print()
    
    # Read existing results to get tested configs
    tested_configs = set()
    all_results = []
    best_config = None
    best_reward = float('-inf')
    
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'weights' in row and row['weights']:
                    tested_configs.add(row['weights'])
                    all_results.append(row)
                    
                    # Track best config
                    if row.get('avg_reward') and row['avg_reward'] != '':
                        try:
                            reward = float(row['avg_reward'])
                            if reward > best_reward:
                                best_reward = reward
                                best_config = row
                        except (ValueError, TypeError):
                            pass
    
    print(f"Found {len(all_results)} existing configurations")
    if best_config:
        print(f"Best so far: {best_config.get('config_name', 'unknown')}")
        print(f"  Avg Reward: {best_config.get('avg_reward', 'N/A')}")
        print(f"  Weights: {best_config.get('weights', 'N/A')}")
    print()
    
    # Determine starting iteration
    max_existing_iteration = 0
    for result in all_results:
        if 'iteration' in result and result['iteration']:
            try:
                iter_num = int(result['iteration'])
                max_existing_iteration = max(max_existing_iteration, iter_num)
            except (ValueError, TypeError):
                pass
    
    start_iteration = max_existing_iteration + 1
    end_iteration = start_iteration + max_iterations - 1
    
    print(f"Starting from iteration {start_iteration} (will run through iteration {end_iteration})")
    print()
    
    # Run refinement iterations
    for iteration in range(start_iteration, end_iteration + 1):
        print("=" * 60)
        print(f"ITERATION {iteration}: Refinement")
        print("=" * 60)
        print()
        
        # Analyze current results
        print("Analyzing results...")
        analysis = analyze_results(csv_file)
        
        if 'error' in analysis:
            print(f"  ✗ Analysis error: {analysis['error']}")
            print("  Skipping further iterations.")
            break
        
        print(f"  ✓ Analyzed {analysis['total_configs']} configurations")
        
        # Show best configs
        if analysis['best_configs']:
            print("\n  Top 3 configurations:")
            for idx, config in enumerate(analysis['best_configs'][:3], 1):
                print(f"    {idx}. {config.get('config_name', 'unknown')}: "
                      f"reward={config.get('avg_reward', 'N/A')}, "
                      f"success={config.get('success_rate', 'N/A')}%")
        
        # Generate suggestions
        print(f"\n  Generating {suggestions_per_iteration} suggestions...")
        
        if fine_grained and best_config and best_config.get('weights'):
            # Fine-grained refinement around best config
            try:
                base_weights = [float(w.strip()) for w in best_config['weights'].split(',')]
                suggestions = refine_around_config(
                    base_weights,
                    base_name=f"fine_iter{iteration}",
                    step_size=0.05,  # Smaller step for fine-grained
                    num_variations=suggestions_per_iteration
                )
                print(f"  ✓ Generated {len(suggestions)} fine-grained refinements")
            except (ValueError, AttributeError) as e:
                print(f"  ⚠ Could not parse best config weights, using analysis-based suggestions")
                suggestions = suggest_next_configs(
                    analysis,
                    num_suggestions=suggestions_per_iteration,
                    tested_configs=tested_configs
                )
        else:
            # Standard analysis-based suggestions
            suggestions = suggest_next_configs(
                analysis,
                num_suggestions=suggestions_per_iteration,
                tested_configs=tested_configs
            )
        
        if not suggestions:
            print("  No new suggestions generated. Stopping iterations.")
            break
        
        print(f"  ✓ Generated {len(suggestions)} suggestions")
        print()
        
        # Run suggested configurations
        iteration_results = []
        for idx, (config_name, weights) in enumerate(suggestions, 1):
            print(f"[{idx}/{len(suggestions)}] Testing: {config_name}")
            print(f"  Weights: {weights}")
            
            try:
                result = run_weight_test(
                    weights,
                    config_name,
                    iteration=iteration,
                    config_type="suggested",
                    suggestion_basis=f"iteration_{iteration}_refinement"
                )
                iteration_results.append(result)
                all_results.append(result)
                tested_configs.add(','.join(map(str, weights)))
                
                print(f"  ✓ Completed - Avg Reward: {result['avg_reward']:.2f}, "
                      f"Success Rate: {result['success_rate']:.1f}%")
                
                # Update best if better
                if result['avg_reward'] > best_reward:
                    best_reward = result['avg_reward']
                    best_config = result
            except Exception as e:
                print(f"  ✗ Error: {e}")
                error_result = {
                    'config_name': config_name,
                    'weights': ','.join(map(str, weights)),
                    'iteration': iteration,
                    'config_type': 'suggested',
                    'suggestion_basis': f"iteration_{iteration}_refinement",
                    'avg_reward': None,
                    'success_rate': None,
                    'death_rate': None,
                    'best_reward': None,
                    'rolling_avg_50': None,
                    'rolling_avg_100': None,
                    'total_engrams': None,
                    'avg_engram_distance': None,
                    'error': str(e)
                }
                iteration_results.append(error_result)
                all_results.append(error_result)
            print()
        
        # Append results to CSV
        _save_results(iteration_results, csv_file, append=True)
        print(f"Iteration {iteration} results appended to: {csv_file}")
        
        # Show best so far
        if best_config:
            print(f"\n  Best configuration so far: {best_config.get('config_name', 'unknown')}")
            print(f"    Avg Reward: {best_config.get('avg_reward', 'N/A')}, "
                  f"Success Rate: {best_config.get('success_rate', 'N/A')}%")
        print()
    
    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if best_config:
        print(f"Best configuration: {best_config.get('config_name', 'unknown')}")
        print(f"  Weights: {best_config.get('weights', 'N/A')}")
        print(f"  Avg Reward: {best_config.get('avg_reward', 'N/A')}")
        print(f"  Success Rate: {best_config.get('success_rate', 'N/A')}%")
    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Results saved to: {csv_file}")
    print("=" * 60)


def _save_results(results: List[Dict[str, Any]], output_file: str, append: bool = False):
    """
    Helper function to save results to CSV.
    
    Args:
        results: List of result dictionaries
        output_file: Path to CSV file
        append: If True, append to existing file; if False, overwrite
    """
    if not results:
        return
    
    fieldnames = [
        'config_name',
        'weights',
        'iteration',
        'config_type',
        'avg_reward',
        'success_rate',
        'death_rate',
        'best_reward',
        'rolling_avg_50',
        'rolling_avg_100',
        'total_engrams',
        'avg_engram_distance'
    ]
    
    # Add optional fields if present
    if any('suggestion_basis' in r for r in results):
        fieldnames.append('suggestion_basis')
    if any('error' in r for r in results):
        fieldnames.append('error')
    
    write_mode = 'a' if append else 'w'
    write_header = not append or not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    
    with open(output_file, write_mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--iterative":
        max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        suggestions = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        run_optimization_iterative(max_iterations=max_iter, suggestions_per_iteration=suggestions)
    elif len(sys.argv) > 1 and sys.argv[1] == "--continue":
        max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        suggestions = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        fine_grained = "--fine" in sys.argv
        csv_file = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4].endswith('.csv') else "weight_optimization_results.csv"
        continue_refinement(csv_file, max_iterations=max_iter, suggestions_per_iteration=suggestions, fine_grained=fine_grained)
    else:
        run_optimization()
