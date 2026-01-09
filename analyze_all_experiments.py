#!/usr/bin/env python3
"""
Analyze all experiments and extract key metrics for recommendations.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

def parse_settings_file(settings_path: str) -> Dict[str, Any]:
    """Parse a settings.txt file and extract key-value pairs."""
    settings = {}
    
    if not os.path.exists(settings_path):
        return settings
    
    with open(settings_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for line in lines:
        if '=' not in line or line.strip().startswith('=') or 'Training Settings' in line:
            continue
        
        pattern = r'(\w+)\s*=\s*([^\n]+)'
        match = re.search(pattern, line)
        if match:
            key, value = match.groups()
            value = value.strip()
            if key.lower() in ['lander3', 'training', 'settings']:
                continue
            if value.lower() == 'true':
                settings[key] = True
            elif value.lower() == 'false':
                settings[key] = False
            elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                if '.' in value:
                    settings[key] = float(value)
                else:
                    settings[key] = int(value)
            else:
                settings[key] = value
    
    return settings

def extract_summary_from_metrics(json_path: str) -> Optional[Dict[str, Any]]:
    """Extract summary statistics from training_metrics.json without loading the whole file."""
    if not os.path.exists(json_path):
        return None
    
    try:
        # Try to read just the summary_stats if it's structured that way
        # Otherwise, calculate from the data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        summary = {}
        
        # Check if it's a list of trial data or a dict with summary_stats
        if isinstance(data, dict):
            if 'summary_stats' in data:
                summary = data['summary_stats'].copy()
            elif 'metrics' in data:
                # Calculate summary from metrics
                metrics = data['metrics']
                if 'rewards' in metrics and len(metrics['rewards']) > 0:
                    rewards = metrics['rewards']
                    summary['total_trials'] = len(rewards)
                    summary['overall_average_reward'] = sum(rewards) / len(rewards)
                    summary['best_episode_reward'] = max(rewards)
                    summary['worst_episode_reward'] = min(rewards)
                    
                    # Calculate rolling averages
                    if len(rewards) >= 50:
                        summary['current_rolling_average_50'] = sum(rewards[-50:]) / 50
                    if len(rewards) >= 100:
                        summary['current_rolling_average_100'] = sum(rewards[-100:]) / 100
                    
                    # Success rate (assuming reward > 200 indicates success for LunarLander)
                    successes = sum(1 for r in rewards if r > 200)
                    summary['success_rate'] = (successes / len(rewards)) * 100
        elif isinstance(data, list) and len(data) > 0:
            # List of trial dictionaries
            rewards = [trial.get('reward', 0) for trial in data if 'reward' in trial]
            if rewards:
                summary['total_trials'] = len(rewards)
                summary['overall_average_reward'] = sum(rewards) / len(rewards)
                summary['best_episode_reward'] = max(rewards)
                summary['worst_episode_reward'] = min(rewards)
                
                if len(rewards) >= 50:
                    summary['current_rolling_average_50'] = sum(rewards[-50:]) / 50
                if len(rewards) >= 100:
                    summary['current_rolling_average_100'] = sum(rewards[-100:]) / 100
                
                successes = sum(1 for r in rewards if r > 200)
                summary['success_rate'] = (successes / len(rewards)) * 100
                
                # Try to get other metrics from last trial
                if 'total_engrams' in data[-1]:
                    summary['total_engrams'] = data[-1]['total_engrams']
                if 'engram_distance' in data[-1]:
                    summary['average_engram_distance'] = data[-1].get('engram_distance', 0)
        
        return summary if summary else None
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def analyze_all_experiments(experiments_dir: str) -> List[Dict[str, Any]]:
    """Analyze all experiments and return structured data."""
    experiments = []
    
    exp_dirs = sorted([d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))])
    
    for exp_name in exp_dirs:
        exp_dir = os.path.join(experiments_dir, exp_name)
        settings_path = os.path.join(exp_dir, 'settings.txt')
        metrics_path = os.path.join(exp_dir, 'training_metrics.json')
        
        settings = parse_settings_file(settings_path)
        summary = extract_summary_from_metrics(metrics_path)
        
        experiments.append({
            'name': exp_name,
            'settings': settings,
            'summary': summary,
            'has_metrics': summary is not None
        })
    
    return experiments

def generate_recommendations(experiments: List[Dict[str, Any]]) -> str:
    """Generate markdown recommendations based on experiment analysis."""
    
    # Filter experiments with metrics
    valid_experiments = [e for e in experiments if e['has_metrics'] and e['summary']]
    
    if not valid_experiments:
        return "# Experiment Analysis\n\nNo experiments with valid metrics found.\n"
    
    # Sort by performance
    valid_experiments.sort(key=lambda x: x['summary'].get('overall_average_reward', float('-inf')), reverse=True)
    
    lines = []
    lines.append("# Ideal Settings Recommendations\n")
    lines.append("Based on comprehensive analysis of all experiments.\n")
    lines.append(f"**Total Experiments Analyzed**: {len(valid_experiments)}\n")
    
    # Top Performers
    lines.append("## Top Performing Experiments\n")
    lines.append("| Rank | Experiment | Avg Reward | Success Rate | Trials | Key Settings |")
    lines.append("|------|------------|------------|--------------|--------|--------------|")
    
    for i, exp in enumerate(valid_experiments[:10], 1):
        s = exp['summary']
        settings = exp['settings']
        key_settings = []
        if 'MIN_RESULTS' in settings:
            key_settings.append(f"MIN_RESULTS={settings['MIN_RESULTS']}")
        if 'TRIAL_SUCCESS_MULTIPLIER_SCALE' in settings:
            key_settings.append(f"MULT={settings['TRIAL_SUCCESS_MULTIPLIER_SCALE']}")
        if 'VECTOR_COMPONENT_WEIGHTS' in settings:
            key_settings.append("CUSTOM_WEIGHTS")
        
        key_str = ", ".join(key_settings[:2]) if key_settings else "default"
        
        lines.append(f"| {i} | {exp['name']} | {s.get('overall_average_reward', 0):.2f} | "
                    f"{s.get('success_rate', 0):.1f}% | {s.get('total_trials', 0)} | {key_str} |")
    
    lines.append("")
    
    # Analyze settings impact
    lines.append("## Settings Impact Analysis\n")
    
    # MIN_RESULTS analysis
    min_results_data = defaultdict(list)
    for exp in valid_experiments:
        if 'MIN_RESULTS' in exp['settings']:
            mr = exp['settings']['MIN_RESULTS']
            min_results_data[mr].append(exp['summary'].get('overall_average_reward', 0))
    
    if min_results_data:
        lines.append("### MIN_RESULTS Impact\n")
        lines.append("| MIN_RESULTS | Experiments | Avg Reward | Success Rate |")
        lines.append("|-------------|-------------|------------|--------------|")
        for mr in sorted(min_results_data.keys()):
            exps_with_mr = [e for e in valid_experiments if e['settings'].get('MIN_RESULTS') == mr]
            if exps_with_mr:
                avg_reward = sum(e['summary'].get('overall_average_reward', 0) for e in exps_with_mr) / len(exps_with_mr)
                avg_success = sum(e['summary'].get('success_rate', 0) for e in exps_with_mr) / len(exps_with_mr)
                lines.append(f"| {mr} | {len(exps_with_mr)} | {avg_reward:.2f} | {avg_success:.1f}% |")
        lines.append("")
    
    # TRIAL_SUCCESS_MULTIPLIER_SCALE analysis
    multiplier_data = defaultdict(list)
    for exp in valid_experiments:
        mult = exp['settings'].get('TRIAL_SUCCESS_MULTIPLIER_SCALE', 0)
        multiplier_data[mult].append(exp['summary'].get('overall_average_reward', 0))
    
    if multiplier_data:
        lines.append("### TRIAL_SUCCESS_MULTIPLIER_SCALE Impact\n")
        lines.append("| Multiplier Scale | Experiments | Avg Reward | Success Rate |")
        lines.append("|------------------|-------------|------------|--------------|")
        for mult in sorted(multiplier_data.keys()):
            exps_with_mult = [e for e in valid_experiments if abs(e['settings'].get('TRIAL_SUCCESS_MULTIPLIER_SCALE', 0) - mult) < 0.01]
            if exps_with_mult:
                avg_reward = sum(e['summary'].get('overall_average_reward', 0) for e in exps_with_mult) / len(exps_with_mult)
                avg_success = sum(e['summary'].get('success_rate', 0) for e in exps_with_mult) / len(exps_with_mult)
                lines.append(f"| {mult} | {len(exps_with_mult)} | {avg_reward:.2f} | {avg_success:.1f}% |")
        lines.append("")
    
    # Best performing experiment details
    best_exp = valid_experiments[0]
    lines.append("## Best Performing Experiment Details\n")
    lines.append(f"**Experiment**: {best_exp['name']}\n")
    lines.append(f"- **Average Reward**: {best_exp['summary'].get('overall_average_reward', 0):.2f}")
    lines.append(f"- **Success Rate**: {best_exp['summary'].get('success_rate', 0):.1f}%")
    lines.append(f"- **Best Episode Reward**: {best_exp['summary'].get('best_episode_reward', 0):.2f}")
    lines.append(f"- **Total Trials**: {best_exp['summary'].get('total_trials', 0)}")
    if 'current_rolling_average_100' in best_exp['summary']:
        lines.append(f"- **Final 100-Episode Average**: {best_exp['summary']['current_rolling_average_100']:.2f}")
    lines.append("")
    
    lines.append("### Settings from Best Experiment\n")
    lines.append("```")
    for key, value in sorted(best_exp['settings'].items()):
        lines.append(f"{key} = {value}")
    lines.append("```\n")
    
    # Recommended settings
    lines.append("## Recommended Ideal Settings\n")
    lines.append("Based on analysis of top performers:\n")
    
    # Find most common best settings
    top_5 = valid_experiments[:5]
    
    # Get most common MIN_RESULTS
    min_results_counts = defaultdict(int)
    for exp in top_5:
        if 'MIN_RESULTS' in exp['settings']:
            min_results_counts[exp['settings']['MIN_RESULTS']] += 1
    best_min_results = max(min_results_counts.items(), key=lambda x: x[1])[0] if min_results_counts else 300
    
    # Get most common multiplier
    multiplier_counts = defaultdict(int)
    for exp in top_5:
        mult = exp['settings'].get('TRIAL_SUCCESS_MULTIPLIER_SCALE', 0.145)
        # Round to nearest 0.05
        mult_rounded = round(mult * 20) / 20
        multiplier_counts[mult_rounded] += 1
    best_multiplier = max(multiplier_counts.items(), key=lambda x: x[1])[0] if multiplier_counts else 0.145
    
    # Get settings from best experiment
    rec_settings = best_exp['settings'].copy()
    
    lines.append("```python")
    lines.append("# Recommended Core Settings")
    for key in ['MIN_RESULTS', 'HIT_POINTS', 'METABOLIC_COST', 'TRIAL_SUCCESS_MULTIPLIER_SCALE',
                'NOISE_START', 'NOISE_END', 'NOISE_DECAY_RATE', 'VECTOR_SAVE_RATE']:
        if key in rec_settings:
            lines.append(f"{key} = {rec_settings[key]}")
    lines.append("")
    lines.append("# Recommended Feature Settings")
    for key in ['PANIC_ENABLED', 'USE_HIT_POINTS', 'DECAY_ENABLED', 'PROBABILISTIC_CHOICE']:
        if key in rec_settings:
            lines.append(f"{key} = {rec_settings[key]}")
    if 'VECTOR_COMPONENT_WEIGHTS' in rec_settings:
        lines.append(f"VECTOR_COMPONENT_WEIGHTS = {rec_settings['VECTOR_COMPONENT_WEIGHTS']}")
    lines.append("```\n")
    
    # Additional recommendations
    lines.append("## Additional Recommendations\n")
    
    # Learning curve analysis
    top_reward_improvement = []
    for exp in valid_experiments[:10]:
        s = exp['summary']
        if 'current_rolling_average_100' in s and 'current_rolling_average_50' in s:
            improvement = s['current_rolling_average_100'] - s.get('overall_average_reward', 0)
            top_reward_improvement.append((exp['name'], improvement))
    
    if top_reward_improvement:
        lines.append("### Learning Progression\n")
        lines.append("Experiments showing best learning progression (improvement over time):")
        top_reward_improvement.sort(key=lambda x: x[1], reverse=True)
        for name, improvement in top_reward_improvement[:5]:
            lines.append(f"- **{name}**: {improvement:+.2f} improvement")
        lines.append("")
    
    # Memory efficiency
    lines.append("### Memory Efficiency\n")
    memory_data = []
    for exp in valid_experiments:
        s = exp['summary']
        if 'total_engrams' in s and s.get('total_trials', 0) > 0:
            engrams_per_trial = s['total_engrams'] / s['total_trials']
            memory_data.append((exp['name'], engrams_per_trial, s.get('overall_average_reward', 0)))
    
    if memory_data:
        memory_data.sort(key=lambda x: (x[2], -x[1]), reverse=True)  # Sort by reward, then by efficiency
        lines.append("Best balance of performance and memory efficiency:")
        for name, ept, reward in memory_data[:5]:
            lines.append(f"- **{name}**: {ept:.1f} engrams/trial, {reward:.2f} avg reward")
        lines.append("")
    
    lines.append("## Notes\n")
    lines.append("- These recommendations are based on observed performance across all experiments")
    lines.append("- Settings should be tuned based on specific performance goals")
    lines.append("- Consider running longer experiments to verify convergence")
    lines.append("- Vector component weights may need domain-specific tuning")
    
    return "\n".join(lines)

if __name__ == "__main__":
    experiments_dir = "/Users/barryearsman/projects/personal/resonant-vectors/experiments"
    print("Analyzing experiments...")
    experiments = analyze_all_experiments(experiments_dir)
    print(f"Found {len(experiments)} experiments, {sum(1 for e in experiments if e['has_metrics'])} with metrics")
    
    print("Generating recommendations...")
    recommendations = generate_recommendations(experiments)
    
    output_path = "/Users/barryearsman/projects/personal/resonant-vectors/ideal_settings_recommendations.md"
    with open(output_path, 'w') as f:
        f.write(recommendations)
    
    print(f"Recommendations saved to {output_path}")
