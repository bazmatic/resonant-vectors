"""
Analysis script for comparing trial results from trial1, panicky, panicky2, and panicky3.
Extracts settings, metrics, and generates a comprehensive comparison report.
"""

import json
import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_settings_file(settings_path: str) -> Dict[str, Any]:
    """Parse a settings.txt file and extract key-value pairs."""
    settings = {}
    
    if not os.path.exists(settings_path):
        return settings
    
    with open(settings_path, 'r') as f:
        content = f.read()
    
    # Skip lines that are just separators or headers
    lines = content.split('\n')
    for line in lines:
        # Skip separator lines and header lines
        if '=' not in line or line.strip().startswith('=') or 'Training Settings' in line:
            continue
        
        # Extract key-value pairs using regex
        pattern = r'(\w+)\s*=\s*([^\n]+)'
        match = re.search(pattern, line)
        if match:
            key, value = match.groups()
            value = value.strip()
            # Skip if key looks like a header or separator
            if key.lower() in ['lander3', 'training', 'settings']:
                continue
            # Try to convert to appropriate type
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


def load_metrics_json(json_path: str) -> Optional[Dict[str, Any]]:
    """Load metrics from a JSON file."""
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def find_trial_json(trial_name: str, root_dir: str) -> Optional[str]:
    """Search for JSON metrics file that might correspond to a trial."""
    # Check common locations
    possible_paths = [
        os.path.join(root_dir, f"{trial_name}_metrics.json"),
        os.path.join(root_dir, f"{trial_name}", "training_metrics.json"),
        os.path.join(root_dir, "training_metrics.json"),  # Main file
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def match_json_to_trial(json_data: Dict[str, Any], trial_settings: Dict[str, Any]) -> bool:
    """Check if a JSON file's settings match a trial's settings."""
    if 'settings' not in json_data:
        return False
    
    json_settings = json_data['settings']
    
    # Compare key distinguishing settings
    key_settings = ['MIN_RESULTS', 'HIT_POINTS']
    
    for key in key_settings:
        if key in trial_settings and key in json_settings:
            if trial_settings[key] != json_settings[key]:
                return False
    
    return True


def extract_trial_data(trial_name: str, root_dir: str) -> Dict[str, Any]:
    """Extract all available data for a trial."""
    trial_dir = os.path.join(root_dir, trial_name)
    settings_path = os.path.join(trial_dir, "settings.txt")
    
    data = {
        'trial_name': trial_name,
        'settings': parse_settings_file(settings_path),
        'metrics': None,
        'summary_stats': None,
        'outcome_stats': None,
        'has_json': False
    }
    
    # If settings are empty, try to extract from JSON files
    has_settings = len(data['settings']) > 0 and 'MIN_RESULTS' not in str(data['settings'].values())
    
    # Try to find matching JSON file
    # First, check the main training_metrics.json
    main_json_path = os.path.join(root_dir, "training_metrics.json")
    if os.path.exists(main_json_path):
        json_data = load_metrics_json(main_json_path)
        if json_data:
            # If we have settings, match them; otherwise use this JSON if it's the first match
            if has_settings:
                if match_json_to_trial(json_data, data['settings']):
                    data['metrics'] = json_data.get('metrics', {})
                    data['summary_stats'] = json_data.get('summary_stats', {})
                    data['outcome_stats'] = json_data.get('outcome_stats', {})
                    data['has_json'] = True
                    data['json_source'] = main_json_path
                    # Extract settings from JSON if we don't have them
                    if not data['settings'] and 'settings' in json_data:
                        data['settings'] = json_data['settings']
                    return data
            else:
                # For trial1 which has no settings, use this if it hasn't been used yet
                # We'll check if this JSON matches other trials first
                data['metrics'] = json_data.get('metrics', {})
                data['summary_stats'] = json_data.get('summary_stats', {})
                data['outcome_stats'] = json_data.get('outcome_stats', {})
                data['has_json'] = True
                data['json_source'] = main_json_path
                # Extract settings from JSON
                if 'settings' in json_data:
                    data['settings'] = json_data['settings']
                return data
    
    # Check backup files
    backup_pattern = os.path.join(root_dir, "training_metrics_backup_trial_*.json")
    import glob
    backup_files = sorted(glob.glob(backup_pattern), key=lambda x: int(re.search(r'trial_(\d+)', x).group(1)) if re.search(r'trial_(\d+)', x) else 0)
    
    for backup_path in backup_files:
        json_data = load_metrics_json(backup_path)
        if json_data:
            if has_settings:
                if match_json_to_trial(json_data, data['settings']):
                    data['metrics'] = json_data.get('metrics', {})
                    data['summary_stats'] = json_data.get('summary_stats', {})
                    data['outcome_stats'] = json_data.get('outcome_stats', {})
                    data['has_json'] = True
                    data['json_source'] = backup_path
                    return data
            else:
                # For trials without settings, try to match by checking if settings in JSON match expected pattern
                # This is a fallback - we'll use the JSON if it seems reasonable
                pass
    
    return data


def calculate_learning_trends(rewards: List[float], window_size: int = 100) -> Dict[str, float]:
    """Calculate learning trends from reward data."""
    if not rewards or len(rewards) < window_size:
        return {}
    
    # Split into early and late periods
    mid_point = len(rewards) // 2
    early_rewards = rewards[:mid_point]
    late_rewards = rewards[mid_point:]
    
    early_avg = np.mean(early_rewards) if early_rewards else 0
    late_avg = np.mean(late_rewards) if late_rewards else 0
    
    # Calculate rolling averages
    rolling_early = []
    rolling_late = []
    
    for i in range(window_size, len(early_rewards)):
        rolling_early.append(np.mean(early_rewards[i-window_size:i]))
    
    for i in range(window_size, len(late_rewards)):
        rolling_late.append(np.mean(late_rewards[i-window_size:i]))
    
    improvement = late_avg - early_avg
    final_trend = np.mean(rolling_late[-window_size:]) - np.mean(rolling_early[:window_size]) if rolling_early and rolling_late else 0
    
    return {
        'early_average': early_avg,
        'late_average': late_avg,
        'improvement': improvement,
        'final_trend': final_trend
    }


def generate_report(trials_data: List[Dict[str, Any]], output_path: str):
    """Generate a comprehensive markdown report."""
    
    report_lines = []
    
    # Title
    report_lines.append("# Trial Results Analysis Report\n")
    report_lines.append("Comprehensive comparison of training results from panicky, panicky2, and panicky3.\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n")
    
    # Find best performing trial
    best_trial = None
    best_reward = float('-inf')
    best_success = -1
    
    for trial in trials_data:
        if trial['summary_stats']:
            reward = trial['summary_stats'].get('overall_average_reward', float('-inf'))
            success = trial['summary_stats'].get('success_rate', -1)
            if reward > best_reward:
                best_reward = reward
                best_trial = trial['trial_name']
            if success > best_success:
                best_success = success
    
    report_lines.append(f"**Best Average Reward**: {best_trial} ({best_reward:.2f})" if best_trial else "**Best Average Reward**: N/A")
    report_lines.append(f"**Highest Success Rate**: {best_success:.1f}%\n" if best_success >= 0 else "**Highest Success Rate**: N/A\n")
    
    # Settings Comparison
    report_lines.append("## Settings Comparison\n")
    trial_names_header = "| Setting | " + " | ".join([t['trial_name'] for t in trials_data]) + " |"
    report_lines.append(trial_names_header)
    report_lines.append("|" + "|".join(["---------"] * (len(trials_data) + 1)) + "|")
    
    # Get all unique setting keys
    all_settings = set()
    for trial in trials_data:
        all_settings.update(trial['settings'].keys())
    
    # Key settings to highlight
    key_settings = ['MIN_RESULTS', 'HIT_POINTS', 'PANIC_ENABLED', 'PANIC_MAX_NOISE', 
                    'METABOLIC_COST', 'DECAY_ENABLED', 'DECAY_FUNCTION', 'TRIAL_SUCCESS_MULTIPLIER_SCALE']
    
    # Add key settings first
    for key in key_settings:
        if key in all_settings:
            row = [key]
            for trial in trials_data:
                value = trial['settings'].get(key, 'N/A')
                if isinstance(value, bool):
                    value = 'True' if value else 'False'
                row.append(str(value))
            report_lines.append("|" + "|".join(row) + "|")
            all_settings.discard(key)
    
    # Add remaining settings
    for key in sorted(all_settings):
        row = [key]
        for trial in trials_data:
            value = trial['settings'].get(key, 'N/A')
            if isinstance(value, bool):
                value = 'True' if value else 'False'
            row.append(str(value))
        report_lines.append("|" + "|".join(row) + "|")
    
    report_lines.append("")
    
    # Performance Metrics
    report_lines.append("## Performance Metrics\n")
    metrics_header = "| Metric | " + " | ".join([t['trial_name'] for t in trials_data]) + " |"
    report_lines.append(metrics_header)
    report_lines.append("|" + "|".join(["--------"] * (len(trials_data) + 1)) + "|")
    
    metrics_to_compare = [
        ('Total Trials', 'total_trials'),
        ('Average Reward', 'overall_average_reward', 2),
        ('Success Rate (%)', 'success_rate', 1),
        ('Best Episode Reward', 'best_episode_reward', 2),
        ('Rolling Avg (50)', 'current_rolling_average_50', 2),
        ('Rolling Avg (100)', 'current_rolling_average_100', 2),
        ('Mean Episode Length', 'mean_episode_length', 1),
        ('Total Engrams', 'total_engrams', 0),
        ('Avg Engram Distance', 'average_engram_distance', 4),
    ]
    
    for metric_info in metrics_to_compare:
        metric_name = metric_info[0]
        metric_key = metric_info[1]
        precision = metric_info[2] if len(metric_info) > 2 else 0
        
        row = [metric_name]
        for trial in trials_data:
            if trial['summary_stats'] and metric_key in trial['summary_stats']:
                value = trial['summary_stats'][metric_key]
                if precision == 0:
                    row.append(str(int(value)))
                else:
                    row.append(f"{value:.{precision}f}")
            else:
                row.append("N/A")
        report_lines.append("|" + "|".join(row) + "|")
    
    report_lines.append("")
    
    # Outcome Statistics
    report_lines.append("### Outcome Statistics\n")
    outcome_header = "| Statistic | " + " | ".join([t['trial_name'] for t in trials_data]) + " |"
    report_lines.append(outcome_header)
    report_lines.append("|" + "|".join(["-----------"] * (len(trials_data) + 1)) + "|")
    
    outcome_metrics = [
        ('Positive Ratio', 'positive_ratio', 3),
        ('Negative Ratio', 'negative_ratio', 3),
        ('Mean Outcome', 'mean_outcome', 4),
    ]
    
    for metric_info in outcome_metrics:
        metric_name = metric_info[0]
        metric_key = metric_info[1]
        precision = metric_info[2]
        
        row = [metric_name]
        for trial in trials_data:
            if trial['outcome_stats'] and metric_key in trial['outcome_stats']:
                value = trial['outcome_stats'][metric_key]
                row.append(f"{value:.{precision}f}")
            else:
                row.append("N/A")
        report_lines.append("|" + "|".join(row) + "|")
    
    report_lines.append("")
    
    # Learning Analysis
    report_lines.append("## Learning Analysis\n")
    
    for trial in trials_data:
        if trial['metrics'] and 'rewards' in trial['metrics']:
            rewards = trial['metrics']['rewards']
            trends = calculate_learning_trends(rewards)
            
            report_lines.append(f"### {trial['trial_name']}\n")
            if trends:
                report_lines.append(f"- **Early Average Reward**: {trends['early_average']:.2f}")
                report_lines.append(f"- **Late Average Reward**: {trends['late_average']:.2f}")
                report_lines.append(f"- **Overall Improvement**: {trends['improvement']:.2f}")
                report_lines.append(f"- **Final Trend**: {trends['final_trend']:.2f}\n")
            else:
                report_lines.append("- Insufficient data for trend analysis\n")
    
    # Key Findings
    report_lines.append("## Key Findings\n")
    
    # Analyze MIN_RESULTS impact
    min_results_trials = {}
    for trial in trials_data:
        min_res = trial['settings'].get('MIN_RESULTS')
        if min_res is not None and trial['summary_stats']:
            if min_res not in min_results_trials:
                min_results_trials[min_res] = []
            min_results_trials[min_res].append(trial)
    
    if len(min_results_trials) > 1:
        report_lines.append("### MIN_RESULTS Impact\n")
        report_lines.append("Higher MIN_RESULTS values retrieve more similar engrams, potentially improving decision quality but requiring more memory.\n")
        for min_res in sorted(min_results_trials.keys()):
            trials_with_min_res = min_results_trials[min_res]
            for trial in trials_with_min_res:
                if trial['summary_stats']:
                    reward = trial['summary_stats'].get('overall_average_reward', 0)
                    engrams = trial['summary_stats'].get('total_engrams', 0)
                    distance = trial['summary_stats'].get('average_engram_distance', 0)
                    report_lines.append(f"- **MIN_RESULTS={min_res}** ({trial['trial_name']}): Avg Reward={reward:.2f}, Total Engrams={engrams}, Avg Distance={distance:.4f}")
        report_lines.append("")
    
    # Analyze HIT_POINTS impact
    hit_points_trials = {}
    for trial in trials_data:
        hp = trial['settings'].get('HIT_POINTS')
        if hp is not None and trial['summary_stats']:
            if hp not in hit_points_trials:
                hit_points_trials[hp] = []
            hit_points_trials[hp].append(trial)
    
    if len(hit_points_trials) > 1:
        report_lines.append("### HIT_POINTS Impact\n")
        report_lines.append("HIT_POINTS determines how long trials can run before termination, affecting exploration time.\n")
        for hp in sorted(hit_points_trials.keys()):
            trials_with_hp = hit_points_trials[hp]
            for trial in trials_with_hp:
                if trial['summary_stats']:
                    reward = trial['summary_stats'].get('overall_average_reward', 0)
                    length = trial['summary_stats'].get('mean_episode_length', 0)
                    deaths = trial['summary_stats'].get('death_count', 0)
                    report_lines.append(f"- **HIT_POINTS={hp}** ({trial['trial_name']}): Avg Reward={reward:.2f}, Mean Length={length:.1f}, Deaths={deaths}")
        report_lines.append("")
    
    # Memory efficiency analysis
    report_lines.append("### Memory Efficiency Analysis\n")
    for trial in trials_data:
        if trial['summary_stats']:
            total_engrams = trial['summary_stats'].get('total_engrams', 0)
            total_trials = trial['summary_stats'].get('total_trials', 0)
            if total_trials > 0:
                engrams_per_trial = total_engrams / total_trials
                avg_distance = trial['summary_stats'].get('average_engram_distance', 0)
                report_lines.append(f"- **{trial['trial_name']}**: {engrams_per_trial:.1f} engrams/trial, avg distance={avg_distance:.4f}")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations\n")
    
    # Find best settings combination
    if best_trial:
        best_trial_data = next(t for t in trials_data if t['trial_name'] == best_trial)
        report_lines.append(f"Based on the analysis, **{best_trial}** showed the best performance with:")
        report_lines.append(f"- MIN_RESULTS: {best_trial_data['settings'].get('MIN_RESULTS', 'N/A')}")
        report_lines.append(f"- HIT_POINTS: {best_trial_data['settings'].get('HIT_POINTS', 'N/A')}")
        report_lines.append("")
    
    report_lines.append("### Suggested Next Steps\n")
    report_lines.append("1. Investigate the relationship between MIN_RESULTS and memory efficiency")
    report_lines.append("2. Test intermediate HIT_POINTS values to find optimal balance")
    report_lines.append("3. Analyze engram distance trends to understand memory quality")
    report_lines.append("4. Consider longer training runs to observe convergence patterns")
    report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report generated: {output_path}")


def create_comparison_charts(trials_data: List[Dict[str, Any]], output_dir: str):
    """Create comparison visualization charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Learning curves comparison
    plt.figure(figsize=(12, 6))
    for trial in trials_data:
        if trial['metrics'] and 'rewards' in trial['metrics']:
            rewards = trial['metrics']['rewards']
            trials = list(range(1, len(rewards) + 1))
            
            # Calculate rolling average
            window_size = 50
            rolling_avg = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                window = rewards[start_idx:i+1]
                rolling_avg.append(np.mean(window))
            
            plt.plot(trials, rolling_avg, label=trial['trial_name'], linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Reward (50 episodes)')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'learning_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics comparison bar chart
    metrics_to_plot = ['overall_average_reward', 'success_rate', 'best_episode_reward']
    metric_labels = ['Average Reward', 'Success Rate (%)', 'Best Episode Reward']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    
    for idx, (metric_key, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        trial_names = []
        values = []
        
        for trial in trials_data:
            if trial['summary_stats'] and metric_key in trial['summary_stats']:
                trial_names.append(trial['trial_name'])
                values.append(trial['summary_stats'][metric_key])
        
        if values:
            axes[idx].bar(trial_names, values)
            axes[idx].set_title(metric_label)
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to {output_dir}")


def main():
    """Main analysis function."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    trial_names = ['panicky', 'panicky2', 'panicky3']  # Exclude trial1 - no settings available
    
    print("Collecting trial data...")
    trials_data = []
    
    for trial_name in trial_names:
        print(f"  Processing {trial_name}...")
        data = extract_trial_data(trial_name, root_dir)
        trials_data.append(data)
        if data['has_json']:
            print(f"    Found JSON data: {data.get('json_source', 'unknown')}")
        else:
            print(f"    No matching JSON data found")
    
    print("\nGenerating report...")
    report_path = os.path.join(root_dir, 'trial_analysis_report.md')
    generate_report(trials_data, report_path)
    
    # Check if we have enough data for charts
    has_metrics = any(trial['metrics'] for trial in trials_data)
    if has_metrics:
        print("\nGenerating comparison charts...")
        charts_dir = os.path.join(root_dir, 'trial_comparison_charts')
        create_comparison_charts(trials_data, charts_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
