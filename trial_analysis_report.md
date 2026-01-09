# Trial Results Analysis Report

Comprehensive comparison of training results from panicky, panicky2, and panicky3.

## Executive Summary

**Best Average Reward**: panicky (-141.33)
**Highest Success Rate**: 0.1%

## Settings Comparison

| Setting | panicky | panicky2 | panicky3 |
|---------|---------|---------|---------|
|MIN_RESULTS|100|300|100|
|HIT_POINTS|1000|1000|400|
|PANIC_ENABLED|True|True|True|
|PANIC_MAX_NOISE|1|1|1|
|METABOLIC_COST|0.2|0.2|0.2|
|DECAY_ENABLED|True|True|True|
|DECAY_FUNCTION|gauss|gauss|gauss|
|TRIAL_SUCCESS_MULTIPLIER_SCALE|0.5|0.5|0.5|
|DECAY_OFFSET_IDS|40000|40000|40000|
|DECAY_SCALE_IDS|200000|200000|200000|
|DECAY_VALUE|0.9|0.9|0.9|
|DISPLAY|False|False|False|
|DROP_COLLECTION|True|True|True|
|MAX_TRIAL_LENGTH|400|400|400|
|NOISE|0.01|0.01|0.1|
|OUTPUT_VECTOR_SIZE|4|4|4|
|PROBABILISTIC_CHOICE|True|True|True|
|READ_ONLY|False|False|False|
|SHOW_ACTION_OUTPUT|True|True|True|
|STATE_VECTOR_SIZE|9|9|9|
|USE_HIT_POINTS|True|True|True|

## Performance Metrics

| Metric | panicky | panicky2 | panicky3 |
|--------|--------|--------|--------|
|Total Trials|1500|1300|1100|
|Average Reward|-141.33|-144.39|-144.69|
|Success Rate (%)|0.0|0.0|0.1|
|Best Episode Reward|127.26|175.07|203.27|
|Rolling Avg (50)|-144.67|-151.84|-140.20|
|Rolling Avg (100)|-122.92|-130.18|-133.02|
|Mean Episode Length|161.4|163.0|164.5|
|Total Engrams|242120|211953|180903|
|Avg Engram Distance|0.0142|0.0189|0.0214|

### Outcome Statistics

| Statistic | panicky | panicky2 | panicky3 |
|-----------|-----------|-----------|-----------|
|Positive Ratio|0.489|0.558|0.522|
|Negative Ratio|0.511|0.442|0.478|
|Mean Outcome|-0.0151|-4.0813|-0.0954|

## Learning Analysis

### panicky

- **Early Average Reward**: -151.37
- **Late Average Reward**: -131.29
- **Overall Improvement**: 20.08
- **Final Trend**: 50.47

### panicky2

- **Early Average Reward**: -153.97
- **Late Average Reward**: -134.81
- **Overall Improvement**: 19.15
- **Final Trend**: 41.55

### panicky3

- **Early Average Reward**: -151.42
- **Late Average Reward**: -137.96
- **Overall Improvement**: 13.46
- **Final Trend**: 17.33

## Key Findings

### MIN_RESULTS Impact

Higher MIN_RESULTS values retrieve more similar engrams, potentially improving decision quality but requiring more memory.

- **MIN_RESULTS=100** (panicky): Avg Reward=-141.33, Total Engrams=242120, Avg Distance=0.0142
- **MIN_RESULTS=100** (panicky3): Avg Reward=-144.69, Total Engrams=180903, Avg Distance=0.0214
- **MIN_RESULTS=300** (panicky2): Avg Reward=-144.39, Total Engrams=211953, Avg Distance=0.0189

### HIT_POINTS Impact

HIT_POINTS determines how long trials can run before termination, affecting exploration time.

- **HIT_POINTS=400** (panicky3): Avg Reward=-144.69, Mean Length=164.5, Deaths=3
- **HIT_POINTS=1000** (panicky): Avg Reward=-141.33, Mean Length=161.4, Deaths=0
- **HIT_POINTS=1000** (panicky2): Avg Reward=-144.39, Mean Length=163.0, Deaths=0

### Memory Efficiency Analysis

- **panicky**: 161.4 engrams/trial, avg distance=0.0142
- **panicky2**: 163.0 engrams/trial, avg distance=0.0189
- **panicky3**: 164.5 engrams/trial, avg distance=0.0214

## Recommendations

Based on the analysis, **panicky** showed the best performance with:
- MIN_RESULTS: 100
- HIT_POINTS: 1000

### Suggested Next Steps

1. Investigate the relationship between MIN_RESULTS and memory efficiency
2. Test intermediate HIT_POINTS values to find optimal balance
3. Analyze engram distance trends to understand memory quality
4. Consider longer training runs to observe convergence patterns
