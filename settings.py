STATE_VECTOR_SIZE = 9
OUTPUT_VECTOR_SIZE = 4
NOISE = 0.1
MIN_RESULTS = 300
READ_ONLY = False
DROP_COLLECTION = True
USE_HIT_POINTS = True
HIT_POINTS = 300
MAX_TRIAL_LENGTH = 400
METABOLIC_COST = 0.2

ORIENTATION_BONUS = 2
PROBABILISTIC_CHOICE = True
DISPLAY = True
SHOW_ACTION_OUTPUT = True

# Decay ranker settings for order-based ranking
DECAY_ENABLED = False
DECAY_FUNCTION = "gauss"  # Options: "gauss", "exp", "linear"
DECAY_OFFSET_IDS = 40000  # No-decay zone in ID units around origin, around 100 trials
DECAY_SCALE_IDS = 200000  # ID distance at which relevance drops to decay value
DECAY_VALUE = 0.9  # Score value at the scale distance

# Trial success multiplier settings
# When scoring resonating vectors, applies a multiplier based on the trial's success.
# Formula: multiplier = 1.0 + (trial_final_success * TRIAL_SUCCESS_MULTIPLIER_SCALE)
# - final_success ranges from -1.0 (worst) to 1.0 (best)
# - With scale=0.5: best trials multiply scores by 1.5, worst by 0.5
# - With scale=1.0: best trials multiply scores by 2.0, worst by 0.0
# - Set to 0.0 to disable trial-based scoring
TRIAL_SUCCESS_MULTIPLIER_SCALE = 0.5  # Controls strength of trial success multiplier on vector scores
