STATE_VECTOR_SIZE = 8
OUTPUT_VECTOR_SIZE = 4
NOISE = 0.05
MIN_RESULTS = 400
READ_ONLY = False
DROP_COLLECTION = False
USE_HIT_POINTS = True
HIT_POINTS = 500
MAX_TRIAL_LENGTH = 400
METABOLIC_COST = 0.2

# Panic feature settings
PANIC_ENABLED = False
PANIC_MAX_NOISE = 1

PROBABILISTIC_CHOICE = False
DISPLAY = False
SHOW_ACTION_OUTPUT = False

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
TRIAL_SUCCESS_MULTIPLIER_SCALE = 0  # Controls strength of trial success multiplier on vector scores

# Vector component weightings
# Weightings for each of the 8 input state vector components to control their relative importance
# in similarity search calculations. Higher weights make components contribute more to distance.
# Default is all 1.0 (equal weighting). Modify to experiment with component importance.
# Component mapping (LunarLander environment):
#   [0]: x position (horizontal position)
#   [1]: y position (vertical position)
#   [2]: vx (horizontal velocity)
#   [3]: vy (vertical velocity)
#   [4]: angle (orientation in radians)
#   [5]: angular velocity (rotation rate,)
#   [6]: leg contact 1 (boolean, set to 0)
#   [7]: leg contact 2 (boolean, set to 0)
#0.6, 1.4, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0
#0.5, 1.4, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0
# VECTOR_COMPONENT_WEIGHTS = [0.3, 1.4, 1, 1.2, 1.2, 1.0, 1.0, 1.0]  # 8 weights, one per input component
VECTOR_COMPONENT_WEIGHTS = [0.5, 1.4, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0]  # 8 weights, one per input component

# Vector sampling settings
# Fraction of vectors to randomly sample and save (0.0 to 1.0)
# 1.0 = save all vectors, 0.5 = save 50% randomly selected, 0.0 = save none
VECTOR_SAVE_RATE = 0.2

# Delete oldest records before insert
# If True, before inserting records via batch_apply_feedback, delete the same number
# of oldest records first (based on insertion_index). This maintains a roughly constant
# collection size over time.
DELETE_OLDEST_BEFORE_INSERT = False

# Training settings
TRIALS_PER_EXPERIMENT = 500  # Number of trials to run per training experiment
