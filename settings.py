STATE_VECTOR_SIZE = 8
OUTPUT_VECTOR_SIZE = 4
MIN_RESULTS = 400

# Noise decay settings (noise decays over total episode, not individual trials)
NOISE_START = 0.4      # Initial noise at start of training
NOISE_END = 0.02       # Final noise level (asymptotic target)
NOISE_DECAY_RATE = 3.0 # Controls speed of exponential decay (higher = faster decay)
READ_ONLY = False
DROP_COLLECTION = False
USE_HIT_POINTS = True
HIT_POINTS = 500
MAX_TRIAL_LENGTH = 400
METABOLIC_COST = 0.2

# Panic feature settings
PANIC_ENABLED = False
PANIC_MAX_NOISE = 1

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

# Temporal discount factor for credit assignment
# Controls how much credit earlier actions get for trial outcome
# 1.0 = all steps get full credit (current behavior)
# 0.99 = actions 100 steps before end get ~37% credit
# 0.95 = actions 100 steps before end get ~0.6% credit
CREDIT_DISCOUNT_GAMMA = 0.995

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
VECTOR_SAVE_RATE = 0.25

# Delete before insert strategy
# Controls whether and how records are deleted before inserting new ones.
# No deletion occurs until SWITCH_TO_DELETE_BEFORE_INSERT_THRESHOLD is reached.
# Valid values:
#   None: No deletion occurs (even after threshold)
#   "Oldest": Delete oldest records first (based on insertion_index) after threshold
#   "Random": Delete random records after threshold. This maintains a roughly constant collection size over time.
DELETE_BEFORE_INSERT_STRATEGY = "Oldest"

# Threshold for enabling deletion before insert
# No records will be deleted until the record count reaches this threshold.
# After threshold is reached, DELETE_BEFORE_INSERT_STRATEGY determines which records to delete.
SWITCH_TO_DELETE_BEFORE_INSERT_THRESHOLD = 100000

# Training settings
TRIALS_PER_EXPERIMENT = 1000  # Number of trials to run per training experiment
