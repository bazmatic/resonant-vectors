STATE_VECTOR_SIZE = 21
OUTPUT_VECTOR_SIZE = 4

# Past history settings
# Number of past steps to include in the state vector as rolling averages
PAST_HISTORY_STEPS = 8
# Noise settings - episode-based decay
# Noise starts at NOISE_START and asymptotically decays to NOISE_END during each episode
NOISE_START = 0.4  # Initial noise at episode start
NOISE_END = 0.001   # Target noise at episode end
NOISE_DECAY_RATE = 5  # Controls decay speed (higher = faster decay)
# Backward compatibility: keep NOISE as an alias for NOISE_START
NOISE = NOISE_START
MIN_RESULTS = 250
READ_ONLY = False
DROP_COLLECTION = False
USE_HIT_POINTS = True
HIT_POINTS = 500
MAX_TRIAL_LENGTH = 400
METABOLIC_COST = 0.2

# Panic feature settings
PANIC_ENABLED = True
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
TRIAL_SUCCESS_MULTIPLIER_SCALE = 0.3  # Controls strength of trial success multiplier on vector scores

# Vector component weightings
# Weightings for each state vector component to control their relative importance
# in similarity search calculations. Higher weights make components contribute more to distance.
# Default is all 1.0 (equal weighting). Modify to experiment with component importance.
# Organized by section with index comments for easy reference
VECTOR_COMPONENT_WEIGHTS = [
    # Current observation (indices 0-7)
    0.3,   # [0]  x position
    1.4,   # [1]  y position
    1.2,   # [2]  vx (horizontal velocity)
    1.2,   # [3]  vy (vertical velocity)
    1.2,   # [4]  angle
    1.0,   # [5]  angular velocity
    1.0,   # [6]  leg contact 1
    1.0,   # [7]  leg contact 2
    # Past input averages (indices 8-15)
    1.0,   # [8]  past avg: x position
    1.0,   # [9]  past avg: y position
    1.0,   # [10] past avg: vx
    1.0,   # [11] past avg: vy
    1.0,   # [12] past avg: angle
    1.0,   # [13] past avg: angular velocity
    1.0,   # [14] past avg: leg contact 1
    1.0,   # [15] past avg: leg contact 2
    # Past action distribution (indices 16-19)
    1.0,   # [16] past action dist: Nothing
    1.0,   # [17] past action dist: Left
    1.0,   # [18] past action dist: Main
    1.0,   # [19] past action dist: Right
    # Past reward average (index 20)
    1.0,   # [20] past avg: reward
]


# Vector sampling settings
# Fraction of vectors to randomly sample and save (0.0 to 1.0)
# 1.0 = save all vectors, 0.5 = save 50% randomly selected, 0.0 = save none
VECTOR_SAVE_RATE = 1

# Delete records before insert strategy
# Controls which records to delete before inserting new ones via batch_apply_feedback.
# No deletion occurs until SWITCH_TO_DELETE_BEFORE_INSERT_THRESHOLD is reached.
# Valid values:
#   None: No deletion occurs (even after threshold)
#   "Oldest": Delete oldest records first (based on insertion_index) after threshold
#   "Random": Delete random records after threshold. This maintains a roughly constant collection size over time.
DELETE_BEFORE_INSERT_STRATEGY = 'Random'

# Threshold for enabling deletion before insert
# No records will be deleted until the record count reaches this threshold.
# After threshold is reached, DELETE_BEFORE_INSERT_STRATEGY determines which records to delete.
SWITCH_TO_DELETE_BEFORE_INSERT_THRESHOLD = 70000

# Vector store backend selection
# Options: "milvus" (persistent, requires Milvus service) or "faiss" (in-memory, no external dependencies)
VECTOR_STORE_TYPE = "faiss"

# Training settings
TRIALS_PER_EXPERIMENT = 8000  # Number of trials to run per training experiment
