import numpy as np
from IResonatorFactory import IResonatorFactory
from settings import VECTOR_COMPONENT_WEIGHTS

class WeightedResonatorFactory(IResonatorFactory):
    def __init__(self, weights: list[float] = None):
        super().__init__()
        # Use provided weights or default from settings
        self.weights = np.array(weights if weights is not None else VECTOR_COMPONENT_WEIGHTS)
    
    def make_resonator(self, input: np.ndarray, success: float) -> np.ndarray:
        # Apply weights to input components: multiply each component by its corresponding weight
        # This affects L2 distance calculations: components with higher weights contribute more to distance
        weighted_input = input * self.weights
        # Return only the weighted input state (success is not used in distance calculations)
        # Success is stored separately in trial_final_success field for scoring purposes
        return weighted_input
