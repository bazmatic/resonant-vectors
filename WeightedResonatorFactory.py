import numpy as np
from IResonatorFactory import IResonatorFactory

class WeightedResonatorFactory(IResonatorFactory):
    def __init__(self, dimension_weights: np.ndarray):
        super().__init__()
        self.dimension_weights = dimension_weights
    
    def make_resonator(self, input: np.ndarray) -> np.ndarray:
        # Apply the dimension weights to the input
        #w = [1, 0, 1, 1, 0.9, 0.9, 1, 0, 0]
        resonator = input * self.dimension_weights
        return resonator
