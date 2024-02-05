import numpy as np
from abc import ABC, abstractmethod

class IResonatorFactory(ABC):

    @abstractmethod
    def make_resonator(self, input: np.ndarray) -> np.ndarray:
        pass


