from engram import Engram, EngramStore
import numpy as np
from settings import NOISE, MIN_RESULTS, TRIAL_SUCCESS_MULTIPLIER_SCALE
from IResonatorFactory import IResonatorFactory
from typing import List, Tuple

# Generic brain that can be trained to make decisions based on input
#

class EngramBrain:

    def __init__(self, input_size: int, output_size: int, engram_store: EngramStore, resonator_factory: IResonatorFactory) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.engram_store = engram_store
        self.resonator_factory = resonator_factory

    # Given an input, generate an output
    def decide(self, input: np.ndarray, success: float, return_distance_info: bool = False):
        resonator = self.input_to_resonator(input, success)
        resonating_engrams = self.get_resonating_engrams(resonator, MIN_RESULTS)
        scored_ngrams = self.score_engrams(resonating_engrams)
        output = self.make_output(input, scored_ngrams)
        
        if return_distance_info:
            # Calculate average distance of retrieved engrams
            if len(resonating_engrams) > 0:
                avg_distance = sum(distance for _, distance in resonating_engrams) / len(resonating_engrams)
            else:
                avg_distance = float('inf')
            return output, avg_distance
        else:
            return output
    
    # Given an input, generate a state vector used to look up engrams
    def input_to_resonator(self, input: np.ndarray, success: float) -> np.ndarray:
        # A neural network could go here.
        # For now, just return the input
        return self.resonator_factory.make_resonator(input, success)
    
    # Query an EngramStore for resonating engrams
    def get_resonating_engrams(self, resonator: np.ndarray, MIN_RESULTS) -> List[Tuple[Engram, float]]:
        results = self.engram_store.nearest(resonator, MIN_RESULTS)
        return results
          
    # Assign a score to the Engrams
    def score_engrams(self, resonating_engrams: List[Tuple[Engram, float]]) -> List[Tuple[Engram, float]]:
        # For each engram, calculate a score by multiplying the engram's outcome by the trial success multiplier.
        # Return an array of engrams and their scores, sorted by score.    
        result = []
        for engram, distance in resonating_engrams:
            score = engram.outcome
            
            # Apply trial success multiplier if trial metadata is available
            if engram.trial_number > 0 and hasattr(engram, 'trial_final_success'):
                multiplier = 1.0 + (engram.trial_final_success * TRIAL_SUCCESS_MULTIPLIER_SCALE)
                score = score * multiplier
            
            result.append((engram, score))
            
        result.sort(key=lambda x: x[1], reverse=True)   
        return result
    
    def make_output(self, input: list[float], scored_engrams: list[Engram, float]) -> list[float]:
        # Return a vector of length output_size (Engram.action)
        # Different algorithms could go here, such as a neural network, which could take into account the scary low-scoring engrams too.


        scored_engrams = [(engram, score) for engram, score in scored_engrams]
        if len(scored_engrams) == 0:
            return np.random.random(self.output_size)
        else:
            # Create a weighting for each action
            # This is the average score of the engrams that suggest it
            # First get the count and total score for each action
            action_scores = np.zeros(self.output_size)
            action_counts = np.zeros(self.output_size)
            for engram, score in scored_engrams:
                action_scores[int(engram.action)] = action_scores[int(engram.action)] + score
                action_counts[int(engram.action)] += 1

            # Now average them, handling the case where there are no results for an action
            action_scores = np.divide(action_scores, action_counts, out=np.zeros_like(action_scores), where=action_counts!=0)

            # Apply noise
            noise = NOISE
            action_scores = action_scores + np.random.normal(0, noise, self.output_size)

            return action_scores.tolist()                          
    
    def apply_feedback(self, input: list[float], action: int, outcome: float, success: float, 
                      trial_number: int = 0, trial_final_success: float = 0.0, 
                      trial_raw_reward: float = 0.0, trial_episode_length: int = 0, 
                      trial_is_success: bool = False) -> None:
        # Given an input, output, and outcome, create a new engram and add it to the EngramStore
        resonator = self.input_to_resonator(input, success)

        #new_engram = Engram(vector=input, action=action, outcome=outcome)
        new_engram = Engram(vector=resonator, action=action, outcome=outcome)
        self.engram_store.insert(new_engram, trial_number, trial_final_success, 
                                trial_raw_reward, trial_episode_length, trial_is_success)
        #self.engram_store.insert(resonator)

        
        
    
 

        








