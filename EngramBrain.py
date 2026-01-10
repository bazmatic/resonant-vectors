from engram import Engram, BaseEngramStore
import numpy as np
from settings import MIN_RESULTS, TRIAL_SUCCESS_MULTIPLIER_SCALE, DELETE_BEFORE_INSERT_STRATEGY, SWITCH_TO_DELETE_BEFORE_INSERT_THRESHOLD
from noise import calculate_noise
from IResonatorFactory import IResonatorFactory
from typing import List, Tuple

# Generic brain that can be trained to make decisions based on input
#

class EngramBrain:

    def __init__(self, input_size: int, output_size: int, engram_store: BaseEngramStore, resonator_factory: IResonatorFactory) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.engram_store = engram_store
        self.resonator_factory = resonator_factory

    # Given an input, generate an output
    def decide(self, input: np.ndarray, success: float, return_distance_info: bool = False, 
               panic_factor: float = 0.0, episode_progress: float = 0.0):
        resonator = self.input_to_resonator(input, success)
        resonating_engrams = self.get_resonating_engrams(resonator, MIN_RESULTS)
        scored_ngrams = self.score_engrams(resonating_engrams)
        output = self.make_output(input, scored_ngrams, panic_factor, episode_progress)
        
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
    def score_engrams(self, resonating_engrams: List[Tuple[Engram, float]]) -> List[Tuple[Engram, float, float]]:
        # For each engram, calculate a score by multiplying the engram's outcome by the trial success multiplier.
        # Return an array of (engram, score, distance) tuples, sorted by score.
        # Distance is preserved for distance-based weighting in make_output.
        result = []
        for engram, distance in resonating_engrams:
            score = engram.outcome
            
            # Apply trial success multiplier if trial metadata is available
            if hasattr(engram, 'trial_final_success') and engram.trial_final_success != 0.0:
                multiplier = 1.0 + (engram.trial_final_success * TRIAL_SUCCESS_MULTIPLIER_SCALE)
                score = score * multiplier
            
            result.append((engram, score, distance))
            
        result.sort(key=lambda x: x[1], reverse=True)   
        return result
    
    def make_output(self, input: list[float], scored_engrams: list[tuple], 
                    panic_factor: float = 0.0, episode_progress: float = 0.0) -> list[float]:
        # Return a vector of length output_size (Engram.action)
        # Different algorithms could go here, such as a neural network, which could take into account the scary low-scoring engrams too.
        # Scores are weighted by distance: closer engrams have more influence.

        #If one of the legs is touching the ground, choose No Action (index 0) with 100% confidence
        if input[6] == 1 or input[7] == 1:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if len(scored_engrams) == 0:
            return np.random.random(self.output_size)
        else:
            # Create a weighting for each action
            # Scores are weighted by distance: weight = 1 / (1 + distance)
            # This gives closer engrams (smaller distance) more influence
            action_weighted_scores = np.zeros(self.output_size)
            action_total_weights = np.zeros(self.output_size)
            
            for item in scored_engrams:
                # Handle both old format (engram, score) and new format (engram, score, distance)
                if len(item) == 3:
                    engram, score, distance = item
                    # Weight by inverse distance: closer = higher weight
                    # Using 1 / (1 + distance) to prevent division by zero and ensure positive weights
                    weight = 1.0 / (1.0 + distance)
                else:
                    # Backward compatibility: if distance not provided, use weight of 1.0
                    engram, score = item
                    weight = 1.0
                
                action_idx = int(engram.action)
                action_weighted_scores[action_idx] += score * weight
                action_total_weights[action_idx] += weight

            # Calculate weighted average: sum(score * weight) / sum(weight)
            # Handle the case where there are no results for an action
            action_scores = np.divide(
                action_weighted_scores, 
                action_total_weights, 
                out=np.zeros_like(action_weighted_scores), 
                where=action_total_weights!=0
            )

            # Calculate noise using centralized helper (decays over total episode, not individual trials)
            noise = calculate_noise(episode_progress, panic_factor)
            
            action_scores = action_scores + np.random.normal(0, noise, self.output_size)

            return action_scores.tolist()                          
    
    def apply_feedback(self, input: list[float], action: int, outcome: float, success: float, 
                      trial_final_success: float = 0.0) -> None:
        # Given an input, output, and outcome, create a new engram and add it to the EngramStore
        # Store only the input state vector (not the resonator with success)
        # Success is stored separately in trial_final_success field for scoring purposes
        new_engram = Engram(vector=input, action=action, outcome=outcome)
        self.engram_store.insert(new_engram, trial_final_success)
    
    def batch_apply_feedback(self, inputs: List[list[float]], actions: List[int], outcomes: List[float], 
                            success: float, trial_final_successes: List[float] = None) -> None:
        """
        Apply feedback for multiple observations in a batch for better performance.
        All inputs, actions, and outcomes should have the same length.
        
        Args:
            inputs: List of observation vectors
            actions: List of actions taken
            outcomes: List of immediate rewards
            success: Overall trial success (for tracking)
            trial_final_successes: Per-step discounted credit values for temporal credit assignment
        """
        if len(inputs) == 0:
            return
        
        # Default to zeros if not provided
        if trial_final_successes is None:
            trial_final_successes = [0.0] * len(inputs)
        
        # Create engrams for all inputs, storing only the input state vector
        # Success is stored separately in trial_final_success field for scoring purposes
        engrams = []
        for input_vec, action, outcome in zip(inputs, actions, outcomes):
            engram = Engram(vector=input_vec, action=action, outcome=outcome)
            engrams.append(engram)
        
        # Determine deletion strategy
        strategy = DELETE_BEFORE_INSERT_STRATEGY
        
        # No deletion occurs until threshold is reached
        if strategy is not None:
            current_count = self.engram_store.get_count()
            if current_count < SWITCH_TO_DELETE_BEFORE_INSERT_THRESHOLD:
                # Before threshold is reached, don't delete anything
                strategy = None
        
        # Delete records based on strategy before inserting new ones
        if strategy is not None:
            num_to_delete = len(engrams)
            if strategy == "Oldest":
                self.engram_store.delete_oldest_records(num_to_delete)
            elif strategy == "Random":
                self.engram_store.delete_random_records(num_to_delete)
        
        # Batch insert all engrams with per-step discounted credits
        self.engram_store.batch_insert(engrams, trial_final_successes)

        
        
    
 

        








