from engram import Engram, EngramStore
import numpy as np
from settings import NOISE, MIN_RESULTS, FEEDBACK_BUFFER_SIZE

# Generic brain that can be trained to make decisions based on input
#

class EngramBrain:

    def __init__(self, input_size: int, output_size: int, engram_store: EngramStore) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.engram_store = engram_store
        self.highest_score = 0.0
        self.lowest_score = 0.0
        self.feedback_queue = []
        # if init_store:
        #     self.engram_store.init()

    # Given an input, generate an output
    def decide(self, input: list[float]) -> list[float]:
        resonator = self.input_to_resonator(input)
        resonating_engrams = self.get_resonating_engrams(resonator, MIN_RESULTS)
        scored_ngrams = self.score_engrams(resonating_engrams)
        return self.make_output(input, scored_ngrams)
    
    # Given an input, generate a state vector used to look up engrams
    def input_to_resonator(self, input: list[float]) -> list[float]:
        # A neural network could go here.
        # For now, just return the input
        return input
    
    # Query an EngramStore for resonating engrams
    def get_resonating_engrams(self, resonator: list[float], MIN_RESULTS) -> list[Engram, float]:
        results = self.engram_store.nearest(resonator, MIN_RESULTS)
        return results
          
    # Assign a score to the Engrams
    def score_engrams(self, resonating_engrams: list[Engram, float]) -> list[Engram, float]:
        # For each engram, calculate a score by multiplying the engram's outcome by the inverse square of its distance from the resonator.
        # Return an array of engrams and their scores, sorted by score.    
        result = []
        for engram, distance in resonating_engrams:
            score = engram.outcome # * (1 - distance**2)
            result.append((engram, score))
            
        result.sort(key=lambda x: x[1], reverse=True)   
        return result
    
    def make_output(self, input: list[float], scored_engrams: list[Engram, float]) -> list[float]:
        # Return a vector of length output_size (Engram.action)
        # Different algorithms could go here, such as a neural network, which could take into account the scary low-scoring engrams too.
        if len(scored_engrams) == 0:
            return np.random.random(self.output_size)
        else:
            # Create a weighting for each action, based on the sum of the scores
            # Make a lookup table of action to score
            action_scores = np.zeros(self.output_size)
            action_counts = np.zeros(self.output_size)
            for engram, score in scored_engrams:
                action_scores[int(engram.action)] = action_scores[int(engram.action)] + score
                action_counts[int(engram.action)] += 1

            # Now average them, handling the case where there are no results for an action
            action_scores = np.divide(action_scores, action_counts, out=np.zeros_like(action_scores), where=action_counts!=0)

            highest_score = np.max(action_scores)
            lowest_score = np.min(action_scores)
            if self.highest_score < highest_score:
                self.highest_score = highest_score
            if self.lowest_score > lowest_score:
                self.lowest_score = lowest_score

            # Overall average of positive scores
            average_score = np.mean([score for engram, score in scored_engrams if score > 0.0])
            if np.isnan(average_score):
                average_score = 0.0

            noise = NOISE

            #print(f"Average score: {average_score}")

            #print(f"Noise: {noise}")
            action_scores = action_scores + np.random.normal(0, noise, self.output_size)
   
            return action_scores.tolist()                          
    
    def apply_feedback(self, input: list[float], action: int, outcome: float) -> None:
        # Given an input, output, and outcome, create a new engram and add it to the EngramStore
        resonator = self.input_to_resonator(input)
        new_engram = Engram(vector=resonator, action=action, outcome=outcome)
        #print(f"New engram> action: {action}, outcome: {outcome}")
        self.queue_feedback(new_engram, FEEDBACK_BUFFER_SIZE)

    def queue_feedback(self, engram: Engram, queue_size) -> None:
        # Add feedback to the queue
        # Remove the first item in the queue and save to the collection
        self.feedback_queue.append(engram)
        if (len(self.feedback_queue) > queue_size):
            # print(f"Saving engram to collection")
            engram_to_save = self.feedback_queue.pop(0)
            self.engram_store.insert(engram_to_save)

    def flush_feedback_queue(self) -> None:
        # Save all items in the queue to the collection
        print("Flushing feedback queue")
        for engram in self.feedback_queue:
            self.engram_store.insert(engram)
        self.feedback_queue = []
        
        

    
 

        








