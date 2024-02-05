
import gymnasium as gym
from EngramBrain import EngramBrain
from engram import EngramStore
import numpy
from WeightedResonatorFactory import WeightedResonatorFactory
from settings import READ_ONLY, USE_HIT_POINTS, HIT_POINTS, MAX_TRIAL_LENGTH, METABOLIC_COST, PROBABILISTIC_CHOICE, LOGARITHMIC

class Trainer:
    # constructor
    def __init__(self, instance_name: str, dimension_weights: list[float], reset: bool = True):
        self.trial_count = 0
        self.reset = reset
        self.instance_name = instance_name
        # Create the resonator factor and brain      
        self.resonator_factory = WeightedResonatorFactory(dimension_weights)
        self.brain = EngramBrain(9, 4, EngramStore(instance_name, reset), self.resonator_factory)
        self.env = gym.make("LunarLander-v2", render_mode="human")

    def reset(self):
        self.env.reset()

    # Run a number of trials and return the average reward
    def train(self) -> float:
        print(f"Training {self.instance_name}")
        # Create the environment
        
        TRIALS = 10
        total_reward = 0.0
        for _ in range(TRIALS):
            total_reward += self.trial()

        self.env.close()
        result = total_reward / TRIALS
        print(f">>>>>>> Average reward: {result}")
        return result


    def trial(self):
        # Run one trial
        self.trial_count += 1
        print(f"Trial: {self.trial_count} ========================")
        self.env.reset()

        hit_points = HIT_POINTS
        total_reward = 0.0
        observation, _ = self.env.reset()
        quit = False

        for time_step in range(MAX_TRIAL_LENGTH):
            
            brain_output = self.brain.decide(observation)
            normalised = [x - min(brain_output) for x in brain_output]

            if PROBABILISTIC_CHOICE == True:    
                output_sum = sum(normalised)
                if output_sum == 0.0:
                    action = numpy.random.choice(numpy.arange(4))
                else:                     
                    probabilities = [x / output_sum for x in normalised]
                    # Then choose one
                    action = numpy.random.choice(numpy.arange(4), p=probabilities)
            else:
                action = numpy.argmax(normalised)

            raw_observation, reward, terminated, truncated, info = self.env.step(action)
            observation = normalise_observation(raw_observation)

            total_reward = total_reward + reward
            
            if READ_ONLY == False:
                #print(f"Reward: {reward}")
                # add time to observation array
                
                self.brain.apply_feedback(observation, action, reward)

            if USE_HIT_POINTS == True:
                hit_points += reward
                hit_points -= METABOLIC_COST
                if hit_points < 0:
                    print("** DEAD\n")
                    quit = True

            if terminated or truncated:
                quit = True

            if quit:
                break

        print(f"*** Length of trial: {time_step}")
        print(f"*** Total reward: {total_reward}")

        return total_reward


def normalise_observation(observation: list[float]) -> list[float]:
    observation[0] = observation[0] / 1.5
    observation[1] = observation[1] / 1.5
    observation[2] = observation[2] / 5.0
    observation[3] = observation[3] / 5.0
    observation[4] = observation[4] / 3.1415927
    observation[5] = observation[5] / 5.0
    observation[6] = 0
    observation[7] = 0
    return observation