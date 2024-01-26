
import gymnasium as gym
from engram_brain import EngramBrain
from engram import EngramStore
import numpy
from settings import READ_ONLY, DROP_COLLECTION, COLLECTION_NAME, HIT_POINTS, MAX_TRIAL_LENGTH, METABOLIC_COST

class Trainer:
    # constructor
    def __init__(self, env, brain):
        self.env = env
        self.brain = brain
        self.trial_count = 0
        self.most_common_action_span = 10
        self.average_reward_span = 10

    def reset(self):
        self.env.reset()

    def trial(self):
        # Run one trial
        self.trial_count += 1
        print(f"Trial: {self.trial_count +1} ========================")
        self.env.reset()

        hit_points = HIT_POINTS
        total_reward = 0.0
        most_common_action = 0
        average_reward = 0.0
        action_window = [0] * self.most_common_action_span
        reward_window = [0] * self.average_reward_span
        observation, _ = self.env.reset()
        quit = False

        for time_step in range(MAX_TRIAL_LENGTH):
            brain_output = self.brain.decide(observation)
            normalised = [x - min(brain_output) for x in brain_output]
            
            output_sum = sum(normalised)
            if output_sum == 0.0:
                action = numpy.random.choice(numpy.arange(4))
            else:
                probabilities = [x / output_sum for x in normalised]
                # Then choose one
                action = numpy.random.choice(numpy.arange(4), p=probabilities)

            observation, reward, terminated, truncated, info = self.env.step(action)
            observation[6] = most_common_action
            observation[7] = average_reward

            total_reward = total_reward + reward
            # Normalise the reward into the range -1.0 to 1.0. Max reward is 200, min is -100
            MAX_REWARD = 20
            if reward > MAX_REWARD:
                reward = MAX_REWARD
            elif reward < -MAX_REWARD:
                reward = -MAX_REWARD
            reward = reward / MAX_REWARD

            #print(f"{action} | {brain_output[action]} | {normalised[action]} | {probabilities[action]} | {observation[6]} | {observation[7]}")
            
            print(f"Hit points: {hit_points}")
            print()
            
            if READ_ONLY == False:
                #print(f"Reward: {reward}")
                self.brain.apply_feedback(observation, action, reward)

            action_window.append(action)
            action_window.pop(0)
            most_common_action = max(set(action_window), key = action_window.count)

            reward_window.append(reward)
            reward_window.pop(0)
            average_reward = sum(reward_window) / len(reward_window)

            hit_points += reward
            hit_points -= METABOLIC_COST
            if hit_points < 0:
                print("** DEAD\n")
                quit = True

            if average_reward == MAX_REWARD:
                quit = True

            if terminated or truncated:
                quit = True

            if quit:
                print(f"*** Length of trial: {time_step}")
                print(f"*** Total reward: {total_reward}")
                return


def train():
    # Create the environment
    env = gym.make("LunarLander-v2", render_mode="human")

    store = EngramStore(COLLECTION_NAME, DROP_COLLECTION)

    # Create the brain
    brain = EngramBrain(8, 4, store)

    trainer = Trainer(env, brain)

    for _ in range(10000000):
        trainer.trial()

    env.close()


train()