
import gymnasium as gym
from engram_brain import EngramBrain
from engram import EngramStore
import numpy
from settings import READ_ONLY, DROP_COLLECTION, COLLECTION_NAME, HIT_POINTS, MAX_TRIAL_LENGTH, METABOLIC_COST, ORIENTATION_BONUS

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

    def trial(self, time: float):
        # Run one trial
        self.trial_count += 1
        print(f"Trial: {self.trial_count +1} ({time}) ========================")
        self.env.reset()

        hit_points = HIT_POINTS
        total_reward = 0.0
        most_common_action = 0
        average_reward = 0.0
        action_window = [0] * self.most_common_action_span
        reward_window = [0] * self.average_reward_span
        observation, _ = self.env.reset()

        # Normalise the observation values, assuming the following ranges:
        # 0: -1.5 to 1.5
        # 1: -1.5 to 1.5
        # 2: -5.0 to 5.0
        # 3: -5.0 to 5.0
        # 4: -3.1415927 to 3.1415927
        # 5: -5.0 to 5.0
        # 6: 0 to 0
        # 7: 0 to 0

        # Bonus reward, based on how close observation 4 is to 0
        

 
        observation = numpy.append(observation, time)
        quit = False

        for time_step in range(MAX_TRIAL_LENGTH):
            
            brain_output = self.brain.decide(observation)
            normalised = [x - min(brain_output) for x in brain_output]

            action = numpy.argmax(normalised)
            
            # output_sum = sum(normalised)
            # if output_sum == 0.0:
            #     action = numpy.random.choice(numpy.arange(4))
            # else:
            #     probabilities = [x / output_sum for x in normalised]
            #     # Then choose one
            #     action = numpy.random.choice(numpy.arange(4), p=probabilities)

            observation, reward, terminated, truncated, info = self.env.step(action)
            observation[0] = observation[0] / 1.5
            observation[1] = observation[1] / 1.5
            observation[2] = observation[2] / 5.0
            observation[3] = observation[3] / 5.0
            observation[4] = observation[4] / 3.1415927
            observation[5] = observation[5] / 5.0
            observation[6] = 0
            observation[7] = 0

            bonus_reward = (abs(observation[4]))*ORIENTATION_BONUS
            reward = reward + bonus_reward

            # observation[6] = most_common_action
            # observation[7] = average_reward
            observation = numpy.append(observation, 0)

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
                # add time to observation array
                
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
                break
        # Only save the last few seconds if the lander is still alive
        if hit_points > 0:
            self.brain.flush_feedback_queue()
        print(f"*** Length of trial: {time_step}")
        print(f"*** Total reward: {total_reward}")


def train():
    # Create the environment
    env = gym.make("LunarLander-v2", render_mode="human")

    store = EngramStore(COLLECTION_NAME, DROP_COLLECTION)

    # Create the brain
    brain = EngramBrain(9, 4, store)

    trainer = Trainer(env, brain)

    time = -1.0
    for x in range(10000000):
        #time += 0.00001
        time = 0
        trainer.trial(time)

    env.close()


train()