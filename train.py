
import gymnasium as gym
from engram_brain import EngramBrain
from engram import EngramStore
import numpy
from settings import READ_ONLY, DROP_COLLECTION, COLLECTION_NAME

# TODO: Include a memory 
def train():
    # Create the environment
    env = gym.make("LunarLander-v2", render_mode="human")

    store = EngramStore(COLLECTION_NAME, DROP_COLLECTION)

    # Create the brain
    brain = EngramBrain(8, 4, store)

    observation, info = env.reset()
    total_reward = 0.0
    most_common_action = 0
    most_common_action_span = 10
    average_reward = 0.0
    average_reward_span = 10
    action_window = [0] * most_common_action_span
    reward_window = [0] * average_reward_span
    trial_count = 0

    for _ in range(10000000):

        brain_output = brain.decide(observation) 
        # Set negative values to zero
        #output = [0 if x < 0 else x for x in brain_output]
        
        # Choose action probabilistically, based on the output values 0.0 to 1.0
        # Normalise to make them all positive. Find the span of the values.
        # Make them sum to 1.0.
        normalised = [x - min(brain_output) for x in brain_output]
        output_sum = sum(normalised)
        if output_sum == 0.0:
            action = numpy.random.choice(numpy.arange(4))
        else:
            probabilities = [x / output_sum for x in normalised]
            # Then choose one
            action = numpy.random.choice(numpy.arange(4), p=probabilities)

        observation, reward, terminated, truncated, info = env.step(action)
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

        print(f"{action} | {brain_output[action]} | {normalised[action]} | {probabilities[action]} | {observation[6]} | {observation[7]}")
        print(f"Reward: {reward}")
        print()
        # if reward > 0.0:
        #     print(f"Reward: {reward}")
        
        if READ_ONLY == False:
            #print(f"Reward: {reward}")
            brain.apply_feedback(observation, action, reward)
        # else:
        #     print("NO")
                # The action is the index of the element most highly activated
        
        action_window.append(action)
        action_window.pop(0)
        most_common_action = max(set(action_window), key = action_window.count)

        reward_window.append(reward)
        reward_window.pop(0)
        average_reward = sum(reward_window) / len(reward_window)
        if average_reward == MAX_REWARD:
            env.reset()

        if terminated or truncated:
            print(f"*** Total reward: {total_reward}")
            total_reward = 0.0
            observation, info = env.reset()
            trial_count += 1
            print(f"Trial: {trial_count +1} ========================")

        trial_count += 1

    env.close()

train()