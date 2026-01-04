# Complex Instincts and Memory-Based Behavior

## The Problem: Complex Instincts

Animals routinely display highly structured, context-sensitive behaviours on their very first attempt, often without practice or parental instruction.

### Classic Examples
- Spiders constructing orb webs
- Birds constructing species-specific nests
- Insects performing multi-stage hunting sequences
- Courtship dances

These behaviours are arguably:
- Too complex to plausibly emerge almost immediately from trial-and-error learning
- Too information-rich to be evolved and encoded as genetic instructions

The dominant explanation is that instincts are "genetically programmed," but this raises the question: where is the program? How does the program develop?

## Genetically Evolved Instincts?

We know that instincts are innate, and appear without any learning. But is it feasible for them to evolve, given our knowledge of genetics?

### Hard-coded Neural Structures

If they are evolved, they likely don't take the form of rigid circuit-board like systems encoding logical steps, which would be too rigid and fragile. In any case, individual brains within a species are structurally similar, but the patterns of connectivity are unique and not the well-defined circuit-board that should be required for pre-programmed complex behaviour.

### Adaptive Learning Models

Genetic regulatory networks can exhibit complex behaviour, even akin to a neural network. Other cellular feedback mechanisms could also be happening that could potentially encode complex behaviour.

If instincts are indeed implemented as more of a statistical, emergent system like modern machine learning, how could genetics implement it? And how could they "train" it?

In machine learning, parameters are adjusted based on feedback across many trials, typically thousands or hundreds of thousands. Genetic evolution is supposed to occur through rare random mutations in genes. Such "genetic" algorithms, where parameters are altered randomly, have been studied in machine learning and are known to be relatively slow compared to other methods. And if the organism must wait for a rare genetic mutation to randomly change parameters, convergence on a solution would take a very long time.

A randomly-adjusted learning system is probably too slow to be a practical method for developing useful algorithms to keep pace with changing environments and ecosystems. This is especially true when long-horizon, multi-step tasks are considered, since the assignment of credit to particular steps is often not possible in real life without instruction.

## Proposal: Instinct as Memory

If not learned, and not encoded in inherited structures, where does complex instinctive behaviour come from? I explore the proposal that instincts emerge from some capability for living organisms to be somehow influenced by the experience and behaviour of previous similar individuals, via an effect like Sheldrake's "morphic resonance".

In this view, behaviour is guided by similarity to past situations:
- Action selection is biased toward historically successful patterns
- Learning is cumulative across populations, not isolated within one individual lifetime

If animal behaviour is affected by the previous behaviors of similar animals, then successful behaviours will tend to create yet more historical instances of that behaviour, making it increasingly easier for an animal to "resonate" with a successful strategy.

Although nature and mechanism for this phenomenon are not known, we can still attempt to explore the potential for this idea. Would such a system work? I have built an example software system that exhibits the properties described above, to see if it could indeed support the learning of complex and adaptive behaviours.

## High-dimensional Vectors

Vector databases seem a good platform to experimenting with this idea. Prior experiences can be stored as collections of numbers, and treated as positions in a multi-dimensional space. They can be retrieved based on similarity, using a euclidean distance as a metric. This makes it easy to experiment with fetching sets of contextually related historical experiences and using them to bias the decisions of a software agent.

**Just to be clear**, the claim is not that nature literally uses a vector database. This is only an analogue that shares some key properties with the hypothetical memory phenomenon. The hypothesis is that this class of mechanism is sufficient to produce instinct-like behaviour.

## The Lunar Lander Environment

To test whether memory-based decision-making can support complex, adaptive behavior, the system was evaluated on the LunarLander-v2 environment from Gymnasium (Klimov, 2016). This environment presents a classic rocket trajectory optimization problem where an agent must learn to land a spacecraft safely on a landing pad.

### Task Description

The landing pad is always positioned at coordinates (0, 0). The agent controls a lunar lander with infinite fuel, allowing it to learn flight and landing strategies. The environment uses discrete actions, which aligns with optimal control theory: according to Pontryagin's maximum principle, it is optimal to fire engines at full throttle or turn them off completely, rather than using intermediate throttle levels.

### Observation Space

The environment provides an 8-dimensional state vector containing:
- **Position**: x and y coordinates of the lander (ranging from approximately -2.5 to +2.5)
- **Linear velocities**: vx and vy components (ranging from approximately -10 to +10)
- **Angle**: The lander's orientation in radians (ranging from approximately -2π to +2π)
- **Angular velocity**: Rate of rotation (ranging from approximately -10 to +10)
- **Leg contact**: Two boolean values indicating whether each leg is in contact with the ground

### Action Space

There are four discrete actions available:
- **0**: Do nothing
- **1**: Fire left orientation engine
- **2**: Fire main engine
- **3**: Fire right orientation engine

### Reward Structure

The reward function is designed to encourage safe, controlled landings:

- **Position reward**: Increased/decreased based on proximity to the landing pad
- **Velocity reward**: Increased/decreased based on how slowly the lander is moving (encouraging gentle landings)
- **Orientation penalty**: Decreased reward when the lander is tilted (encouraging horizontal landings)
- **Leg contact bonus**: +10 points for each leg in contact with the ground
- **Engine firing costs**: -0.03 points per frame for side engines, -0.3 points per frame for the main engine

The episode receives an additional reward of -100 points for crashing or +100 points for landing safely. An episode is considered solved if it achieves at least 200 points.

### Episode Termination

An episode terminates when:
1. The lander crashes (body contacts the moon surface)
2. The lander exits the viewport (x coordinate exceeds 1)
3. The lander becomes inactive (Box2D physics determines the body has come to rest)

### Why This Environment?

The Lunar Lander environment is well-suited for testing memory-based decision-making because:
- It requires **multi-step planning**: The agent must navigate from an initial position to the landing pad, requiring a sequence of coordinated actions
- It has **delayed feedback**: The consequences of early actions (like firing engines to adjust trajectory) only become apparent later in the episode
- It involves **continuous control in discrete action space**: While the state is continuous, the discrete actions require the agent to learn when to apply each control
- It has a **clear success criterion**: Landing safely provides unambiguous feedback about performance

This combination of features makes it a challenging test case for whether similarity-based memory retrieval can guide effective behavior without traditional reinforcement learning algorithms.

## Procedure

### System Initialization

The system uses Milvus, a vector database, to store and retrieve historical experiences. Each experience is stored as an **engram** containing three fields: a state vector (9 dimensions for the LunarLander environment), the action taken (an integer from 0-3), and the outcome (a normalized reward value between -1 and 1).

The Milvus collection is initialized with an IVF_FLAT index using L2 (Euclidean) distance as the similarity metric. The index uses 1024 clusters (nlist=1024) for efficient approximate nearest neighbor search, with 16 clusters probed during each query (nprobe=16) to balance search speed and accuracy.

### Training Procedure

Training proceeds through a series of trials on the LunarLander-v2 environment from Gymnasium (see "The Lunar Lander Environment" section above). Each trial consists of the following steps:

1. **Observation Normalization**: The raw state observation from the environment (8 dimensions: x, y, velocity components, angle, angular velocity, and leg contact booleans) is normalized by dividing each component by its typical range. The normalized observation is then extended to 9 dimensions by zero-padding.

2. **Resonator Creation**: The normalized state is converted into a **resonator vector** for similarity matching. This transformation applies learned dimension weights to emphasize or de-emphasize different state components, then appends a success metric (the agent's recent average performance) as an additional dimension. The resonator vector thus encodes both the current state and the agent's current performance level.

3. **Similarity Search**: The system queries the Milvus database to find the 300 most similar historical engrams to the current resonator vector, using Euclidean distance in the 9-dimensional space.

4. **Action Scoring**: For each possible action (0-3), the system calculates a score by averaging the outcomes of all retrieved engrams that recommended that action. Actions with no supporting engrams receive a score of zero.

5. **Action Selection**: The action scores are normalized (by subtracting the minimum score) and either:
   - Selected deterministically using argmax (the highest-scoring action), or
   - Selected probabilistically with probabilities proportional to the normalized scores
   
   Gaussian noise (standard deviation 0.1) is added to the scores before selection to encourage exploration.

6. **Environment Step**: The selected action is executed in the environment, producing a new observation and reward.

7. **Feedback Queueing**: The observation, action, and immediate reward are stored in a feedback queue for later processing.

Steps 2-7 repeat until the trial terminates (either by landing successfully, crashing, or reaching the maximum step limit of 400).

### Learning Mechanism

After each trial completes, the feedback queue is processed to create new engrams:

1. **Outcome Normalization**: The total trial reward (typically ranging from approximately -300 to +300) is normalized to the range [-1, 1] by dividing by 300 and clamping extreme values.

2. **Success Tracking**: The system maintains two success metrics:
   - **Best success**: The highest normalized outcome achieved across all trials
   - **Mean success**: A running average of the last 20 trials' normalized outcomes (or all trials if fewer than 20 have completed)

3. **Engram Creation**: For each (observation, action, reward) triplet in the feedback queue, a new engram is created:
   - The resonator vector is computed from the observation and the trial's success value
   - The action is the integer action taken
   - The outcome is the normalized reward from that step
   
   The engram is then inserted into the Milvus collection, where it becomes available for future similarity searches.

This mechanism ensures that successful trials contribute more engrams with positive outcomes, while unsuccessful trials contribute engrams with negative outcomes. Over time, the collection accumulates a memory of which actions tend to work well in which situations, creating a bias toward historically successful patterns.

### Genetic Algorithm Evolution (Optional)

The system can optionally evolve optimal dimension weights using a genetic algorithm. The **Breeder** maintains a population of genomes, where each genome is a vector of 8 weights (one per state dimension).

For each generation:
1. **Fitness Evaluation**: Each genome is evaluated by training a complete agent using those dimension weights and measuring the average reward achieved.
2. **Selection**: The top 50% of the population (by fitness) are selected as parents.
3. **Breeding**: Offspring are created by randomly selecting each gene from either parent with equal probability (shuffle crossover).
4. **Mutation**: Each offspring is mutated by adding small random noise (mutation rate 0.001).
5. **Replacement**: The bottom 50% of the population is replaced with the new offspring.

This process allows the system to discover which state dimensions are most important for decision-making, potentially improving performance by focusing similarity matching on the most relevant aspects of the state.

### Experimental Configuration

Experiments were conducted on the LunarLander-v2 environment, which requires the agent to learn to land a spacecraft safely. The environment provides 8-dimensional state observations and 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine).

Key parameters used in the experiments:
- State vector size: 9 dimensions (8 from environment + 1 success metric)
- Action space: 4 discrete actions
- Minimum similar engrams retrieved: 300
- Maximum trial length: 400 steps
- Noise standard deviation: 0.1
- Success metric window: 20 trials

The system can operate in read-only mode (for evaluation without learning) or with learning enabled. Visualization can be toggled to observe the agent's behavior during training.
