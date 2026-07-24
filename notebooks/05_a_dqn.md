# DQN - Deep Q-Network

You just built a Q-learning agent that works perfectly on a 4x4 gridworld.
Now imagine the state is not a number from 0 to 15. Imagine it is the pixels of a video game screen.
Or a 100-dimensional sensor reading. Or a chess board.

The Q-table breaks immediately. You cannot have a row for every possible pixel configuration.
There are more possible Atari game frames than atoms in the universe.

DQN solves this with one idea: replace the table with a neural network.

---

## The Core Idea

In Q-learning, you look up Q-values from a table:

```
Q-value = q_table[state, action]
```

In DQN, you compute Q-values using a neural network:

```
Q-values = neural_network(state)   # returns one Q-value per action
```

The network takes the current state as input and outputs a Q-value for every possible action.
You still pick the action with the highest Q-value. The Bellman equation still applies.
The only thing that changed is where the Q-values come from.

This is why DQN generalises. A neural network can learn patterns across states it has never
seen before. The Q-table could not - it only knew about states it had explicitly visited.

---

## What the Network Looks Like

For our gridworld, the network is tiny:

```
input:  one-hot encoding of current state  (16 numbers, one per state)
        |
hidden layer: 64 neurons, ReLU activation
        |
hidden layer: 64 neurons, ReLU activation
        |
output: Q-value for each action  (4 numbers, one per action)
```

One-hot encoding just means: state 3 becomes [0, 0, 0, 1, 0, 0, ..., 0].
A vector of zeros with a 1 in the position of the current state.
This is how you feed a discrete state into a neural network.

---

## Training the Network

In Q-learning, we updated one cell of the table:

```
q_table[s, a] += alpha * (target - q_table[s, a])
```

In DQN, we do gradient descent on the network weights:

```
loss = (target - network(state)[action]) ** 2
gradients = jax.grad(loss)(network_params)
network_params = network_params - learning_rate * gradients
```

Same idea. Nudge toward the Bellman target. Just using a neural network instead of a table.

---

## Two Problems DQN Had to Solve

The original attempt to use neural networks for RL (before DeepMind's 2013 paper) kept failing.
The network would train for a while and then suddenly forget everything and collapse.

DeepMind's DQN paper solved this with two tricks. Both are in our code.

### Problem 1 - Correlated samples

In a normal neural network training loop, you shuffle your dataset and sample random batches.
This is important because consecutive examples should not be too similar - correlation between
samples makes training unstable.

In RL, consecutive states are extremely correlated. State 4 is always followed by state 5 or 8.
If you train on transitions as they come in, the network sees nothing but correlated data.

**Fix: Experience Replay**

Store every (state, action, reward, next_state) transition in a memory buffer.
When it is time to train, sample a random batch from this buffer instead of using the
most recent transitions. This breaks the correlation.

```
replay_buffer.add(state, action, reward, next_state, done)
batch = replay_buffer.sample(batch_size=32)
train_on(batch)
```

### Problem 2 - Moving targets

During training, the Bellman target is:

```
target = reward + gamma * max(Q(next_state))
```

But Q(next_state) is computed by the same network you are training.
So every time you update the network, the target moves too.
It is like trying to hit a moving bullseye - the network chases its own tail and diverges.

**Fix: Target Network**

Keep two copies of the network:
- The online network: the one you update every step
- The target network: a frozen copy, updated only every N steps

Use the target network to compute Q(next_state) in the Bellman equation.
This keeps the target stable while the online network trains.

```
target = reward + gamma * max(target_network(next_state))  # stable target
loss   = (target - online_network(state)[action]) ** 2     # update online only
```

Every 100 steps: copy online network weights into the target network. Refresh the anchor.

---

## The Full DQN Training Loop

```
initialise online_network and target_network (same weights)
initialise replay_buffer (empty)

for each episode:
    state = environment.reset()

    while not done:
        action = epsilon_greedy(online_network, state)
        next_state, reward, done = environment.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        if replay_buffer has enough samples:
            batch = replay_buffer.sample(32)
            targets = reward + gamma * max(target_network(next_state))
            loss = mean((targets - online_network(state)[action]) ** 2)
            update online_network using gradient of loss

        every 100 steps:
            target_network = copy of online_network

        state = next_state
```

Compare this to the Q-learning loop from Day 4. The structure is almost identical.
The differences are: experience replay buffer, two networks, and gradient descent instead of a table update.

---

## Summary

| | Q-learning | DQN |
|---|---|---|
| How Q-values are stored | Table | Neural network |
| Scales to large state spaces | No | Yes |
| Training method | Direct table update | Gradient descent |
| Key extra components | None | Replay buffer, target network |
| Generalises to unseen states | No | Yes |

DQN was the paper that started the deep RL revolution. DeepMind published it in 2013 and it
learned to play Atari games at superhuman level from raw pixels. The architecture we build
today is the same one. The only difference is our environment is smaller.