# %%
import numpy as np
import jax.numpy as jnp

# Note on this file:
# The Q-table is a small array we update thousands of times.
# We use numpy for the Q-table itself (mutable, fast for small tables).
# JAX comes in when we scale this up to neural networks in the next file (DQN).
# This is honest - tabular Q-learning predates JAX. Learn the algorithm first,
# then we port it to JAX when we build DQN.

# %%
# -------------------------------------------------------
# SECTION 1 - The Gridworld Environment
# -------------------------------------------------------

# 4x4 grid. States numbered 0 to 15, left to right, top to bottom.
#
#  0   1   2   3
#  4   5   6   7
#  8   9  10  11
# 12  13  14  15
#
# S = start (state 0), G = goal (state 15)
# H = hole  (states 5, 7, 11, 12)
#
#  S   .   .   .
#  .   H   .   H
#  .   .   .   H
#  H   .   .   G

GRID_SIZE = 4
NUM_STATES = 16
NUM_ACTIONS = 4   # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

START_STATE = 0
GOAL_STATE = 15
HOLES = {5, 7, 11, 12}
TERMINAL_STATES = HOLES | {GOAL_STATE}

REWARD_GOAL = 1.0
REWARD_HOLE = -1.0
REWARD_STEP = -0.01   # small penalty every step - encourages finding goal quickly


def step(state, action):
    """
    Take one step in the gridworld.
    Returns (next_state, reward, done).
    """
    row = state // GRID_SIZE
    col = state % GRID_SIZE

    if action == 0:    # UP
        row = max(0, row - 1)
    elif action == 1:  # DOWN
        row = min(GRID_SIZE - 1, row + 1)
    elif action == 2:  # LEFT
        col = max(0, col - 1)
    elif action == 3:  # RIGHT
        col = min(GRID_SIZE - 1, col + 1)

    next_state = row * GRID_SIZE + col

    if next_state == GOAL_STATE:
        return next_state, REWARD_GOAL, True
    elif next_state in HOLES:
        return next_state, REWARD_HOLE, True
    else:
        return next_state, REWARD_STEP, False


def print_grid(values, label=""):
    """Print any 16-value array as a 4x4 grid."""
    if label:
        print(f"\n{label}")
    for row in range(GRID_SIZE):
        print("  ", end="")
        for col in range(GRID_SIZE):
            state = row * GRID_SIZE + col
            val = values[state]
            if state == START_STATE:
                print(f" S({val:+.2f})", end="")
            elif state == GOAL_STATE:
                print(f" G({val:+.2f})", end="")
            elif state in HOLES:
                print(f" H({val:+.2f})", end="")
            else:
                print(f"  ({val:+.2f})", end="")
        print()


# Quick sanity check - take one step manually
next_s, r, done = step(0, 1)  # from state 0, go DOWN
print(f"From state 0, action DOWN -> state {next_s}, reward {r}, done {done}")
# state 0 is top-left, DOWN takes us to state 4

# %%
# -------------------------------------------------------
# SECTION 2 - Q-table initialisation
# -------------------------------------------------------

# Q-table shape: (16 states, 4 actions)
# Every value starts at 0 - the agent knows nothing
q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

print("Q-table shape:", q_table.shape)
print("Q-table at state 0 (all actions = 0):", q_table[0])

# %%
# -------------------------------------------------------
# SECTION 3 - Epsilon-greedy action selection
# -------------------------------------------------------

# With probability epsilon: explore (random action)
# With probability 1 - epsilon: exploit (best known action)

def choose_action(state, q_table, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(NUM_ACTIONS)   # random action
    else:
        return np.argmax(q_table[state])         # best known action


# %%
# -------------------------------------------------------
# SECTION 4 - The Q-learning update rule
# -------------------------------------------------------

# This is the Bellman equation applied as a small correction:
#
# Q(s, a) = Q(s, a) + alpha * (target - Q(s, a))
#
# target = reward + gamma * max(Q(next_state, all_actions))
#
# alpha = learning rate (how big a correction to make)
# gamma = discount factor (how much future rewards matter)

def q_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    current_q = q_table[state, action]

    if done:
        target = reward   # no future rewards if episode is over
    else:
        target = reward + gamma * np.max(q_table[next_state])

    q_table[state, action] = current_q + alpha * (target - current_q)
    return q_table


# %%
# -------------------------------------------------------
# SECTION 5 - Training loop
# -------------------------------------------------------

# Hyperparameters
ALPHA = 0.1          # learning rate
GAMMA = 0.99         # discount factor
EPSILON_START = 1.0  # start fully random
EPSILON_END = 0.01   # end mostly greedy
EPSILON_DECAY = 0.995
NUM_EPISODES = 2000

np.random.seed(42)
q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
epsilon = EPSILON_START

episode_rewards = []
episode_lengths = []

for episode in range(NUM_EPISODES):
    state = START_STATE
    total_reward = 0
    steps = 0

    while True:
        action = choose_action(state, q_table, epsilon)
        next_state, reward, done = step(state, action)

        q_table = q_update(q_table, state, action, reward, next_state, done, ALPHA, GAMMA)

        state = next_state
        total_reward += reward
        steps += 1

        if done or steps > 200:   # 200 step limit prevents infinite wandering
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_steps = np.mean(episode_lengths[-100:])
        print(f"Episode {episode + 1:4d} | avg reward (last 100): {avg_reward:.3f} | avg steps: {avg_steps:.1f} | epsilon: {epsilon:.3f}")

# %%
# -------------------------------------------------------
# SECTION 6 - What did the agent learn?
# -------------------------------------------------------

# Print the best Q-value for each state (how good is the best action here?)
best_q_per_state = np.max(q_table, axis=1)
print_grid(best_q_per_state, label="Best Q-value per state (higher = better position):")

# Print the learned policy as arrows
ACTION_SYMBOLS = {0: "^", 1: "v", 2: "<", 3: ">"}

print("\nLearned policy (what action does the agent take in each state?):")
print()
for row in range(GRID_SIZE):
    print("  ", end="")
    for col in range(GRID_SIZE):
        state = row * GRID_SIZE + col
        if state == GOAL_STATE:
            print("  G ", end="")
        elif state in HOLES:
            print("  H ", end="")
        else:
            best_action = np.argmax(q_table[state])
            print(f"  {ACTION_SYMBOLS[best_action]} ", end="")
    print()

# %%
# -------------------------------------------------------
# SECTION 7 - Watch the trained agent run one episode
# -------------------------------------------------------

print("\nWatching the trained agent navigate from start to goal:")
print("(epsilon = 0, fully greedy now)\n")

state = START_STATE
path = [state]
total_reward = 0

for step_num in range(50):
    action = np.argmax(q_table[state])   # always take the best action
    next_state, reward, done = step(state, action)
    total_reward += reward
    path.append(next_state)
    state = next_state
    if done:
        break

print(f"Path taken: {path}")
print(f"Steps to goal: {len(path) - 1}")
print(f"Total reward: {total_reward:.2f}")

if state == GOAL_STATE:
    print("Agent reached the goal!")
elif state in HOLES:
    print("Agent fell in a hole - try training longer or tuning hyperparameters.")