# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax                  # pip install optax  (JAX's optimiser library)
from collections import deque
import random

# %%
# -------------------------------------------------------
# SECTION 1 - The environment (same gridworld as Day 4)
# -------------------------------------------------------

GRID_SIZE = 4
NUM_STATES = 16
NUM_ACTIONS = 4
START_STATE = 0
GOAL_STATE = 15
HOLES = {5, 7, 11, 12}

REWARD_GOAL = 1.0
REWARD_HOLE = -1.0
REWARD_STEP = -0.01


def env_step(state, action):
    row, col = state // GRID_SIZE, state % GRID_SIZE
    if action == 0: row = max(0, row - 1)
    elif action == 1: row = min(GRID_SIZE - 1, row + 1)
    elif action == 2: col = max(0, col - 1)
    elif action == 3: col = min(GRID_SIZE - 1, col + 1)
    next_state = row * GRID_SIZE + col
    if next_state == GOAL_STATE:
        return next_state, REWARD_GOAL, True
    elif next_state in HOLES:
        return next_state, REWARD_HOLE, True
    return next_state, REWARD_STEP, False


def one_hot(state, num_states=NUM_STATES):
    """Convert a state integer to a one-hot vector for the network input."""
    return jnp.zeros(num_states).at[state].set(1.0)


# Quick check
print("State 3 as one-hot:", one_hot(3))
# Should be: [0, 0, 0, 1, 0, 0, ..., 0]

# %%
# -------------------------------------------------------
# SECTION 2 - The neural network (raw JAX, no Flax)
# -------------------------------------------------------

# Network: input(16) -> hidden(64) -> hidden(64) -> output(4)
# Params are just a dict of weight matrices and bias vectors

def init_params(key):
    """Initialise network weights using He initialisation."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    def he(k, fan_in, fan_out):
        std = jnp.sqrt(2.0 / fan_in)
        return jax.random.normal(k, (fan_in, fan_out)) * std

    return {
        "w1": he(k1, 16, 64),  "b1": jnp.zeros(64),
        "w2": he(k3, 64, 64),  "b2": jnp.zeros(64),
        "w3": he(k5, 64, 4),   "b3": jnp.zeros(4),
    }


def forward(params, x):
    """One forward pass. Returns Q-values for all 4 actions."""
    x = jax.nn.relu(x @ params["w1"] + params["b1"])
    x = jax.nn.relu(x @ params["w2"] + params["b2"])
    return x @ params["w3"] + params["b3"]   # no activation on output


# Test the network with a random state
key = jax.random.PRNGKey(42)
params = init_params(key)
test_input = one_hot(0)
q_values = forward(params, test_input)
print("Q-values for state 0 (random init):", q_values)
# Random numbers - the network knows nothing yet

# %%
# -------------------------------------------------------
# SECTION 3 - Experience Replay Buffer
# -------------------------------------------------------

class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.
    When full, old transitions are automatically discarded (deque).
    Training samples a random batch from this buffer.
    """

    def __init__(self, max_size=10_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            jnp.array([one_hot(s) for s in states]),
            jnp.array(actions),
            jnp.array(rewards, dtype=jnp.float32),
            jnp.array([one_hot(s) for s in next_states]),
            jnp.array(dones, dtype=jnp.float32),
        )

    def __len__(self):
        return len(self.buffer)


buffer = ReplayBuffer()
buffer.add(0, 1, -0.01, 4, False)
buffer.add(4, 3, -0.01, 5, True)
print(f"\nBuffer size: {len(buffer)}")

# %%
# -------------------------------------------------------
# SECTION 4 - Loss function and training step
# -------------------------------------------------------

GAMMA = 0.99


def compute_loss(online_params, target_params, batch):
    """
    DQN loss: mean squared error between Bellman target and online Q-values.

    target  = reward + gamma * max(target_network(next_state))   if not done
    target  = reward                                              if done
    loss    = mean( (target - online_network(state)[action]) ** 2 )
    """
    states, actions, rewards, next_states, dones = batch

    # Q-values from online network for current states
    online_q = jax.vmap(forward, in_axes=(None, 0))(online_params, states)

    # Best Q-values from target network for next states (used for Bellman target)
    target_q_next = jax.vmap(forward, in_axes=(None, 0))(target_params, next_states)
    best_next_q = jnp.max(target_q_next, axis=1)

    # Bellman target: if done, only use reward (no future)
    targets = rewards + GAMMA * best_next_q * (1.0 - dones)

    # Q-value of the action actually taken
    batch_size = states.shape[0]
    action_q = online_q[jnp.arange(batch_size), actions]

    loss = jnp.mean((targets - action_q) ** 2)
    return loss


# Compile the gradient function
grad_fn = jax.jit(jax.value_and_grad(compute_loss))


# %%
# -------------------------------------------------------
# SECTION 5 - Training loop
# -------------------------------------------------------

# Hyperparameters
BATCH_SIZE = 32
REPLAY_START = 200        # start training only after buffer has this many samples
TARGET_UPDATE_FREQ = 100  # copy online -> target every N steps
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997
NUM_EPISODES = 1500
LR = 1e-3

np.random.seed(0)
key = jax.random.PRNGKey(0)

online_params = init_params(key)
target_params = online_params   # start identical

optimizer = optax.adam(LR)
opt_state = optimizer.init(online_params)

replay_buffer = ReplayBuffer()
epsilon = EPSILON_START
total_steps = 0
episode_rewards = []

print("Training DQN...\n")

for episode in range(NUM_EPISODES):
    state = START_STATE
    total_reward = 0.0
    steps = 0

    while True:
        # epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(NUM_ACTIONS)
        else:
            q_vals = forward(online_params, one_hot(state))
            action = int(jnp.argmax(q_vals))

        next_state, reward, done = env_step(state, action)
        replay_buffer.add(state, action, reward, next_state, done)

        # only train once buffer has enough samples
        if len(replay_buffer) >= REPLAY_START:
            batch = replay_buffer.sample(BATCH_SIZE)
            loss, grads = grad_fn(online_params, target_params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            online_params = optax.apply_updates(online_params, updates)

        # copy online -> target every TARGET_UPDATE_FREQ steps
        total_steps += 1
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_params = online_params

        state = next_state
        total_reward += reward
        steps += 1

        if done or steps > 200:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    if (episode + 1) % 300 == 0:
        avg = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1:4d} | avg reward (last 100): {avg:.3f} | epsilon: {epsilon:.3f}")

# %%
# -------------------------------------------------------
# SECTION 6 - Watch the trained DQN agent run
# %%
# -------------------------------------------------------

ACTION_SYMBOLS = {0: "^", 1: "v", 2: "<", 3: ">"}

print("\nLearned policy from DQN:")
for row in range(GRID_SIZE):
    print("  ", end="")
    for col in range(GRID_SIZE):
        state = row * GRID_SIZE + col
        if state == GOAL_STATE:
            print("  G ", end="")
        elif state in HOLES:
            print("  H ", end="")
        else:
            q_vals = forward(online_params, one_hot(state))
            best = int(jnp.argmax(q_vals))
            print(f"  {ACTION_SYMBOLS[best]} ", end="")
    print()

print("\nWatching trained DQN agent navigate:")
state = START_STATE
path = [state]
total_reward = 0.0

for _ in range(50):
    q_vals = forward(online_params, one_hot(state))
    action = int(jnp.argmax(q_vals))
    next_state, reward, done = env_step(state, action)
    total_reward += reward
    path.append(next_state)
    state = next_state
    if done:
        break

print(f"Path: {path}")
print(f"Steps: {len(path) - 1}")
print(f"Total reward: {total_reward:.2f}")

if state == GOAL_STATE:
    print("DQN agent reached the goal!")
elif state in HOLES:
    print("Agent fell in a hole - try more episodes.")

# %%
# -------------------------------------------------------
# SECTION 7 - Q-table vs DQN: side by side comparison
# -------------------------------------------------------

print("\n--- What changed from Q-learning to DQN ---")
print()
print("Q-learning:")
print("  Q-values stored in : a table  (16 x 4 = 64 numbers)")
print("  Update method      : direct Bellman correction on one cell")
print("  Scales to          : small, discrete state spaces only")
print()
print("DQN:")
print("  Q-values stored in : a neural network")
print("  Update method      : gradient descent on MSE loss")
print("  Scales to          : large or continuous state spaces")
print("  Extra components   : replay buffer + target network")
print()
print("Same Bellman equation. Same epsilon-greedy. Same goal.")
print("The only difference is where the Q-values come from.")