import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

# ── Environment (same gridworld as Days 4 and 5) ──────────────────────────────
# %%
GRID_SIZE = 4
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4  # 0=up, 1=down, 2=left, 3=right
START_STATE = 0
GOAL_STATE = 15
HOLES = {5, 7, 11, 12}

MOVES = {0: -GRID_SIZE, 1: GRID_SIZE, 2: -1, 3: 1}

def env_step(state, action):
    row, col = divmod(state, GRID_SIZE)
    if action == 0 and row == 0: next_state = state
    elif action == 1 and row == GRID_SIZE - 1: next_state = state
    elif action == 2 and col == 0: next_state = state
    elif action == 3 and col == GRID_SIZE - 1: next_state = state
    else: next_state = state + MOVES[action]

    if next_state == GOAL_STATE:
        return next_state, 1.0, True
    elif next_state in HOLES:
        return next_state, -1.0, True
    else:
        return next_state, 0.0, False

def one_hot(state):
    return jnp.zeros(NUM_STATES).at[state].set(1.0)

print("Environment ready: same 4x4 gridworld as Days 4 and 5.")


# ── Section 1: Policy Network : ──────────────────────────────────────────────────
# %%
# The policy network outputs probabilities over actions (softmax output)
# This is the key difference from DQN - DQN output Q-values, this outputs probs

key = jax.random.PRNGKey(42)

def init_policy(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (NUM_STATES, 32)) * 0.1,
        "b1": jnp.zeros(32),
        "w2": jax.random.normal(k2, (32, NUM_ACTIONS)) * 0.1,
        "b2": jnp.zeros(NUM_ACTIONS),
    }

def forward(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    logits = h @ params["w2"] + params["b2"]
    return jax.nn.softmax(logits)  # probabilities, not Q-values

params = init_policy(key)
test_probs = forward(params, one_hot(0))
print(f"\nPolicy network initialized.")
print(f"Output for state 0: {test_probs}")
print(f"Sum of probabilities: {jnp.sum(test_probs):.4f}  (should be 1.0)")
print(f"Actions: 0=up, 1=down, 2=left, 3=right")


# ── Section 2: Collecting an Episode ─────────────────────────────────────────
# %%
# REINFORCE needs a complete episode before it can update.
# We sample actions from the probability distribution.

def collect_episode(params, rng_key, max_steps=100):
    states, actions, rewards = [], [], []
    state = START_STATE
    for _ in range(max_steps):
        probs = forward(params, one_hot(state))
        # sample from distribution (not argmax - that's the key difference)
        rng_key, subkey = jax.random.split(rng_key)
        action = int(jax.random.choice(subkey, NUM_ACTIONS, p=probs))
        next_state, reward, done = env_step(state, action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if done:
            break
    return states, actions, rewards

key, subkey = jax.random.split(key)
s, a, r = collect_episode(params, subkey)
print(f"\nSample episode with untrained policy:")
print(f"Length: {len(s)} steps")
print(f"Total reward: {sum(r):.1f}")
print(f"Final state: {s[-1]}  (15 = goal, holes = {HOLES})")


# ── Section 3: Computing Discounted Returns ───────────────────────────────────
# %%
# G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
# Computed backwards from the end of the episode.

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = jnp.array(returns)
    # normalize returns - reduces variance, stabilizes training
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

sample_rewards = [0.0, 0.0, 0.0, 1.0]  # reached goal on step 4
sample_returns = compute_returns(sample_rewards)
print(f"\nReturn computation example:")
print(f"Rewards:  {sample_rewards}")
print(f"Returns:  {[f'{g:.3f}' for g in sample_returns]}")
print(f"Note: earlier steps get credit too (discounted)")


# ── Section 4: REINFORCE Loss and Gradient Update ─────────────────────────────
# %%
# loss = -sum(G_t * log π(a_t | s_t))
# Negative because JAX minimizes - we want to maximize expected return.

def reinforce_loss(params, states, actions, returns):
    total_loss = 0.0
    for state, action, G in zip(states, actions, returns):
        probs = forward(params, one_hot(state))
        log_prob = jnp.log(probs[action] + 1e-8)  # +1e-8 avoids log(0)
        total_loss += -G * log_prob  # negative: minimize loss = maximize return
    return total_loss / len(states)

grad_fn = jax.jit(jax.value_and_grad(reinforce_loss))

optimizer = optax.adam(learning_rate=3e-3)
opt_state = optimizer.init(params)

print(f"\nREINFORCE setup complete.")
print(f"Optimizer: Adam, lr=3e-3")
print(f"Loss = -G_t * log π(a_t | s_t) averaged over episode")


# ── Section 5: Training Loop ──────────────────────────────────────────────────
# %%
NUM_EPISODES = 3000
GAMMA = 0.99
PRINT_EVERY = 500

recent_rewards = []

for episode in range(NUM_EPISODES):
    key, subkey = jax.random.split(key)
    states, actions, rewards = collect_episode(params, subkey)
    returns = compute_returns(rewards, GAMMA)

    loss, grads = grad_fn(params, states, actions, returns)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    total_reward = sum(rewards)
    recent_rewards.append(total_reward)

    if (episode + 1) % PRINT_EVERY == 0:
        avg = np.mean(recent_rewards[-100:])
        print(f"Episode {episode + 1:4d} | avg reward (last 100): {avg:+.2f} | loss: {float(loss):.4f}")

print("\nTraining complete.")


# ── Section 6: Visualize Learned Policy ───────────────────────────────────────
# %%
ACTION_SYMBOLS = {0: "^", 1: "v", 2: "<", 3: ">"}

print("\nLearned policy from REINFORCE (most probable action per state):")
for row in range(GRID_SIZE):
    print("  ", end="")
    for col in range(GRID_SIZE):
        state = row * GRID_SIZE + col
        if state == GOAL_STATE:
            print("  G ", end="")
        elif state in HOLES:
            print("  H ", end="")
        else:
            probs = forward(params, one_hot(state))
            best = int(jnp.argmax(probs))
            print(f"  {ACTION_SYMBOLS[best]} ", end="")
    print()


# ── Section 7: Watch the Agent Navigate ───────────────────────────────────────
# %%
print("\nWatching trained REINFORCE agent navigate (greedy):")
state = START_STATE
path = [state]
total_reward = 0.0

for _ in range(50):
    probs = forward(params, one_hot(state))
    action = int(jnp.argmax(probs))  # greedy at test time
    next_state, reward, done = env_step(state, action)
    total_reward += reward
    path.append(next_state)
    state = next_state
    if done:
        break

print(f"Path:         {path}")
print(f"Steps:        {len(path) - 1}")
print(f"Total reward: {total_reward:.2f}")

if state == GOAL_STATE:
    print("REINFORCE agent reached the goal!")
elif state in HOLES:
    print("Agent fell in a hole - try more episodes or tune lr.")


# ── Section 8: REINFORCE vs DQN Comparison ────────────────────────────────────
# %%
print("""
REINFORCE vs DQN - side by side:

                    DQN                         REINFORCE
Network output:     Q-value per action          Probability per action
Action selection:   argmax (greedy)             sample from distribution
Update signal:      Bellman target              Episode return G_t
When to update:     Every step (replay buffer)  End of episode only
Works with:         Discrete actions only       Discrete + continuous
Variance:           Low (one-step TD)           High (Monte Carlo)
Sample efficiency:  Higher                      Lower

Both solve the same gridworld. But REINFORCE is the foundation
of everything modern: PPO, A3C, SAC, and GRPO all build on this.
""")