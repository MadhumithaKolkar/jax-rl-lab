import jax
import jax.numpy as jnp
import optax
import numpy as np

# ── Environment (same 4x4 gridworld) ─────────────────────────────────────────
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

    if next_state == GOAL_STATE: return next_state, 1.0, True
    elif next_state in HOLES: return next_state, -1.0, True
    else: return next_state, 0.0, False

def one_hot(state):
    return jnp.zeros(NUM_STATES).at[state].set(1.0)

print("Environment ready.")


# ── Section 1: Two Networks - Actor and Critic ────────────────────────────────
# %%
# Actor:  state -> action probabilities  (decides what to do)
# Critic: state -> single value number   (judges how good the state is)

key = jax.random.PRNGKey(0)

def init_actor(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (NUM_STATES, 32)) * 0.1,
        "b1": jnp.zeros(32),
        "w2": jax.random.normal(k2, (32, NUM_ACTIONS)) * 0.1,
        "b2": jnp.zeros(NUM_ACTIONS),
    }

def init_critic(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (NUM_STATES, 32)) * 0.1,
        "b1": jnp.zeros(32),
        "w2": jax.random.normal(k2, (32, 1)) * 0.1,  # single output
        "b2": jnp.zeros(1),
    }

def actor_forward(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    logits = h @ params["w2"] + params["b2"]
    return jax.nn.softmax(logits)  # probabilities

def critic_forward(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    value = h @ params["w2"] + params["b2"]
    return value[0]  # single number

key, k1, k2 = jax.random.split(key, 3)
actor_params = init_actor(k1)
critic_params = init_critic(k2)

test_probs = actor_forward(actor_params, one_hot(0))
test_value = critic_forward(critic_params, one_hot(0))

print(f"Actor output for state 0:  {test_probs}  (probabilities, sum={jnp.sum(test_probs):.2f})")
print(f"Critic output for state 0: {test_value:.4f}  (estimated value of this state)")


# ── Section 2: Collect Episode ────────────────────────────────────────────────
# %%
def collect_episode(actor_params, rng_key, max_steps=100):
    states, actions, rewards = [], [], []
    state = START_STATE
    for _ in range(max_steps):
        probs = actor_forward(actor_params, one_hot(state))
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

def compute_returns(rewards, gamma=0.99):
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return jnp.array(returns)

key, subkey = jax.random.split(key)
s, a, r = collect_episode(actor_params, subkey)
print(f"\nSample episode: {len(s)} steps, total reward: {sum(r):.1f}")


# ── Section 3: The Advantage - The Key Concept ────────────────────────────────
# %%
# Advantage = actual return - critic's estimate of the state
# Positive advantage: this was better than expected -> reinforce the action
# Negative advantage: this was worse than expected -> push away from the action

returns = compute_returns(r)
advantages = jnp.array([
    float(G) - float(critic_forward(critic_params, one_hot(s)))
    for G, s in zip(returns, s)
])

print(f"\nAdvantage examples from sample episode:")
for i in range(min(4, len(s))):
    print(f"  state={s[i]:2d}, return={float(returns[i]):+.3f}, "
          f"critic_est={float(critic_forward(critic_params, one_hot(s[i]))):+.3f}, "
          f"advantage={float(advantages[i]):+.3f}")
print("(critic estimates are random at this point - will improve during training)")


# ── Section 4: Actor and Critic Losses ────────────────────────────────────────
# %%
# Actor loss:  -advantage * log_prob  (same as REINFORCE but uses advantage)
# Critic loss: advantage^2            (how wrong was the value estimate?)

def actor_loss_fn(actor_params, critic_params, states, actions, returns):
    total = 0.0
    for state, action, G in zip(states, actions, returns):
        x = one_hot(state)
        probs = actor_forward(actor_params, x)
        log_prob = jnp.log(probs[action] + 1e-8)
        advantage = G - critic_forward(critic_params, x)
        total += -advantage * log_prob
    return total / len(states)

def critic_loss_fn(critic_params, states, returns):
    total = 0.0
    for state, G in zip(states, returns):
        x = one_hot(state)
        value = critic_forward(critic_params, x)
        total += (G - value) ** 2
    return total / len(states)

actor_grad_fn = jax.jit(jax.value_and_grad(actor_loss_fn))
critic_grad_fn = jax.jit(jax.value_and_grad(critic_loss_fn))

print("Actor and critic loss functions ready.")
print("Actor loss:  -advantage * log_prob")
print("Critic loss: (return - value_estimate)^2")


# ── Section 5: Training Loop ──────────────────────────────────────────────────
# %%
NUM_EPISODES = 3000
GAMMA = 0.99

actor_optimizer = optax.adam(3e-3)
critic_optimizer = optax.adam(1e-2)  # critic learns a bit faster
actor_opt_state = actor_optimizer.init(actor_params)
critic_opt_state = critic_optimizer.init(critic_params)

recent_rewards = []

for episode in range(NUM_EPISODES):
    key, subkey = jax.random.split(key)
    states, actions, rewards = collect_episode(actor_params, subkey)
    returns = compute_returns(rewards, GAMMA)

    # update critic first (so actor uses fresh estimates)
    c_loss, c_grads = critic_grad_fn(critic_params, states, returns)
    c_updates, critic_opt_state = critic_optimizer.update(c_grads, critic_opt_state)
    critic_params = optax.apply_updates(critic_params, c_updates)

    # update actor using advantage from updated critic
    a_loss, a_grads = actor_grad_fn(actor_params, critic_params, states, actions, returns)
    a_updates, actor_opt_state = actor_optimizer.update(a_grads, actor_opt_state)
    actor_params = optax.apply_updates(actor_params, a_updates)

    recent_rewards.append(sum(rewards))

    if (episode + 1) % 500 == 0:
        avg = np.mean(recent_rewards[-100:])
        print(f"Episode {episode + 1:4d} | avg reward (last 100): {avg:+.2f} | "
              f"actor loss: {float(a_loss):.4f} | critic loss: {float(c_loss):.4f}")

print("\nTraining complete.")


# ── Section 6: Visualize Policy ───────────────────────────────────────────────
# %%
ACTION_SYMBOLS = {0: "^", 1: "v", 2: "<", 3: ">"}

print("\nLearned policy from Actor-Critic:")
for row in range(GRID_SIZE):
    print("  ", end="")
    for col in range(GRID_SIZE):
        state = row * GRID_SIZE + col
        if state == GOAL_STATE: print("  G ", end="")
        elif state in HOLES: print("  H ", end="")
        else:
            probs = actor_forward(actor_params, one_hot(state))
            best = int(jnp.argmax(probs))
            print(f"  {ACTION_SYMBOLS[best]} ", end="")
    print()

print("\nCritic's value estimates (how good is each state?):")
for row in range(GRID_SIZE):
    print("  ", end="")
    for col in range(GRID_SIZE):
        state = row * GRID_SIZE + col
        if state == GOAL_STATE: print("  G   ", end="")
        elif state in HOLES: print("  H   ", end="")
        else:
            v = float(critic_forward(critic_params, one_hot(state)))
            print(f" {v:+.2f}", end="")
    print()
print("(higher value = critic thinks this state leads to better outcomes)")


# ── Section 7: Watch the Agent Navigate ───────────────────────────────────────
# %%
print("\nWatching trained Actor-Critic agent navigate:")
state = START_STATE
path = [state]
total_reward = 0.0

for _ in range(50):
    probs = actor_forward(actor_params, one_hot(state))
    action = int(jnp.argmax(probs))
    next_state, reward, done = env_step(state, action)
    total_reward += reward
    path.append(next_state)
    state = next_state
    if done:
        break

print(f"Path:         {path}")
print(f"Steps:        {len(path) - 1}")
print(f"Total reward: {total_reward:.2f}")

if state == GOAL_STATE: print("Actor-Critic agent reached the goal!")
elif state in HOLES: print("Fell in a hole - try more episodes.")


# ── Section 8: The Family Tree ────────────────────────────────────────────────
# %%
print("""
The Actor-Critic family tree:

Actor-Critic (Day 7)
    |
    ├── A3C  (DeepMind, 2016) - run many actor-critics in parallel
    |
    ├── PPO  (OpenAI, 2017)   - used to train ChatGPT
    |        adds a "clip" to stop updates from being too large
    |
    ├── SAC  (Berkeley, 2018) - adds entropy bonus for more exploration
    |        used heavily in robotics
    |
    └── GRPO (DeepSeek, 2024) - replaces the critic with group comparisons
             the algorithm behind DeepSeek-R1

All of them: an actor that picks actions + something that judges quality.
The differences are in HOW they compute the advantage and stabilize training.
""")