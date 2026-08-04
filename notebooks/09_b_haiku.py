import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

# ── Environment (same 4x4 gridworld) ─────────────────────────────────────────
# %%
GRID_SIZE = 4
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4
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
    else: return next_state, -0.05, False

def one_hot(state):
    return jnp.zeros(NUM_STATES).at[state].set(1.0)

print("Environment ready.")


# ── Section 1: Define Networks with Haiku ────────────────────────────────────
# %%
# Compare this to Days 4-8. No manual weight dicts. No init_actor() functions.
# Just describe what the network does, and Haiku handles the rest.

def actor_fn(x):
    return hk.Sequential([
        hk.Linear(64),
        jax.nn.tanh,
        hk.Linear(NUM_ACTIONS),
        jax.nn.softmax,
    ])(x)

def critic_fn(x):
    return hk.Sequential([
        hk.Linear(64),
        jax.nn.tanh,
        hk.Linear(1),
    ])(x).squeeze()  # output a single number

# hk.transform converts these into (init, apply) pairs
actor  = hk.without_apply_rng(hk.transform(actor_fn))
critic = hk.without_apply_rng(hk.transform(critic_fn))

# initialize: this creates the parameter dictionaries
key = jax.random.PRNGKey(42)
key, k1, k2 = jax.random.split(key, 3)
sample = one_hot(0)

actor_params  = actor.init(k1, sample)
critic_params = critic.init(k2, sample)

print("Networks defined with Haiku.")
print(f"\nActor params structure:")
for layer, weights in actor_params.items():
    for name, array in weights.items():
        print(f"  {layer}/{name}: shape {array.shape}")

print(f"\nSample actor output: {actor.apply(actor_params, sample)}")
print(f"Sample critic output: {critic.apply(critic_params, sample):.4f}")


# ── Section 2: Same Actor-Critic Algorithm, Cleaner Code ─────────────────────
# %%
# Everything from here is identical to Day 7 in logic.
# The only difference is actor.apply(params, x) instead of actor_forward(params, x).

def collect_episode(actor_params, rng_key, max_steps=100):
    states, actions, rewards = [], [], []
    state = START_STATE
    for _ in range(max_steps):
        probs = actor.apply(actor_params, one_hot(state))
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
print(f"\nSample episode: {len(s)} steps, reward: {sum(r):.2f}")


# ── Section 3: Loss Functions ─────────────────────────────────────────────────
# %%
def actor_loss_fn(actor_params, critic_params, states, actions, returns):
    total = 0.0
    for state, action, G in zip(states, actions, returns):
        x = one_hot(state)
        probs = actor.apply(actor_params, x)
        log_prob = jnp.log(probs[action] + 1e-8)
        advantage = G - critic.apply(critic_params, x)
        total += -advantage * log_prob
    return total / len(states)

def critic_loss_fn(critic_params, states, returns):
    total = 0.0
    for state, G in zip(states, returns):
        value = critic.apply(critic_params, one_hot(state))
        total += (G - value) ** 2
    return total / len(states)

actor_grad_fn  = jax.jit(jax.value_and_grad(actor_loss_fn))
critic_grad_fn = jax.jit(jax.value_and_grad(critic_loss_fn))

print("Loss functions ready.")


# ── Section 4: Training Loop ──────────────────────────────────────────────────
# %%
actor_opt  = optax.adam(3e-3)
critic_opt = optax.adam(1e-2)
actor_opt_state  = actor_opt.init(actor_params)
critic_opt_state = critic_opt.init(critic_params)

recent_rewards = []

for episode in range(3000):
    key, subkey = jax.random.split(key)
    states, actions, rewards = collect_episode(actor_params, subkey)
    returns = compute_returns(rewards)

    c_loss, c_grads = critic_grad_fn(critic_params, states, returns)
    c_updates, critic_opt_state = critic_opt.update(c_grads, critic_opt_state)
    critic_params = optax.apply_updates(critic_params, c_updates)

    a_loss, a_grads = actor_grad_fn(actor_params, critic_params, states, actions, returns)
    a_updates, actor_opt_state = actor_opt.update(a_grads, actor_opt_state)
    actor_params = optax.apply_updates(actor_params, a_updates)

    recent_rewards.append(sum(rewards))

    if (episode + 1) % 500 == 0:
        avg = np.mean(recent_rewards[-100:])
        print(f"Episode {episode + 1:4d} | avg reward: {avg:+.2f} | "
              f"actor loss: {float(a_loss):.4f} | critic loss: {float(c_loss):.4f}")

print("\nTraining complete.")


# ── Section 5: Visualize Policy ───────────────────────────────────────────────
# %%
ACTION_SYMBOLS = {0: "^", 1: "v", 2: "<", 3: ">"}

print("\nLearned policy (Haiku Actor-Critic):")
for row in range(GRID_SIZE):
    print("  ", end="")
    for col in range(GRID_SIZE):
        state = row * GRID_SIZE + col
        if state == GOAL_STATE: print("  G ", end="")
        elif state in HOLES: print("  H ", end="")
        else:
            probs = actor.apply(actor_params, one_hot(state))
            best = int(jnp.argmax(probs))
            print(f"  {ACTION_SYMBOLS[best]} ", end="")
    print()

print("\nCritic value estimates:")
for row in range(GRID_SIZE):
    print("  ", end="")
    for col in range(GRID_SIZE):
        state = row * GRID_SIZE + col
        if state == GOAL_STATE: print("  G   ", end="")
        elif state in HOLES: print("  H   ", end="")
        else:
            v = float(critic.apply(critic_params, one_hot(state)))
            print(f" {v:+.2f}", end="")
    print()


# ── Section 6: Watch the Agent Navigate ───────────────────────────────────────
# %%
print("\nWatching Haiku Actor-Critic agent navigate:")
state = START_STATE
path = [state]
total_reward = 0.0

for _ in range(50):
    probs = actor.apply(actor_params, one_hot(state))
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
if state == GOAL_STATE: print("Agent reached the goal!")
elif state in HOLES: print("Fell in a hole - try more episodes.")


# ── Section 7: Raw JAX vs Haiku - Side by Side ────────────────────────────────
# %%
print("""
Raw JAX (Days 4-8) vs Haiku (Day 9):

RAW JAX:
    def init_actor(key):
        k1, k2 = jax.random.split(key)
        return {
            "w1": jax.random.normal(k1, (16, 64)) * 0.1,
            "b1": jnp.zeros(64),
            "w2": jax.random.normal(k2, (64, 4)) * 0.1,
            "b2": jnp.zeros(4),
        }
    def actor_forward(params, x):
        h = jnp.tanh(x @ params["w1"] + params["b1"])
        return jax.nn.softmax(h @ params["w2"] + params["b2"])

HAIKU:
    def actor_fn(x):
        return hk.Sequential([hk.Linear(64), jax.nn.tanh,
                               hk.Linear(4), jax.nn.softmax])(x)
    actor = hk.without_apply_rng(hk.transform(actor_fn))

Same result. Half the code. No manual weight shapes.
This is why DeepMind code looks the way it does.
""")