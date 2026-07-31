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
    else: return next_state, -0.05, False  # step penalty built into env

def one_hot(state):
    return jnp.zeros(NUM_STATES).at[state].set(1.0)

print("Environment: same 4x4 gridworld, step penalty of -0.05 built in.")


# ── Section 1: Actor and Critic Networks ─────────────────────────────────────
# %%
key = jax.random.PRNGKey(42)

def init_actor(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (NUM_STATES, 64)) * 0.1,
        "b1": jnp.zeros(64),
        "w2": jax.random.normal(k2, (64, NUM_ACTIONS)) * 0.1,
        "b2": jnp.zeros(NUM_ACTIONS),
    }

def init_critic(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (NUM_STATES, 64)) * 0.1,
        "b1": jnp.zeros(64),
        "w2": jax.random.normal(k2, (64, 1)) * 0.1,
        "b2": jnp.zeros(1),
    }

def actor_forward(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    logits = h @ params["w2"] + params["b2"]
    return jax.nn.softmax(logits)

def critic_forward(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return (h @ params["w2"] + params["b2"])[0]

key, k1, k2 = jax.random.split(key, 3)
actor_params = init_actor(k1)
critic_params = init_critic(k2)

print(f"\nNetworks initialized.")
print(f"Actor output (probs): {actor_forward(actor_params, one_hot(0))}")
print(f"Critic output (value): {critic_forward(critic_params, one_hot(0)):.4f}")


# ── Section 2: Collecting a Batch of Episodes ────────────────────────────────
# %%
# PPO collects multiple full episodes before updating.
# This gives a more stable, diverse batch of experience.

def collect_batch(actor_params, critic_params, rng_key, num_episodes=10):
    all_states, all_actions, all_returns, all_advantages, all_log_probs = [], [], [], [], []

    for _ in range(num_episodes):
        states, actions, rewards, log_probs = [], [], [], []
        state = START_STATE

        for _ in range(100):
            x = one_hot(state)
            probs = actor_forward(actor_params, x)
            rng_key, subkey = jax.random.split(rng_key)
            action = int(jax.random.choice(subkey, NUM_ACTIONS, p=probs))
            log_prob = float(jnp.log(probs[action] + 1e-8))

            next_state, reward, done = env_step(state, action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            if done:
                break

        # compute returns
        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        # compute advantages = return - critic estimate
        advantages = [
            G - float(critic_forward(critic_params, one_hot(s)))
            for G, s in zip(returns, states)
        ]

        all_states.extend(states)
        all_actions.extend(actions)
        all_returns.extend(returns)
        all_advantages.extend(advantages)
        all_log_probs.extend(log_probs)

    # normalize advantages across the whole batch
    adv = jnp.array(all_advantages)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return (all_states, all_actions,
            jnp.array(all_returns), adv,
            jnp.array(all_log_probs))

key, subkey = jax.random.split(key)
states, actions, returns, advantages, old_log_probs = collect_batch(
    actor_params, critic_params, subkey, num_episodes=5
)
print(f"\nSample batch: {len(states)} total steps from 5 episodes")
print(f"Advantage range: [{float(advantages.min()):.2f}, {float(advantages.max()):.2f}]")


# ── Section 3: The PPO Clipped Loss ──────────────────────────────────────────
# %%
# This is the core of PPO.
#
# ratio = new_prob / old_prob  (how much has the policy changed for this action?)
# clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)  (cap the change)
# actor_loss = -min(ratio * advantage, clipped_ratio * advantage)
#
# The min() means: if the update is too large, the clipped version kicks in
# and the gradient effectively becomes zero. The policy stops updating further.

EPSILON = 0.2  # max allowed change per update: policy can shift by +-20%

def ppo_actor_loss(actor_params, states, actions, advantages, old_log_probs):
    total_loss = 0.0
    total_entropy = 0.0

    for state, action, adv, old_lp in zip(states, actions, advantages, old_log_probs):
        probs = actor_forward(actor_params, one_hot(state))
        new_log_prob = jnp.log(probs[action] + 1e-8)

        ratio = jnp.exp(new_log_prob - old_lp)  # new_prob / old_prob

        # clipped surrogate loss
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1 - EPSILON, 1 + EPSILON) * adv
        policy_loss = -jnp.minimum(unclipped, clipped)

        # entropy bonus: reward the agent for staying exploratory
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

        total_loss += policy_loss
        total_entropy += entropy

    n = len(states)
    return total_loss / n - 0.01 * (total_entropy / n)  # subtract entropy to maximize it

def ppo_critic_loss(critic_params, states, returns):
    total = 0.0
    for state, G in zip(states, returns):
        value = critic_forward(critic_params, one_hot(state))
        total += (G - value) ** 2
    return total / len(states)

actor_grad_fn = jax.jit(jax.value_and_grad(ppo_actor_loss))
critic_grad_fn = jax.jit(jax.value_and_grad(ppo_critic_loss))

print(f"\nPPO loss functions ready.")
print(f"Epsilon (clip): {EPSILON}  (policy can change by at most 20% per update)")
print(f"Entropy weight: 0.01  (small bonus for staying exploratory)")


# ── Section 4: Training Loop ──────────────────────────────────────────────────
# %%
# PPO's training loop has two nested loops:
# Outer: collect a fresh batch of episodes
# Inner: run multiple gradient updates (epochs) on that same batch
# This is what makes PPO sample-efficient

NUM_ITERATIONS = 200   # outer loop: collect new batch each iteration
EPISODES_PER_BATCH = 10
PPO_EPOCHS = 4         # inner loop: reuse each batch 4 times

actor_optimizer = optax.adam(3e-4)
critic_optimizer = optax.adam(1e-3)
actor_opt_state = actor_optimizer.init(actor_params)
critic_opt_state = critic_optimizer.init(critic_params)

recent_rewards = []

for iteration in range(NUM_ITERATIONS):
    key, subkey = jax.random.split(key)
    states, actions, returns, advantages, old_log_probs = collect_batch(
        actor_params, critic_params, subkey, num_episodes=EPISODES_PER_BATCH
    )

    # track raw reward before normalizing
    recent_rewards.append(float(returns.mean()))

    # PPO epochs: update multiple times on the same batch
    for epoch in range(PPO_EPOCHS):
        a_loss, a_grads = actor_grad_fn(
            actor_params, states, actions, advantages, old_log_probs
        )
        a_updates, actor_opt_state = actor_optimizer.update(a_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, a_updates)

        c_loss, c_grads = critic_grad_fn(critic_params, states, returns)
        c_updates, critic_opt_state = critic_optimizer.update(c_grads, critic_opt_state)
        critic_params = optax.apply_updates(critic_params, c_updates)

    if (iteration + 1) % 25 == 0:
        avg = np.mean(recent_rewards[-20:])
        print(f"Iter {iteration + 1:3d} | avg return: {avg:+.3f} | "
              f"actor loss: {float(a_loss):.4f} | critic loss: {float(c_loss):.4f}")

print("\nTraining complete.")


# ── Section 5: Visualize Policy ───────────────────────────────────────────────
# %%
ACTION_SYMBOLS = {0: "^", 1: "v", 2: "<", 3: ">"}

print("\nLearned policy from PPO:")
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

print("\nCritic value estimates:")
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


# ── Section 6: Watch the Agent Navigate ───────────────────────────────────────
# %%
print("\nWatching PPO agent navigate:")
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

if state == GOAL_STATE: print("PPO agent reached the goal!")
elif state in HOLES: print("Fell in a hole - try more iterations.")


# ── Section 7: What PPO Added Over Actor-Critic ───────────────────────────────
# %%
print("""
What PPO added over Actor-Critic:

1. Clipping
   ratio = new_prob / old_prob
   loss  = -min(ratio * adv, clip(ratio, 0.8, 1.2) * adv)
   -> policy cannot change more than 20% per update
   -> no more training collapse from one bad episode

2. Multiple epochs per batch
   same data, 4 gradient updates instead of 1
   -> more learning per episode collected
   -> more sample efficient

3. Entropy bonus
   loss -= 0.01 * entropy
   -> agent stays exploratory, does not collapse to one action too early

Result: stable training, rarely collapses, works on almost any RL problem.
This is why PPO became the default algorithm for RLHF (training ChatGPT).
""")