# Day 6: REINFORCE - Learning to Act Directly

## The gap DQN left open

DQN learns Q-values. From Q-values, it derives a policy: always pick the action with the highest Q-value. The policy is implicit - it's a side effect of the Q-values, not something the network learns directly.

This works for the gridworld. But it has two problems that come up in real research:

**Problem 1: The policy is always greedy.**
DQN has no natural way to say "I am 70% sure I should go right and 30% sure I should go down." It always picks one action with full confidence. Epsilon-greedy is a hack around this - random noise bolted on from outside.

**Problem 2: Continuous action spaces break Q-networks entirely.**
If an action is a steering angle between -180 and 180 degrees, you cannot output one Q-value per action - there are infinitely many actions. DQN has no answer for this. Policy gradients do.

---

## The core idea: parameterize the policy directly

Instead of learning Q(s, a) and deriving a policy from it, skip the middle step.

Learn the policy itself: a function that takes a state and outputs a probability distribution over actions.

```
DQN:        state -> Q-values -> argmax -> action
REINFORCE:  state -> action probabilities -> sample -> action
```

The policy is a neural network with parameters θ. We write it as π(a|s; θ).

Input: the current state. Output: probability of taking each action.

---

## The intuition behind the update rule

After running one full episode, you have a sequence of (state, action, reward) triples.

Look at what happened:
- Some actions led to high total return. Those were probably good choices.
- Some actions led to low or negative total return. Those were probably bad.

The update rule translates this directly:

- If an action led to high return: increase its probability.
- If an action led to low return: decrease its probability.
- Scale the update by how large the return was. A massive reward = a big update.

This is the entirety of REINFORCE. The math formalizes it, but the intuition is just: reward what worked, penalize what did not.

---

## The algorithm in 4 steps

```
Step 1: Run one full episode using the current policy.
        Collect (s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_T, a_T, r_T)

Step 2: Compute the discounted return G_t at each timestep.
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

Step 3: For each timestep, compute the REINFORCE loss:
        loss_t = -G_t * log π(a_t | s_t; θ)

Step 4: Sum the losses, backprop, update θ.
```

That is the full algorithm. One episode, one gradient update.

---

## The log probability trick - why log?

You want to maximize J(θ) = expected total return. But the expectation depends on which trajectories you visit, which depends on θ. Differentiating through that is circular.

The policy gradient theorem solves this:

```
∇J(θ) = E[ G_t * ∇ log π(a_t | s_t; θ) ]
```

Two things to unpack:

**Why log π?**
∇ log π(a|s; θ) tells you the direction to nudge θ to make action a more likely at state s.
If you multiply that by G_t (the return), you push θ harder toward actions that actually paid off.

**Why is this valid?**
There is a mathematical derivation (the log-derivative trick), but the intuition is: log π has the same gradient direction as π, and the log turns multiplication into addition - which makes the gradient computation much cleaner.

In code, this becomes:
```python
loss = -G_t * log_prob  # negative because JAX minimizes
```

Minimizing negative expected return = maximizing expected return.

---

## What changes compared to DQN

| | DQN | REINFORCE |
|---|---|---|
| What the network outputs | Q-value per action | Probability per action |
| How actions are selected | argmax (greedy) | sample from distribution |
| Update signal | Bellman target | Episode return G_t |
| Needs full episode? | No (one step) | Yes (Monte Carlo) |
| Handles continuous actions? | No | Yes (with modifications) |

Both run on the same gridworld. Both learn to reach the goal. The difference is in the mechanism.

---

## Why this matters

REINFORCE is the foundation. Every modern policy gradient algorithm - PPO, A3C, SAC - is a refinement of this core idea. The problems they solve (high variance, sample inefficiency, training instability) are all problems REINFORCE has that later algorithms fix.

Understanding REINFORCE means you understand the thing that all of them are improving on.

In Day 7, we look at one of those improvements: the baseline, which reduces variance without changing the expected gradient.