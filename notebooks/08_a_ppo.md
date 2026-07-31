# Day 8: PPO - The Algorithm Behind ChatGPT

## Where we are

Day 7 ended with Actor-Critic. Two networks: an actor that picks actions, a critic that judges states. The actor uses the critic's estimates (advantage) to get a cleaner learning signal.

It worked. But it had one big problem.

---

## The problem Actor-Critic left unsolved

In Actor-Critic, after each episode you update the network weights. Sometimes those updates are huge. A single lucky or unlucky episode could push the weights so far in one direction that the policy completely changes.

Imagine you are learning to cook. You make a great dish once. Someone tells you to "do more of what you just did" so aggressively that you throw out your entire recipe book and only cook that one dish forever. That is too much of a good thing.

In RL terms: a big gradient update can destroy a policy that was previously working well. The agent unlearns things it already knew. Training collapses.

PPO solves this with one idea: **do not let the policy change too much in a single update.**

---

## The clipping trick - PPO's core idea

PPO compares the new policy to the old policy. It asks: for this action in this state, how much more (or less) likely is it under the new policy vs the old one?

```
ratio = new_policy_probability / old_policy_probability
```

If ratio = 1.0: the policy has not changed for this action.
If ratio = 1.5: the new policy is 50% more likely to take this action.
If ratio = 0.5: the new policy is 50% less likely to take this action.

PPO then clips this ratio. It says: **I will not let this ratio go above 1.2 or below 0.8** (using epsilon = 0.2).

```python
clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
                           0.8            1.2
```

The final loss uses whichever is smaller - the clipped or unclipped version:

```python
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

This one line is PPO. It lets the policy improve when the advantage is positive, but puts a hard cap on how much it can change in one step.

---

## Why does clipping help?

Think of it as a speed limit for learning.

Without clipping: the gradient says "this action was great, make it 10x more likely." Policy overshoots, collapses, training restarts from scratch.

With clipping: the gradient still says "this action was great." But PPO says "you can make it at most 20% more likely this update. Come back next episode and nudge it again if you want."

Small, stable steps. Every update. Training stays on track.

---

## The full PPO algorithm

```
Step 1: Collect a batch of episodes using the current policy.
        (PPO collects more data per update than REINFORCE)

Step 2: Compute advantages using the critic.
        advantage = actual_return - critic_estimate

Step 3: Run multiple gradient updates on this same batch.
        (This is called "epochs" - PPO reuses data efficiently)
        Each update: compute the clipped loss, update the actor.

Step 4: Update the critic separately.

Step 5: Repeat from Step 1.
```

Two things PPO adds over Actor-Critic:
1. The clipping (the main idea)
2. Multiple epochs per batch - PPO squeezes more learning out of each set of episodes before collecting new ones

---

## The three losses in PPO

PPO's full loss has three parts:

```
total_loss = actor_loss + value_loss_weight * critic_loss - entropy_weight * entropy
```

**Actor loss:** the clipped policy gradient (the main PPO idea).

**Critic loss:** same as Actor-Critic, mean squared error between predicted and actual returns.

**Entropy bonus:** this one is new. Entropy measures how "spread out" the probability distribution is. High entropy = the agent considers many actions. Low entropy = the agent always picks the same action.

The entropy bonus rewards the agent for staying curious. Without it, the policy can collapse to always picking one action too early, before it has properly explored.

```python
entropy = -sum(prob * log(prob))  # high if probs are spread, low if one prob dominates
```

---

## PPO vs the algorithms we have seen

| | DQN | REINFORCE | Actor-Critic | PPO |
|---|---|---|---|---|
| Output | Q-values | Probabilities | Probabilities | Probabilities |
| Update | Every step | End of episode | End of episode | End of batch (multiple epochs) |
| Stability | Good | Poor | Medium | Good |
| Sample efficiency | High | Low | Medium | Medium-High |
| Used in | Games | Rarely in practice | Foundation | ChatGPT, robotics, everything |

---

## Why PPO became the standard

Three reasons:

1. Stable. The clipping means training rarely collapses.
2. Simple. The clipping idea is one line. Easier to implement and debug than alternatives.
3. General. Works for discrete actions (games) and continuous actions (robotics, LLM fine-tuning).

When OpenAI trained ChatGPT with RLHF (Reinforcement Learning from Human Feedback), the RL part was PPO. When DeepMind trains robots, they often use PPO. When researchers need a reliable RL baseline, they reach for PPO.

GRPO (the algorithm in your forge project) is a direct variant of PPO. Instead of one critic estimating advantage, it runs a group of episodes and computes relative performance within the group. Same clipping. Same stability. Different advantage computation.

Day 9: we rewrite all of this using Haiku, DeepMind's neural network library. The code becomes cleaner, more modular, and looks like actual research code.