# Day 7: Actor-Critic - Fixing REINFORCE's Big Problem

## The problem with REINFORCE

REINFORCE works. But it is noisy.

Here is why. REINFORCE plays a full episode, then uses the total return to update. That return depends on everything that happened - some good decisions, some bad ones, maybe some bad luck too. One unlucky episode gives you a terrible return even if you made mostly good decisions. One lucky episode gives you a great return even if you made bad ones.

The signal is too noisy. Training is slow because the network is getting confused by all this noise.

This is called high variance. It is the main weakness of REINFORCE.

---

## The fix: give the agent a sense of "normal"

Imagine you are a basketball player. You score 15 points in a game.

Is that good or bad? Depends. If you normally score 10, it is great. If you normally score 25, it is a bad game.

The number only makes sense relative to what is normal for you.

REINFORCE does not have this. It sees "return = 0.7" and tries to update. But is 0.7 good or bad for this state? It has no idea. It just treats every positive return as good and every negative return as bad.

Actor-Critic adds a baseline: a second network that estimates "what return do I normally get from this state?" Now instead of updating on the raw return, we update on:

```
advantage = actual return - expected return
         = "how much better (or worse) than normal was this?"
```

If the return was better than expected: reinforce the action. If worse than expected: push away from it. Now the signal is sharper and less noisy.

---

## Two networks, two jobs

**The Actor** - same as REINFORCE. A network that outputs action probabilities. It decides what to do.

**The Critic** - a new network. It takes a state and outputs a single number: the estimated value of that state. It is basically asking "how good is it to be here?"

```
Actor:   state -> [0.1, 0.3, 0.2, 0.4]   <- probabilities over actions
Critic:  state -> 0.72                    <- estimated value of this state
```

The critic does not decide actions. It just watches and judges. Its job is to give the actor a better signal to learn from.

---

## The advantage: the key number

Advantage = actual return - critic's estimate

```python
advantage = G_t - critic(state)
```

- Advantage is positive: this action did better than expected. Do it more.
- Advantage is negative: this action did worse than expected. Do it less.
- Advantage is zero: this action was exactly average. No change needed.

This is much cleaner than raw returns. The noise from lucky/unlucky episodes gets subtracted out.

---

## How the two networks update

**Actor loss** (same as REINFORCE, but now uses advantage instead of raw return):
```python
actor_loss = -advantage * log(probability of action taken)
```

**Critic loss** (simple: how wrong was the value estimate?):
```python
critic_loss = (G_t - critic(state)) ** 2
```

The critic is trained like a supervised regression problem. It just needs to get better at predicting returns. The actor uses the critic's estimates to get a cleaner learning signal.

Both networks update every episode. They improve together.

---

## The full algorithm

```
Step 1: Play a full episode using the actor network.

Step 2: Compute returns G_t for each step (same as REINFORCE).

Step 3: For each step:
        advantage = G_t - critic(state)
        actor_loss = -advantage * log_prob
        critic_loss = advantage ** 2

Step 4: Update both networks.
```

The only real addition from REINFORCE: one extra network (the critic) and one subtraction (advantage instead of raw return).

---

## Why does this matter?

Actor-Critic is not just an improvement over REINFORCE. It is the architecture that almost everything modern is built on.

PPO (the algorithm that trained ChatGPT): actor-critic.
A3C (DeepMind's async training algorithm): actor-critic.
SAC (for robotics): actor-critic.
GRPO (DeepSeek-R1): a clever variant where the critic is replaced by comparing within a group of episodes.

Once you understand actor-critic, you have the skeleton of every modern RL algorithm. The differences between them are mostly about how they compute the advantage and how they stabilize training.

Day 8: we go hands-on with Haiku (DeepMind's neural network library). Same algorithms, much cleaner code.