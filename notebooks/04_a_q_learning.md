# Q-Learning

This is your first reinforcement learning algorithm. Everything in modern RL - DQN, PPO, SAC -
builds on the ideas here. Understand this one deeply and the rest will follow naturally.

---

## What the Agent is Trying to Do

The agent lives in an environment. At each step:
1. It observes the current state
2. It picks an action
3. The environment gives it a reward and moves it to a new state
4. Repeat

The goal: learn a strategy (policy) that maximises total future reward.

---

## The Q-Value - One Sentence

**Q(s, a) = the total discounted reward the agent expects to collect if it takes action a in
state s, and then plays optimally from that point forward.**

It is a table. Rows are states. Columns are actions. Each cell is a number.
The agent looks up its current state, picks the action with the highest Q-value, done.

```
         action: LEFT   RIGHT   UP   DOWN
state 0:          1.2    3.4   0.8    2.1
state 1:          0.5    0.9   2.3    1.1
state 2:          4.0    1.2   3.1    0.7
...
```

At state 0, the agent picks RIGHT (3.4 is the highest). That is the policy.

---

## The Problem: We Do Not Know the Q-Values

At the start, the agent has no idea what Q(s, a) is for any state or action.
It has to learn them by exploring the environment and collecting rewards.

Q-learning is the algorithm that figures out the correct Q-values through experience.

---

## The Bellman Equation - The Core Idea

This is the equation Q-learning is built on. It says:

**The Q-value of taking action a in state s equals the immediate reward plus the best
Q-value available in the next state.**

```
Q(s, a) = reward + gamma * max(Q(next_state, all_actions))
```

- `reward` is what you got immediately
- `gamma` is the discount factor (usually 0.99) - future rewards are worth slightly less
- `max(Q(next_state, ...))` is the best possible Q-value from where you ended up

This is recursive. The value of where you are now depends on the value of where you go next.
Q-learning solves this recursion by iterating until the values stabilise.

---

## How Q-Learning Updates the Table

The agent starts with a Q-table full of zeros (or random numbers). It does not know anything.

At each step, it takes an action, sees what reward it gets, and uses the Bellman equation
to make a small correction to one cell in the table:

```
Q(s, a) = Q(s, a) + alpha * (target - Q(s, a))
```

Where:
- `target = reward + gamma * max(Q(next_state, ...))`  (what Bellman says it should be)
- `Q(s, a)` is what we currently think it is
- `alpha` is the learning rate (how big a correction to make each time, e.g. 0.1)

This is just: nudge the current estimate a little bit toward the target. Same idea as gradient
descent, but for a table instead of neural network weights.

---

## Exploration vs Exploitation - The Epsilon Problem

If the agent always picks the action with the highest Q-value, it never tries new things.
It might get stuck with a mediocre strategy because it never explored better options.

The solution is epsilon-greedy:
- With probability epsilon: pick a random action (explore)
- With probability 1 - epsilon: pick the best known action (exploit)

Epsilon starts high (lots of exploration at the start) and decays over time (exploit more
as the agent gets better). This is called epsilon decay.

---

## The Gridworld We Will Use

A 4x4 grid. The agent starts at the top-left (state 0). The goal is the bottom-right (state 15).
There are two holes. Fall in a hole and the episode ends with a negative reward.

```
 S  .  .  .
 .  H  .  H
 .  .  .  H
 H  .  .  G
```

S = start, G = goal, H = hole, . = safe cell

States: 0 to 15 (16 total, one per cell)
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

Rewards:
- Reach goal: +1.0
- Fall in hole: -1.0
- Every other step: -0.01 (small penalty to encourage finding the goal quickly)

The Q-table will be shape (16, 4) - 16 states, 4 actions.

---

## What You Will See When It Works

At the start the agent wanders randomly. It falls in holes, misses the goal.

After a few hundred episodes, the Q-table starts filling in with useful values.
The agent learns that cells near the goal have high Q-values and cells near holes have low ones.

After training, the agent walks a near-optimal path from start to goal every time.

---

## Summary

| Concept | What it means |
|---|---|
| Q(s, a) | Expected total future reward for taking action a in state s |
| Bellman equation | Q(s,a) = reward + gamma * max Q(next state) |
| Q-table | A grid of numbers, one per (state, action) pair |
| Update rule | Nudge Q(s,a) toward the Bellman target a little bit each step |
| Epsilon-greedy | Explore randomly sometimes, exploit best known action otherwise |
| Episode | One run from start to goal or hole - then reset and try again |