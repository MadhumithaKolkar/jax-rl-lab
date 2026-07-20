# JAX RL Lab

A daily learning log for going from NumPy/PyTorch to JAX, and eventually into RL. Each day introduces one idea, explained simply first, then practiced in code.

Dev Blog Series : Learning RL and JAX in Public - from zero to DeepMind ! : https://dev.to/madhumithakolkar

## Structure

Each day lives in `notebooks/` as a pair of files:

```
NN_a_<topic>.md   -> the intro: plain-English explanation, analogies, small snippets
NN_b_<topic>.py   -> the practice: runnable code with # %% cells, exploring the same idea
```

- `NN` is a two-digit day/lesson prefix.
- `_a_` files are markdown write-ups, meant to be read before touching code.
- `_b_` files are Python scripts using `# %%` cell markers (run interactively, e.g. in VS Code's Jupyter mode), organized into commented `SECTION` blocks that build from a toy example up to the real pattern used in practice.

If both files for a day already exist, that lesson is done. Otherwise it's next up.

## Lessons so far

| Day | Intro | Code | Topic |
|---|---|---|---|
| 1 | [00_a_what_is_jax.md](notebooks/00_a_what_is_jax.md) | [01_b_arrays.py](notebooks/01_b_arrays.py) | What JAX is, how it differs from NumPy/PyTorch |
| 2 | [02_a_intro_grad_and_jit.md](notebooks/02_a_intro_grad_and_jit.md) | [02_b_grad_and_jit.py](notebooks/02_b_grad_and_jit.py) | Derivatives, gradients, `jax.grad`, `jax.jit` |
| 3 | [03_a_vmap.md](notebooks/03_a_vmap.md) | [03_b_vmap.py](notebooks/03_b_vmap.py) | Vectorising functions with `jax.vmap` |

## Setup

```bash
cd notebooks
python -m venv .venv
source .venv/bin/activate
pip install jax jaxlib ipython jupyter
```

## Goal

Build up JAX fundamentals (`grad`, `jit`, `vmap`, and beyond) as a foundation for implementing reinforcement learning algorithms in JAX.
