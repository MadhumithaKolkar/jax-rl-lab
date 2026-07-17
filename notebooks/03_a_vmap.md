# vmap - Vectorising Your Functions

You have learned three JAX superpowers: GPU acceleration, `grad`, and `jit`.
`vmap` is the fourth. It is the one that makes JAX genuinely different from everything else.

---

## The Problem vmap Solves

Imagine you have a function that works on a single input.

```python
def score(x):
    return jnp.sum(x ** 2)
```

Now you have a batch of 100 inputs and you want to run this function on all of them.

The naive way is a Python loop:

```python
results = []
for x in batch:
    results.append(score(x))
```

This is slow. Python loops are slow. And in JAX, loops also break `jit` compilation in ways that hurt you later.

The PyTorch way is to rewrite your function to handle batches explicitly - you reshape tensors,
add batch dimensions, make sure every operation broadcasts correctly. This is tedious and error-prone.

The JAX way is `vmap`.

---

## What vmap Does

`vmap` takes a function that works on a single input and returns a new function that works on
a whole batch, automatically, without you changing the original function at all.

```python
import jax

def score(x):
    return jnp.sum(x ** 2)

# vmap transforms score into a batched version
batch_score = jax.vmap(score)

# now feed it a whole batch at once
batch = jnp.ones((100, 10))   # 100 inputs, each of size 10
results = batch_score(batch)   # shape: (100,) - one result per input
```

You wrote `score` thinking about one input. JAX handles the batching for you.

---

## The Key Mental Model

Think of it this way:

- `jit` makes your function faster
- `grad` makes your function differentiable
- `vmap` makes your function work on batches

And all three compose together:

```python
# the most common pattern in real JAX code:
fast_batched_grad = jax.jit(jax.vmap(jax.grad(my_loss_fn)))
```

One line. Your loss function is now: differentiable, batched across all examples at once, and compiled.
This is the pattern you will see in every JAX research codebase including DeepMind's.

---

## Why Not Just Use a Loop?

Three reasons:

**1. Speed.** vmap does not actually loop. It vectorises - it runs all inputs in parallel as one
single array operation on the hardware. It is much faster than a Python loop.

**2. Composability.** vmap + jit + grad compose cleanly. A Python loop inside a jitted function
causes problems (JAX cannot always trace through Python control flow).

**3. Clean code.** You write simple single-input functions. vmap handles the rest.
Your functions stay readable and testable on one example at a time.

---

## A Real Example - Computing Gradients for a Whole Batch

Without vmap, to get gradients for each example in a batch you would loop:

```python
grads = [jax.grad(loss)(w, x, y) for x, y in zip(xs, ys)]  # slow loop
```

With vmap:

```python
# loss takes a single (x, y) pair
def loss(w, x, y):
    return (jnp.dot(w, x) - y) ** 2

# grad w.r.t. first argument
grad_loss = jax.grad(loss)

# vmap over x and y (the 2nd and 3rd arguments), not over w
batch_grad = jax.vmap(grad_loss, in_axes=(None, 0, 0))

# now run on the whole batch at once
grads = batch_grad(w, xs, ys)  # shape: (batch_size, w_dim)
```

`in_axes=(None, 0, 0)` means: do not batch over `w` (it is shared), batch over the 0th axis
of `xs` and `ys`.

---

## Summary

| What you want | How you get it |
|---|---|
| Run on GPU | JAX does it automatically |
| Compute gradients | `jax.grad(fn)` |
| Compile and speed up | `jax.jit(fn)` |
| Run on a whole batch without a loop | `jax.vmap(fn)` |
| All three at once | `jax.jit(jax.vmap(jax.grad(fn)))` |

vmap is the reason JAX code looks so clean compared to PyTorch.
You write simple functions. JAX scales them.