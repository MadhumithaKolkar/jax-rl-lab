# What is JAX?

You already know NumPy. That library where you do `np.array([1,2,3])` and matrix math and `np.sum()`? JAX is that, but with three superpowers added on top.

---

## NumPy to JAX: the same syntax, then better

```python
# NumPy
import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(np.sum(x))   # 6.0

# JAX - identical syntax
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
print(jnp.sum(x))  # 6.0
```

Almost every NumPy function you know exists in JAX with the same name. That is intentional. The learning curve is not about new syntax, it is about three new ideas.

---

## The Three Superpowers

### 1. Run on GPU/TPU automatically

You do nothing special. JAX looks for a GPU, uses it if it is there, and falls back to CPU if not. No `.to(device)`, no `.cuda()`. It just works.

### 2. `grad` - automatic differentiation

JAX can compute the derivative (gradient) of *any* Python function you write. You write a loss function, JAX gives you its gradient. This is what PyTorch's `.backward()` does, but in JAX it is a standalone function you can apply to anything.

```python
import jax

def square(x):
    return x ** 2

derivative = jax.grad(square)
print(derivative(3.0))   # 6.0
# because d/dx of x^2 = 2x, and 2 * 3 = 6
```

### 3. `jit` - compile your code to run fast

By default, Python is slow. `jit` (Just-In-Time compilation) takes your function, compiles it to optimized machine code the first time you call it, and every call after that is very fast.

```python
import jax

def my_fn(x):
    return x * 2 + 1

fast_fn = jax.jit(my_fn)  # now this runs compiled and optimized
```

---

## JAX vs PyTorch

| | PyTorch | JAX |
|---|---|---|
| Arrays | `torch.Tensor` | `jax.Array` (looks like numpy) |
| Gradients | `.backward()` on the loss | `jax.grad(fn)` - a function |
| GPU | `.to('cuda')` | automatic |
| Speed | eager by default | compile with `jax.jit()` |
| Style | object-oriented | functional (no in-place mutation) |

---

## The One Big Rule: No Mutation

PyTorch lets you modify arrays in place. JAX does not. Every operation creates a new array.

```python
# This is fine in NumPy/PyTorch:
x[0] = 99

# JAX will error on this.
# Instead you do:
x = x.at[0].set(99)   # creates a new array with that value changed
```

This feels strange for about two days, then becomes natural. The reason JAX enforces this is that `jit` and `grad` both need functions to be mathematically pure (same input always gives same output, no hidden state). Mutation breaks that guarantee.

---

## Why DeepMind Uses JAX

Because `jit` and `grad` compose together cleanly at scale. When you are training a model with millions of parameters across hundreds of TPUs, you need compiled, mathematically clean code. JAX was built at Google for exactly that. Almost every GDM paper's code is in JAX/Flax.

---

## What You Already Know

If you know NumPy and PyTorch, you already understand 80% of JAX. The new things to learn are:

1. `jax.grad` - differentiate any function
2. `jax.jit` - compile any function
3. The no-mutation rule - always return a new array

That is the whole foundation. Everything else is built on these three ideas.