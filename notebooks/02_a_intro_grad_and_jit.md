# Understanding grad and jit

Before writing any code, you need to understand two ideas: derivatives and gradients.
These are not advanced math. They are one simple idea, stated two ways.

---

## Part 1 - What is a Derivative?

Imagine you are adjusting a single dial on a machine.

The dial controls how loud the music is. Your goal is to get the volume to exactly 50.
Right now the volume is 30.

You turn the dial a tiny bit to the right. The volume goes up to 32.

That tells you something: turning right increases the volume.
The question the derivative answers is: **if I nudge this dial a tiny bit, how much does the output change?**

That is it. That is a derivative.

In math, it looks like this:

```
f(x) = x * x       (a function, like your dial)
f(3) = 9
f(3.001) = 9.006

nudge of 0.001 caused a change of 0.006
rate of change = 0.006 / 0.001 = 6
```

The derivative of `x * x` at `x = 3` is `6`. It means: at this exact point, if x increases
by a tiny amount, the output increases 6 times as fast.

---

## Part 2 - What is a Gradient?

A derivative is for a function with one input.

A gradient is the same idea, but for a function with many inputs.

Imagine instead of one dial, you have 10 dials. Your machine has 10 knobs.
The gradient tells you: for each knob, if I nudge it a tiny bit, how much does the output change?

The gradient is just a list of derivatives - one per input.

```
function with 1 input  ->  derivative  (a single number)
function with N inputs ->  gradient    (a list of N numbers)
```

---

## Part 3 - Why Does Any of This Matter for ML?

Every machine learning model has weights. Thousands or millions of them.

Training a model means finding the weights that make the model least wrong.
"How wrong the model is" is measured by a number called the **loss**.

The goal of training is: find the weights that make the loss as small as possible.

How do you do that? You use the gradient.

The gradient of the loss tells you: for each weight, does increasing it make the loss
bigger or smaller, and by how much?

Then you nudge every weight in the direction that makes the loss smaller.
Do this thousands of times. That is gradient descent. That is how every neural network trains.

```
high loss  ->  check gradient  ->  nudge all weights  ->  slightly lower loss
repeat until loss is small enough
```

The hard part used to be: computing that gradient by hand for a complex model is
extremely tedious. You would need pages of calculus.

JAX solves this. You write any function in Python. JAX computes its gradient automatically.
That is what `jax.grad` does.

---

## Part 4 - What is jit?

This one is simpler.

Python runs one line at a time, slowly. When you are doing millions of math operations
on large arrays, this is too slow.

`jit` stands for Just-In-Time compilation. It means:

- First time you call the function: JAX reads it, compiles it into fast machine code, runs it
- Every time after that: JAX runs the already-compiled version, which is much faster. ( The machine code is cached in memory / RAM )

You do not need to change your function at all. You just wrap it.

```python
# slow version
result = my_function(x)

# fast version - same function, just compiled
fast_function = jax.jit(my_function)
result = fast_function(x)
```

The first call is slower because of the compile step.
Every call after that is significantly faster.

---

## Without jit - Python reads the recipe every time

Imagine you are a chef. Every time someone orders a dish, you pick up the recipe book, read it line by line, and cook it. Even if you have made that dish 100 times. That is how Python normally runs - it reads and executes your code line by line, every single call.

## With jit - JAX memorises the cooking steps

The first time you call a jit-wrapped function, JAX does something different. Instead of just running your code, it watches what you do and converts it into a highly optimised set of low-level instructions (called XLA - Accelerated Linear Algebra). Think of it as translating your recipe into a laminated, step-optimised card that a machine can follow instantly.

That compiled version is stored in memory (yes, RAM/cache). Every call after the first one skips Python entirely and runs the compiled version directly. No recipe reading. Just execution.

### The one catch - shapes must stay the same

JAX compiles for a specific input shape. If you call the function with a different shaped array, JAX has to recompile.


fast_fn(jnp.ones((100,)))    # compiles for shape (100,)
fast_fn(jnp.ones((100,)))    # uses cache, instant
fast_fn(jnp.ones((200,)))    # different shape, recompiles
That is why the first call is always slow - that is the compile step. Every call after with the same shape is fast.

## Summary

| Idea | One sentence |
|---|---|
| Derivative | If I nudge this one input, how much does the output change? |
| Gradient | Same idea, but for a function with many inputs - one number per input |
| `jax.grad` | Give JAX a function, it returns a new function that computes the gradient |
| `jax.jit` | Give JAX a function, it returns a faster compiled version of it |

In the code file, we will use `grad` to compute a gradient and verify it by hand,
then use `jit` and feel the speed difference.