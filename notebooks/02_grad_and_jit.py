# %%
import jax
import jax.numpy as jnp
import time

# -------------------------------------------------------
# SECTION 1 - Your first gradient
# -------------------------------------------------------

# Here is a simple function: f(x) = x * x
# In math: f(x) = x^2
# Its derivative is: 2x
# So at x=3, the derivative should be 2*3 = 6

def square(x):
    return x * x

# jax.grad takes your function and returns a NEW function
# that computes the gradient (derivative) of the original
grad_of_square = jax.grad(square)

result = grad_of_square(3.0)
print("Gradient of x^2 at x=3:", result)
# You should see 6.0
# Verify in your head: derivative of x^2 is 2x, and 2*3 = 6. Matches.

# %%
# -------------------------------------------------------
# SECTION 2 - A slightly more real example
# -------------------------------------------------------

# In ML, your function is a loss.
# Loss measures how wrong your model is.
# Here is a simple loss: the difference between prediction and target, squared.

# prediction = w * x  (w is the weight, x is the input)
# loss = (prediction - target)^2

# We want to know: if I nudge w, how does the loss change?
# That gradient tells us which direction to move w to reduce the loss.

def loss(w):
    x = 2.0        # input (fixed)
    target = 10.0  # what we want the output to be
    prediction = w * x
    return (prediction - target) ** 2

grad_of_loss = jax.grad(loss)

w = 1.0
print("Loss at w=1:", loss(w))
print("Gradient at w=1:", grad_of_loss(w))
# A negative gradient means: increase w to reduce the loss
# A positive gradient means: decrease w to reduce the loss

# %%
# -------------------------------------------------------
# SECTION 3 - Gradient descent by hand (5 steps)
# -------------------------------------------------------

# This is the core training loop of every neural network.
# We nudge w in the opposite direction of the gradient, repeatedly.

learning_rate = 0.1
w = 1.0

print("--- Gradient Descent ---")
for step in range(5):
    grad = grad_of_loss(w)
    w = w - learning_rate * grad
    print(f"step {step + 1}: w = {w:.4f}, loss = {loss(w):.4f}")

# After 5 steps, loss should be shrinking.
# The target was 10 = w * 2, so the ideal w is 5.0
# Watch w move toward 5.0 as loss approaches 0.

# %%
# -------------------------------------------------------
# SECTION 4 - jit (speed comparison)
# -------------------------------------------------------

def slow_computation(x):
    return jnp.sum(jnp.sin(x) ** 2 + jnp.cos(x) ** 2)

fast_computation = jax.jit(slow_computation)

x = jnp.ones((1_000_000,))

# Warm up jit (first call compiles - ignore this time)
fast_computation(x).block_until_ready()

# Time the un-jitted version
start = time.time()
for _ in range(100):
    slow_computation(x).block_until_ready()
slow_time = time.time() - start

# Time the jitted version
start = time.time()
for _ in range(100):
    fast_computation(x).block_until_ready()
fast_time = time.time() - start

print(f"Without jit: {slow_time:.3f} seconds")
print(f"With jit:    {fast_time:.3f} seconds")
print(f"Speedup:     {slow_time / fast_time:.1f}x")
# %%


# Note to self : block_until_ready() :
# JAX is asynchronous by default - when you run a computation, JAX sends it to the CPU/GPU and moves on without waiting for the result to come back.
# block_until_ready() says: "wait here until the result is actually finished."
# We needed it in the timing code because without it, you would be timing just the moment JAX sent the work, not the moment it finished. That would make everything look instant and the benchmark would be meaningless.