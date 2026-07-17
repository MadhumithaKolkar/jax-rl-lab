# %%
import jax
import jax.numpy as jnp
import time

# %%
# -------------------------------------------------------
# SECTION 1 - The problem vmap solves
# -------------------------------------------------------

# A simple function that works on ONE input (a single vector)
def score(x):
    return jnp.sum(x ** 2)

# One input, works fine
x_single = jnp.array([1.0, 2.0, 3.0])
print("Single input score:", score(x_single))   # 1 + 4 + 9 = 14

# Now we have a batch of 5 inputs
batch = jnp.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [0.0, 1.0, 0.0],
    [2.0, 2.0, 2.0],
    [1.0, 0.0, 1.0],
])

# The slow way: Python loop
loop_results = jnp.array([score(batch[i]) for i in range(5)])
print("Loop results:  ", loop_results)

# The JAX way: vmap
batch_score = jax.vmap(score)
vmap_results = batch_score(batch)
print("vmap results:  ", vmap_results)

# Same answers, but vmap runs as one parallel operation on the hardware

# %%
# -------------------------------------------------------
# SECTION 2 - Speed comparison
# -------------------------------------------------------

# Large batch: 1000 inputs, each of size 512
large_batch = jnp.ones((1000, 512))

# Loop version (jitted to be fair)
@jax.jit
def loop_score(batch):
    return jnp.array([score(batch[i]) for i in range(batch.shape[0])])

# vmap version (also jitted)
fast_batch_score = jax.jit(jax.vmap(score))

# Warm up both
loop_score(large_batch).block_until_ready()
fast_batch_score(large_batch).block_until_ready()

# Time the loop version
start = time.time()
for _ in range(50):
    loop_score(large_batch).block_until_ready()
loop_time = time.time() - start

# Time the vmap version
start = time.time()
for _ in range(50):
    fast_batch_score(large_batch).block_until_ready()
vmap_time = time.time() - start

print(f"\nLoop time : {loop_time:.3f}s")
print(f"vmap time : {vmap_time:.3f}s")
print(f"Speedup   : {loop_time / vmap_time:.1f}x")

# %%
# -------------------------------------------------------
# SECTION 3 - in_axes: controlling what gets batched
# -------------------------------------------------------

# in_axes tells vmap: which argument to batch over, and which to keep fixed

# Example: a loss function with a shared weight vector w
# w is shared across all examples (same w for every input)
# x and y are per-example (different for each input)

def loss(w, x, y):
    prediction = jnp.dot(w, x)
    return (prediction - y) ** 2

# in_axes=(None, 0, 0) means:
#   None = do NOT batch over w (it is the same for all examples)
#   0    = batch over the 0th axis of x
#   0    = batch over the 0th axis of y
batch_loss = jax.vmap(loss, in_axes=(None, 0, 0))

w = jnp.array([1.0, 2.0, 3.0])

xs = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

ys = jnp.array([1.0, 2.0, 3.0])  # targets

losses = batch_loss(w, xs, ys)
print("\nPer-example losses:", losses)
# w dot [1,0,0] = 1.0, target = 1.0, loss = 0
# w dot [0,1,0] = 2.0, target = 2.0, loss = 0
# w dot [0,0,1] = 3.0, target = 3.0, loss = 0
# All zero because w already perfectly predicts these targets

# %%
# -------------------------------------------------------
# SECTION 4 - The real JAX pattern: jit + vmap + grad together
# -------------------------------------------------------

# This is what you will see in every JAX research codebase
# Write a simple single-example loss function
def single_loss(w, x, y):
    prediction = jnp.dot(w, x)
    return (prediction - y) ** 2

# Step 1: differentiate it (grad w.r.t. w, the first argument)
grad_single_loss = jax.grad(single_loss)

# Step 2: vmap over x and y (not w - it is shared)
batch_grad = jax.vmap(grad_single_loss, in_axes=(None, 0, 0))

# Step 3: jit the whole thing for speed
fast_batch_grad = jax.jit(batch_grad)

# Now run gradient descent on a small dataset
w = jnp.array([0.0, 0.0, 0.0])  # start from zero

xs = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
ys = jnp.array([2.0, 3.0, 5.0, 5.0])

lr = 0.1
print("\n--- Batched gradient descent ---")
for step in range(10):
    # get per-example gradients for the whole batch at once
    grads = fast_batch_grad(w, xs, ys)      # shape: (4, 3) - one gradient per example
    mean_grad = grads.mean(axis=0)           # average across the batch
    w = w - lr * mean_grad
    mean_loss = batch_loss(w, xs, ys).mean()
    print(f"step {step + 1:2d}: w = {w}, loss = {mean_loss:.4f}")

# w should converge toward [2, 3, 5] since those are our targets