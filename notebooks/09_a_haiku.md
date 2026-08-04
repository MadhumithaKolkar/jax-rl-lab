# Day 9: Haiku - Writing Neural Networks the DeepMind Way

## What is Haiku?

Haiku is DeepMind's neural network library built on top of JAX. It is what DeepMind researchers actually use when they write RL code.

Until now we have been building neural networks by hand:

```python
# what we have been doing
params = {
    "w1": jax.random.normal(key, (16, 64)) * 0.1,
    "b1": jnp.zeros(64),
    "w2": jax.random.normal(key, (64, 4)) * 0.1,
    "b2": jnp.zeros(4),
}

def forward(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return jax.nn.softmax(h @ params["w2"] + params["b2"])
```

This works. But it gets messy fast. More layers means more dictionaries, more manual weight initialization, more room for bugs.

Haiku gives you a cleaner way to define networks while keeping all of JAX's superpowers.

---

## The one thing that makes Haiku different

In PyTorch, the network object stores its own weights:

```python
model = MyNetwork()        # weights live inside model
output = model(input)      # weights are used internally
```

In Haiku, weights are always stored separately from the network definition. The network is just a function. The weights are a dictionary you pass in.

```python
# Haiku style
params = network.init(key, input)   # weights created here, stored in params
output = network.apply(params, input)  # pass weights in explicitly
```

This might look like more work. But it means the network is a pure function - no hidden state, no side effects. JAX can jit it, vmap it, grad it, just like any other function. This is why research code is cleaner in Haiku than in PyTorch.

---

## How to define a network in Haiku

```python
import haiku as hk

def actor_fn(x):
    network = hk.Sequential([
        hk.Linear(64),
        jax.nn.tanh,
        hk.Linear(4),
        jax.nn.softmax,
    ])
    return network(x)

# wrap it - this gives you init and apply
actor = hk.without_apply_rng(hk.transform(actor_fn))
```

Three things happening here:

1. You define your network as a plain Python function
2. `hk.transform` wraps it into something with an `init` and `apply` method
3. `hk.without_apply_rng` just says "I don't need random numbers during the forward pass" - keeps the API cleaner

Then you use it like this:

```python
params = actor.init(key, sample_input)   # initialize weights
probs  = actor.apply(params, x)          # run the network
```

---

## hk.Linear - the basic building block

`hk.Linear(output_size)` is a fully connected layer. It is equivalent to:

```python
output = x @ W + b
```

Haiku handles the weight initialization and parameter naming automatically. You never manually create `w1`, `b1` etc. again.

---

## hk.Sequential - stacking layers

`hk.Sequential([layer1, layer2, ...])` runs your input through each layer in order. Same as PyTorch's `nn.Sequential`.

```python
network = hk.Sequential([
    hk.Linear(64),    # linear layer: 16 inputs -> 64 outputs
    jax.nn.tanh,      # activation function
    hk.Linear(4),     # linear layer: 64 -> 4
    jax.nn.softmax,   # output probabilities
])
```

---

## hk.transform - the key wrapper

`hk.transform` is the step that makes everything work with JAX. It converts your network function (which internally calls `hk.Linear` etc.) into two pure functions:

- `init(key, sample_input)` - creates and returns the parameter dictionary
- `apply(params, input)` - runs the forward pass using those parameters

After this, `params` is just a nested dictionary of arrays. You can pass it to `jax.grad`, `jax.jit`, `jax.vmap` - anything you'd do with a regular JAX array.

---

## Why research teams use Haiku

When you read DeepMind papers and look at their code releases (AlphaFold, Acme, AlphaStar), the network definitions look like the Haiku style above. Clean, modular, easy to swap layers.

More importantly: because params is just a dictionary, you can:
- Save and load checkpoints easily
- Pass params between processes for distributed training
- Inspect individual layer weights
- Merge params from different networks (useful in meta-learning)

All things that are awkward with PyTorch's stateful approach.

---

## Day 9 in one line

Day 9 is Day 7 (Actor-Critic) rewritten in Haiku. Same algorithm, same gridworld, same result. But the code is half the length and looks like actual DeepMind research code.

From Day 10 onwards we use Haiku for everything.