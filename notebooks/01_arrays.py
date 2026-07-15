import jax
import jax.numpy as jnp

# print(jax.__version__)
# print(jax.devices())

x = jnp.array([1,2,3])
print(x)
print(x.min())
print(x.max())
print(jnp.sqrt(4))

a = jnp.array([1, 2, 3])
b = jnp.array([10, 20, 30])

print(a + b)
print(a * b)
print(a - b)
print(a / b)

print(jnp.mean(a))
print(jnp.sum(a))
print(jnp.max(a))
print(jnp.min(a))

matrix = jnp.array([
    [1,2],
    [3,4]
])

print(matrix)
print(matrix.shape)
print(matrix.dtype)
print(matrix.T)