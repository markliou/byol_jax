import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import flax.nnx

def mae(y, y_hat):
    return jnp.mean(jnp.abs(y - y_hat))

def mse(y, y_hat):
    return jnp.mean(jnp.square(y - y_hat))