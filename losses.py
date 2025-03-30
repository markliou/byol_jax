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

def byol_loss(y, y_hat):
    y_normal = y / jnp.linalg.norm(y, ord=2, axis=-1, keepdims=True)
    y_hat_normal = y_hat / jnp.linalg.norm(y_hat, ord=2, axis=-1, keepdims=True)
    return jnp.mean(jnp.sum(2 - 2 * (y_normal * y_hat_normal), axis=1))