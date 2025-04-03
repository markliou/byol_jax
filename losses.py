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

def byol_loss(y_teacher, y_student):
    return jnp.mean(2 - 2 * optax.losses.cosine_similarity(y_student, jax.lax.stop_gradient(y_teacher)))