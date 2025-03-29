import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import flax.nnx
import tensorflow_datasets as tfds
import tensorflow as tf
import stem


# create the dataset
batchSize = 128
dsimg = tfds.load("beans", split='train', shuffle_files=True, batch_size=-1)['image'].numpy()
reImg = tf.image.resize(dsimg, [256,256])
dataset = tf.data.Dataset.from_tensor_slices(reImg)
# dataset = dataset.batch(batchSize, drop_remainder=True).repeat().shuffle(3, reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
dataset = dataset.repeat().shuffle(3, reshuffle_each_iteration=True).batch(batchSize, drop_remainder=True).map(lambda x: tf.image.random_crop(value=x, size=(batchSize, 128, 128, 3)), num_parallel_calls=8).prefetch(8)
ds_iter = iter(dataset)

teacher = stem.CNNStem(flax.nnx.Rngs(0))
student = stem.CNNStem(flax.nnx.Rngs(1))
projector = stem.projectLayer(flax.nnx.Rngs(2))

def student_proj(x):
    return projector(student(x))



print(teacher(next(ds_iter)).shape)
print(student(next(ds_iter)).shape)
print(student_proj(next(ds_iter)).shape)