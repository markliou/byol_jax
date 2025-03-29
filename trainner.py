import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import flax.nnx
import tensorflow_datasets as tfds
import tensorflow as tf

import stem
import losses

# training parameters
batchSize = 128
learningRate = 1e-4
trainingStep = 10000

# create the dataset
dsimg = tfds.load("beans", split='train', shuffle_files=True, batch_size=-1)['image'].numpy()
reImg = tf.image.resize(dsimg, [256,256])
dataset = tf.data.Dataset.from_tensor_slices(reImg)
# dataset = dataset.batch(batchSize, drop_remainder=True).repeat().shuffle(3, reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
dataset = dataset.repeat().shuffle(3, reshuffle_each_iteration=True).batch(batchSize, drop_remainder=True).map(lambda x: tf.image.random_crop(value=x, size=(batchSize, 128, 128, 3)), num_parallel_calls=8).prefetch(8)
ds_iter = iter(dataset)

# creating teacher and student model
teacher = stem.CNNStem(flax.nnx.Rngs(0))
student = stem.CNNStem(flax.nnx.Rngs(1))
projectStudent = stem.StudentProject(student, flax.nnx.Rngs(2))


# making the optimizer
optChain = optax.chain(
   optax.clip_by_global_norm(1.0),
   optax.adamw(learningRate),
)
opt = flax.nnx.Optimizer(projectStudent, optChain)

# using MSE as training loss
# @flax.nnx.jit
def loss_fn(projectStudent, teacher, x1, x2):
    y1 = teacher(x1)
    y2 = projectStudent(x2)
    return losses.mse(y1, y2)
grad_fn = flax.nnx.value_and_grad(loss_fn)

def update_model_weights(teacher, student, projectStudent, x1, x2):
   loss, grads = grad_fn(projectStudent, teacher, x1, x2)
   # update student
   opt.update(grads)
   # update teacher
   stem.teacher_weights_ma_update(teacher, student)
   return loss

# training loop
for step in range(trainingStep):
    x1 = jnp.array(next(ds_iter))
    x2 = jnp.array(tf.image.random_crop(value=x1, size=(batchSize, 128, 128, 3)))
    loss = update_model_weights(teacher, student, projectStudent, x1, x2)
    
    if step % 100 == 0:
        print("step:{}  loss:{}".format(step, loss))
    
    
    

