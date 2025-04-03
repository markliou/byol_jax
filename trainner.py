import os
os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = '50'  # 限制 CUDA MPS 資源

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import flax.nnx
import tensorflow_datasets as tfds
import tensorflow as tf
import keras as k

import stem
import losses

# training parameters
batchSize = 512
learningRate = 1e-4
trainingStep = 10000

# create the dataset
random_rotate = k.layers.RandomRotation(0.05)
def imagePreprocess(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = random_rotate(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    # image = tf.image.stateless_random_crop(image, [20, 20, 3], seed=(1, 2))
    image = tf.image.random_crop(value=image, size=(batchSize, 64, 64, 3))
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.resize(image, (128, 128), tf.image.ResizeMethod.BICUBIC)
    image = (tf.cast(image, tf.float32) - 128 / 128.0) 
    return image

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
@flax.nnx.jit
def loss_fn(projectStudent, teacher, x1, x2):
    y1 = teacher(x1)
    y2 = projectStudent(x2)
    return losses.byol_loss(y1, y2)
    # return losses.mse(y1,y2)
grad_fn = flax.nnx.value_and_grad(loss_fn)

# @flax.nnx.jit # ==> cause abnormal output
def update_model_weights(teacher, student, projectStudent, x1, x2):
   loss, grads = grad_fn(projectStudent, teacher, x1, x2)
   # update student
   opt.update(grads)
   # update teacher
   stem.teacher_weights_ma_update(teacher, student)
   return loss

# training loop
for step in range(trainingStep):
    
    # inspect the performance for certain loops
    innerStep = 100
    for innerloop in range(innerStep):
        img = next(ds_iter)
        x1 = jnp.array(imagePreprocess(img))
        x2 = jnp.array(imagePreprocess(img))
        loss = update_model_weights(teacher, student, projectStudent, x1, x2)
    
    print("step:{}  loss:{}".format((step+1) * innerStep, loss))
    
    
    

