BYOL model uses 2 model weight manipulation that would cause some issues to the model built from Jax:
1. the model weights need to be update by EMA
2. if the regularizer is applied, how to manipulate the model weights for optimizer

Here are the example codes for these issues.
