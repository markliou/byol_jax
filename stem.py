import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import flax.nnx

class CNNStem(flax.nnx.Module):
    def __init__(self, 
                 rngs: flax.nnx.Rngs,
                 *args, **kwargs):
        super(flax.nnx.Module, self).__init__(*args, **kwargs)
        self.rngs = rngs
        
        # input: [n, 128, 128, 3]
        self.conv1 = flax.nnx.Conv(3, 16, (11,11), strides=2, rngs=self.rngs) # [-1, 64, 64, 16]
        self.conv1c = flax.nnx.Conv(16, 16, (3,3), strides=1, rngs=self.rngs) # [-1, 64, 64, 16]
        self.conv2 = flax.nnx.Conv(16, 32, (11,11), strides=2, rngs=self.rngs) # [-1, 32, 32, 32]
        self.conv2c = flax.nnx.Conv(32, 32, (3,3), strides=1, rngs=self.rngs) # [-1, 32, 32, 32]
        self.conv3 = flax.nnx.Conv(32, 64, (11,11), strides=2, rngs=self.rngs) # [-1, 16, 16, 64]
        self.conv3c = flax.nnx.Conv(64, 64, (3,3), strides=1, rngs=self.rngs) # [-1, 16, 16, 64]
        
        # down sampling 1d feature map
        # input shape: [-1, 16 * 16, 64]
        self.dconv1 = flax.nnx.Conv(64, 32, (9), strides=2, rngs=self.rngs) # [-1, 128, 32]
        self.dconv2 = flax.nnx.Conv(32, 16, (9), strides=2, rngs=self.rngs) # [-1, 64, 16]
        self.dconv3 = flax.nnx.Conv(16, 1, (9), strides=2, rngs=self.rngs) # [-1, 32, 1]
        
        # fft by long conv
        self.longConv1 = flax.nnx.Conv(1, 1, (32), strides=1, rngs=self.rngs) # [-1, 32, 1]
        self.longConv2 = flax.nnx.Conv(1, 1, (32), strides=1, rngs=self.rngs) # [-1, 32, 1]
                
    def __call__(self, x):
        conv1 = self.conv1(x)
        conv1c = self.conv1c(conv1)
        conv1c = self.mish(conv1c)
        
        conv2 = self.conv2(conv1c)
        conv2c = self.conv2c(conv2)
        conv2c = self.mish(conv2c)
        
        conv3 = self.conv3(conv2c)
        conv3c = self.conv3c(conv3)
        conv3c = self.mish(conv3c)
        
        # reshape the feature maps for down sampling
        dInput = conv3c.reshape([-1, 16 * 16, 64])
        dconv1 = self.dconv1(dInput)
        dconv2 = self.mish(self.dconv2(dconv1))
        dconv3 = self.dconv3(dconv2)
        
        long1 = self.longConv1(dconv3)
        long2 = self.longConv2(long1)
        
        output = long2.reshape([-1, 32])
        
        return output
        

    def mish(self, x):
        return x * flax.nnx.tanh(flax.nnx.softplus(x))
    
class projectLayer(flax.nnx.Module):
    def __init__(self, 
                 rngs,
                 *args, 
                 **kwargs):
        super(flax.nnx.Module, self).__init__(*args, **kwargs)
        self.rngs = rngs
        
        # input: [-1, 32]
        
        self.longConv1 = flax.nnx.Conv(1, 4, (32), strides=1, rngs=self.rngs) # [-1, 32, 4]
        self.longConv2 = flax.nnx.Conv(4, 1, (32), strides=1, rngs=self.rngs) # [-1, 32, 1]
        
    def __call__(self, x):
        longConv1 = self.longConv1(x)
        longConv2 = self.longConv2(longConv1)
        
        return longConv2.reshape([-1, 32])
    
    
def model_weights_ma_update(teacher_model, student_model, tau):
    
    pass

    
if __name__ == "__main__":
    model = CNNStem(flax.nnx.Rngs(1))
    print(model(np.ones([5, 128, 128, 3])))
    print(model(np.ones([5, 128, 128, 3])).shape)
    pass
        