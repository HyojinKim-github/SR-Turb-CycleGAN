import h5py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from scipy import interpolate
from scipy.io import FortranFile

#--------------------------- Prameter -----------------------------------------------

m=4    # resolution ratio 
n=32   # input data size
batch = 16
 
#---------------------------  MODEL -------------------------------------------------

def act(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

def get_weight(name, shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    return tf.get_variable(name, shape=shape, initializer=tf.initializers.random_normal(0, std))

def apply_bias(name, x):
     if len(x.shape) == 2:
        b = tf.get_variable(name, shape=[x.shape[1]], initializer=tf.initializers.zeros())
        b = tf.cast(b, x.dtype)
        return x + b
     else:
        b = tf.get_variable(name, shape=[x.shape[3]], initializer=tf.initializers.zeros())
        b = tf.cast(b, x.dtype)
        return x + tf.reshape(b, [1, 1, 1, -1])

def dense(name, x, fmaps):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight(name, [x.shape[1].value, fmaps])
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

def downsample2X2(x):
    return tf.add_n([x[:,0::2,0::2,:],x[:,1::2,0::2,:],x[:,0::2,1::2,:],x[:,1::2,1::2,:]]) / 4.

def upsampling(image, p, q):
    s = tf.shape(image)
    W, H, C = s[1], s[2], s[3]
    # Add two dimensions to A for tiling
    A_exp = tf.reshape(image, [-1, W, 1, H, 1, C])
    # Tile A along new dimensions
    A_tiled = tf.tile(A_exp, [1, 1, p, 1, q, 1])
    # Reshape
    A_tiled = tf.reshape(A_tiled, [-1, W * p, H * q, C])
    return A_tiled



X = tf.placeholder(tf.float32,shape=[None,n,n,3])         # [batch, nz, nx, input_maps]
Y = tf.placeholder(tf.float32,shape=[None,m*n,m*n,3])     # [batch, nz, nx, output_maps]
learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

def generator_G(LR, reuse=False):
    H = [ 3, 3, 3, 3, 3,3] # kernel size : HxH
    K = [64,64,32,32,16,3] # number of feature map
    
    with tf.variable_scope('G', reuse=reuse):   
        Kernel_1 = get_weight('W1', [H[0],H[0],3,K[0]])
        Conv_1 = apply_bias('B1',tf.nn.conv2d(LR, Kernel_1, strides=[1,1,1,1], padding='SAME'))
        Activation_1 = act(Conv_1)
        Kernel_2 = get_weight('W2', [H[1],H[1],K[0],K[1]])
        Conv_2 = apply_bias('B2', tf.nn.conv2d(Activation_1, Kernel_2, strides=[1,1,1,1], padding='SAME'))
        Activation_2 = act(Conv_2)
        Upsampling_1 = upsampling(Activation_2, 2,2)
        
        Kernel_3 = get_weight('W3', [H[2],H[2],K[1],K[2]])
        Conv_3 = apply_bias('B3', tf.nn.conv2d(Upsampling_1, Kernel_3, strides=[1,1,1,1], padding='SAME'))
        Activation_3 = act(Conv_3)
        Kernel_4 = get_weight('W4', [H[3],H[3],K[2],K[3]])
        Conv_4 = apply_bias('B4', tf.nn.conv2d(Activation_3, Kernel_4, strides=[1,1,1,1], padding='SAME'))
        Activation_4 = act(Conv_4)
        Upsampling_2 = upsampling(Activation_4, 2,2)
        
        Kernel_5 = get_weight('W5', [H[4],H[4],K[3],K[4]])
        Conv_5 = apply_bias('B5',tf.nn.conv2d(Upsampling_2, Kernel_5, strides=[1,1,1,1], padding='SAME'))
        Activation_5 = act(Conv_5)

        Kernel_6 = get_weight('W6', [H[5],H[5],K[4],K[5]])
        Conv_6 = apply_bias('B6', tf.nn.conv2d(Activation_5, Kernel_6, strides=[1,1,1,1], padding='SAME'))
        
        Y_predict = Conv_6

    return Y_predict


def generator_F(HR, reuse=False):
    H = [ 3, 3, 3, 3, 3,3] # kernel size : HxH
    K = [16,16,32,32,64,3] # number of feature map
    
    with tf.variable_scope('F', reuse=reuse):   
        Kernel_1 = get_weight('W1', [H[0],H[0],3,K[0]])
        Conv_1 = apply_bias('B1',tf.nn.conv2d(HR, Kernel_1, strides=[1,1,1,1], padding='SAME'))
        Activation_1 = act(Conv_1)
        Kernel_2 = get_weight('W2', [H[1],H[1],K[0],K[1]])
        Conv_2 = apply_bias('B2', tf.nn.conv2d(Activation_1, Kernel_2, strides=[1,1,1,1], padding='SAME'))
        Activation_2 = act(Conv_2)
        Downsampling_1 = downsample2X2(Activation_2)

        Kernel_3 = get_weight('W3', [H[2],H[2],K[1],K[2]])
        Conv_3 = apply_bias('B3', tf.nn.conv2d(Downsampling_1, Kernel_3, strides=[1,1,1,1], padding='SAME'))
        Activation_3 = act(Conv_3)
        Kernel_4 = get_weight('W4', [H[3],H[3],K[2],K[3]])
        Conv_4 = apply_bias('B4', tf.nn.conv2d(Activation_3, Kernel_4, strides=[1,1,1,1], padding='SAME'))
        Activation_4 = act(Conv_4)
        Downsampling_2 = downsample2X2(Activation_4)
        
        Kernel_5 = get_weight('W5', [H[4],H[4],K[3],K[4]])
        Conv_5 = apply_bias('B5',tf.nn.conv2d(Downsampling_2, Kernel_5, strides=[1,1,1,1], padding='SAME'))
        Activation_5 = act(Conv_5)

        Kernel_6 = get_weight('W6', [H[5],H[5],K[4],K[5]])
        Conv_6 = apply_bias('B6', tf.nn.conv2d(Activation_5, Kernel_6, strides=[1,1,1,1], padding='SAME'))
        
        X_predict = Conv_6

    return X_predict   


def discriminator_X(LR, reuse = False) :
    H = [ 3, 3, 3,  3,  3,  3,  3] # kernel size 
    K = [32,64,64,128,128,256,256] # number of feature map

    with tf.variable_scope('DX', reuse=reuse):   
        Kernel_1 = get_weight('W1', [H[0],H[0],3,K[0]])
        Conv_1 = apply_bias('B1',tf.nn.conv2d(LR, Kernel_1, strides=[1,1,1,1], padding='SAME'))
        Activation_1 = act(Conv_1)
        Kernel_2 = get_weight('W2', [H[1],H[1],K[0],K[1]])
        Conv_2 = apply_bias('B2', tf.nn.conv2d(Activation_1, Kernel_2, strides=[1,1,1,1], padding='SAME'))
        Activation_2 = act(Conv_2)
        Downsampling_1 = downsample2X2(Activation_2)

        Kernel_3 = get_weight('W3', [H[2],H[2],K[1],K[2]])
        Conv_3 = apply_bias('B3', tf.nn.conv2d(Downsampling_1, Kernel_3, strides=[1,1,1,1], padding='SAME'))
        Activation_3 = act(Conv_3)
        Kernel_4 = get_weight('W4', [H[3],H[3],K[2],K[3]])
        Conv_4 = apply_bias('B4', tf.nn.conv2d(Activation_3, Kernel_4, strides=[1,1,1,1], padding='SAME'))
        Activation_4 = act(Conv_4)
        Downsampling_2 = downsample2X2(Activation_4)

        Kernel_5 = get_weight('W5', [H[4],H[4],K[3],K[4]])
        Conv_5 = apply_bias('B5',tf.nn.conv2d(Downsampling_2, Kernel_5, strides=[1,1,1,1], padding='SAME'))
        Activation_5 = act(Conv_5)
        Kernel_6 = get_weight('W6', [H[5],H[5],K[4],K[5]])
        Conv_6 = apply_bias('B6', tf.nn.conv2d(Activation_5, Kernel_6, strides=[1,1,1,1], padding='SAME'))
        Activation_6 = act(Conv_6)
        Downsampling_3 = downsample2X2(Activation_6)

        Kernel_7 = get_weight('W7', [H[6],H[6],K[5],K[6]])
        Conv_7 = apply_bias('B7',tf.nn.conv2d(Downsampling_3, Kernel_7, strides=[1,1,1,1], padding='SAME'))
        Activation_7 = act(Conv_7)
        
        FC1 = apply_bias('B8', dense('W8',Activation_7,256))
        Activation_8= act(FC1)
        FC2 = apply_bias('B9', dense('W9',Activation_8,1))
        OUT = FC2

    return OUT


def discriminator_Y(HR, reuse = False) :
    H = [ 3, 3, 3, 3, 3,  3,  3,  3,  3] # kernel size 
    K = [16,32,32,64,64,128,128,256,256] # number of feature map
    with tf.variable_scope('DY', reuse=reuse):   
        Kernel_1 = get_weight('W1', [H[0],H[0],3,K[0]])
        Conv_1 = apply_bias('B1',tf.nn.conv2d(HR, Kernel_1, strides=[1,1,1,1], padding='SAME'))
        Activation_1 = act(Conv_1)
        Kernel_2 = get_weight('W2', [H[1],H[1],K[0],K[1]])
        Conv_2 = apply_bias('B2', tf.nn.conv2d(Activation_1, Kernel_2, strides=[1,1,1,1], padding='SAME'))
        Activation_2 = act(Conv_2)
        Downsampling_1 = downsample2X2(Activation_2)

        Kernel_3 = get_weight('W3', [H[2],H[2],K[1],K[2]])
        Conv_3 = apply_bias('B3', tf.nn.conv2d(Downsampling_1, Kernel_3, strides=[1,1,1,1], padding='SAME'))
        Activation_3 = act(Conv_3)
        Kernel_4 = get_weight('W4', [H[3],H[3],K[2],K[3]])
        Conv_4 = apply_bias('B4', tf.nn.conv2d(Activation_3, Kernel_4, strides=[1,1,1,1], padding='SAME'))
        Activation_4 = act(Conv_4)
        Downsampling_2 = downsample2X2(Activation_4)
        
        Kernel_5 = get_weight('W5', [H[4],H[4],K[3],K[4]])
        Conv_5 = apply_bias('B5',tf.nn.conv2d(Downsampling_2, Kernel_5, strides=[1,1,1,1], padding='SAME'))
        Activation_5 = act(Conv_5)
        Kernel_6 = get_weight('W6', [H[5],H[5],K[4],K[5]])
        Conv_6 = apply_bias('B6', tf.nn.conv2d(Activation_5, Kernel_6, strides=[1,1,1,1], padding='SAME'))
        Activation_6 = act(Conv_6)
        Downsampling_3 = downsample2X2(Activation_6)

        Kernel_7 = get_weight('W7', [H[6],H[6],K[5],K[6]])
        Conv_7 = apply_bias('B7',tf.nn.conv2d(Downsampling_3, Kernel_7, strides=[1,1,1,1], padding='SAME'))
        Activation_7 = act(Conv_7)
        Kernel_8 = get_weight('W8', [H[7],H[7],K[6],K[7]])
        Conv_8 = apply_bias('B8', tf.nn.conv2d(Activation_7, Kernel_8, strides=[1,1,1,1], padding='SAME'))
        Activation_8 = act(Conv_8)
        Downsampling_4 = downsample2X2(Activation_8)

        Kernel_9 = get_weight('W9', [H[8],H[8],K[7],K[8]])
        Conv_9 = apply_bias('B9',tf.nn.conv2d(Downsampling_4, Kernel_9, strides=[1,1,1,1], padding='SAME'))
        Activation_9 = act(Conv_9)

        FC1 = apply_bias('B10', dense('W10', Activation_9, 256))
        Activation_10 = act(FC1)
        FC2 = apply_bias('B11', dense('W11', Activation_10, 1))
        OUT = FC2

    return OUT


Y_predict = generator_G(X, reuse=False)
X_predict = generator_F(Y, reuse=False)

Y_predict2 = generator_G(X_predict,reuse=True)
X_predict2 = generator_F(Y_predict,reuse=True)

DX_real = discriminator_X(X, reuse=False)
DX_gene = discriminator_X(X_predict, reuse=True)

DY_real = discriminator_Y(Y, reuse=False)
DY_gene = discriminator_Y(Y_predict, reuse=True)


G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "G")
F_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "F")
DX_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "DX")
DY_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "DY")

#--------------------------------------- Loss -------------------------------------------------

DX_gene_loss = tf.reduce_mean(DX_gene)
DX_real_loss = tf.reduce_mean(DX_real)  

epsilon_X = tf.random_uniform(shape=[batch, 1, 1, 1], minval=0.0, maxval=1.0)
X_hat = X + epsilon_X * (X_predict - X)
DX_hat = discriminator_X(X_hat, reuse=True) 
grad_DX = tf.gradients(DX_hat, [X_hat])[0]
slopes_X = tf.sqrt(tf.reduce_sum(tf.square(grad_DX), axis=[1,2,3]))
gp_X_loss = tf.reduce_mean((slopes_X - 1.0)**2)

DY_gene_loss1 = tf.reduce_mean(DY_gene)
DY_real_loss2 = tf.reduce_mean(DY_real) 

epsilon_Y = tf.random_uniform(shape=[batch, 1, 1, 1], minval=0.0, maxval=1.0)
Y_hat = Y + epsilon_Y * (Y_predict - Y)
DY_hat = discriminator_Y(Y_hat, reuse=True) 
grad_DY = tf.gradients(DY_hat, [Y_hat])[0]
slopes_Y = tf.sqrt(tf.reduce_sum(tf.square(grad_DY), axis=[1,2,3]))
gp_Y_loss = tf.reduce_mean((slopes_Y - 1.0)**2)

# Discriminator loss
DX_loss = DX_gene_loss - DX_real_loss + 10.0*gp_X_loss  
DY_loss = DY_gene_loss - DY_real_loss + 10.0*gp_Y_loss 

# cycle consistency loss
cycle_X_loss = tf.reduce_mean((X_predict2-X)**2)  # forward cycle-consistency loss
cycle_Y_loss = tf.reduce_mean((Y_predict2-Y)**2)  # backward cycle-consistency loss
cycle_loss = cycle_X_loss + cycle_Y_loss

# Generator loss
G_loss = - DY_gene_loss + 10.0*cycle_loss 
F_loss = - DY_real_loss + 10.0*cycle_loss

beta1 = 0
train_G = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(G_loss, var_list=G_vars)
train_F = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(F_loss, var_list=F_vars)
train_DX = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(DX_loss, var_list=DX_vars)
train_DY = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(DY_loss, var_list=DY_vars)

saver = tf.train.Saver(max_to_keep=None)
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
