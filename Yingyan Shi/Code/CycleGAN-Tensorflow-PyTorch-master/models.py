from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim


conv = partial(slim.conv2d, activation_fn=None)
deconv = partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
lrelu = partial(tf.nn.leaky_relu, alpha=0.2)
batch_norm = partial(slim.batch_norm, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)


def discriminator(img, scope, dim=64, train=True):
    bn = partial(batch_norm, is_training=train)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(scope + '_discriminator', reuse=tf.AUTO_REUSE):
        net = lrelu(conv(img, dim, 4, 2))
        net = conv_bn_lrelu(net, dim * 2, 4, 2)
        net = conv_bn_lrelu(net, dim * 4, 4, 2)
        net = conv_bn_lrelu(net, dim * 8, 4, 1)
        net = conv(net, 1, 4, 1)

        return net


def generator(img, scope, dim=64, train=True):
    bn = partial(batch_norm, is_training=train)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    deconv_bn_relu = partial(deconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    def _residule_block(x, dim):
        y = conv_bn_relu(x, dim, 3, 1)
        y = bn(conv(y, dim, 3, 1))
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
        net = conv_bn_relu(img, dim, 7, 1)
        net = conv_bn_relu(net, dim * 2, 3, 2)
        net = conv_bn_relu(net, dim * 4, 3, 2)

        for i in range(9):
            net = _residule_block(net, dim * 4)

        net = deconv_bn_relu(net, dim * 2, 3, 2)
        net = deconv_bn_relu(net, dim, 3, 2)
        net = conv(net, 3, 7, 1)
        net = tf.nn.tanh(net)

        return net

'''
tensorflow: name_scope 和 variable_scope区别及理解 - 一遍看不懂，我就再看一遍 - CSDN博客  
https://blog.csdn.net/u012609509/article/details/80045529
在实际使用中，三种创建变量方式的用途也是分工非常明确的。其中

tf.placeholder() 占位符。* trainable==False *
tf.Variable() 一般变量用这种方式定义。 * 可以选择 trainable 类型 *
tf.get_variable() 一般都是和 tf.variable_scope() 配合使用，从而实现变量共享的功能。 * 可以选择 trainable 类型 *

即为了使得在代码的任何部分可以使用某一个已经创建的变量，TF引入了变量共享机制，
使得可以轻松的共享变量，而不用传一个变量的引用

tf.Variable()：只要使用该函数，一律创建新的variable，
如果出现重名，变量名后面会自动加上后缀1，2….

tf.get_variable()：如果变量存在，则使用以前创建的变量，
如果不存在，则新创建一个变量。

tensorflow中的两种作用域
命名域(name scope)：通过tf.name_scope()来实现；
变量域（variable scope）：通过tf.variable_scope()来实现；可以通过设置reuse 标志以及初始化方式来影响域下的变量。 
这两种作用域都会给tf.Variable()创建的变量加上词头，而tf.name_scope对tf.get_variable()创建的变量没有词头影响

tf.name_scope() 并不会对 tf.get_variable() 创建的变量有任何影响。 
tf.name_scope() 主要是用来管理命名空间的，这样子让我们的整个模型更加有条理。
而 tf.variable_scope() 的作用是为了实现变量共享，它和 tf.get_variable() 来完成变量共享的功能。
首先我们要确立一种 Graph 的思想。在 TensorFlow 中，我们定义一个变量，相当于往 Graph 中添加了一个节点。
和普通的 python 函数不一样，在一般的函数中，我们对输入进行处理，然后返回一个结果，而函数里边定义的一些局部变量我们就不管了。
但是在 TensorFlow 中，我们在函数里边创建了一个变量，就是往 Graph 中添加了一个节点。出了这个函数后，这个节点还是存在于 Graph 中的。
'''
