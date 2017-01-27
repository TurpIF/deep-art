import numpy as np
import scipy as sp
import scipy.misc
import scipy.io
import tensorflow as tf
import .vgg
from functools import reduce


def loss_content(content_layers, layers_weights, x_layers):
    def unit_loss(p, x):
        shape = tf.shape(p)
        N = shape[3]
        M = shape[1] * shape[2]
        Nf = tf.cast(N, tf.float32)
        Mf = tf.cast(M, tf.float32)
        ratio = 1. / (4. * Nf * Mf)
        return ratio * tf.reduce_sum(tf.pow(x - p, 2))
    return reduce(tf.add, (wl * unit_loss(content_layers[l], x_layers[l]) for l, wl in layers_weights.items()))


def loss_style(style_layers, layers_weights, x_layers):
    def unit_loss(a, x):
        def gram_matrix(m, N, M):
            mt = tf.reshape(m, (M, N))
            return tf.matmul(tf.transpose(mt), mt)

        shape = tf.shape(a)
        N = shape[3]
        M = shape[1] * shape[2]
        A = gram_matrix(a, N, M)
        G = gram_matrix(x, N, M)
        Nf = tf.cast(N, tf.float32)
        Mf = tf.cast(M, tf.float32)
        ratio = 1. / (4. * Nf**2 * Mf**2)
        return ratio * tf.reduce_sum(tf.pow(G - A, 2))
    return reduce(tf.add, (wl * unit_loss(style_layers[l], x_layers[l]) for l, wl in layers_weights.items()))


def load_image(path):
    img = sp.misc.imread(path)
    if img.shape[-1] != 3:
        img = img[:,:,:,:3]
    return img


def noisy_image(content, ratio, shape):
    noise = tf.Variable(initial_value=np.random.uniform(-20, 20, size=shape), dtype=tf.float32)
    return ratio * noise + (1 - ratio) * content


def normalized_weights(weights):
    s = sum(weights.values())
    return {k: v * 1.0 / s for k, v in weights.items()}


def save_image(path, img):
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)


def execute(content, style,  output_dir, vgg_path, content_layers_weights, style_layers_weights, alpha, beta, device='/gpu:0', total_range=range(10000), snapshot_step=100):
    if content.shape != style.shape:
        style = sp.misc.imresize(style[0,:,:,:], content.shape[1:], interp='bicubic')
        style = np.reshape(style, content.shape)

    with tf.device(device):
        # Use constant to improve performance
        # http://stackoverflow.com/questions/37596333/tensorflow-store-training-data-on-gpu-memory
        img_content = tf.constant(content, dtype=tf.float32) # FIXME don't use shape of loaded content
        img_style = tf.constant(style, dtype=tf.float32) # FIXME don't use shape of loaded style

        img_target = noisy_image(img_content, ratio=0.6, shape=content.shape) # FIXME don't use shape of loaded content

        vgg_factory = vgg.factoryFromPath(vgg_path)
        vgg_content = vgg_factory.build(img_content)
        vgg_style = vgg_factory.build(img_style)
        vgg_target = vgg_factory.build(img_target)

        content_loss = loss_content(vgg_content, content_layers_weights, vgg_target)
        style_loss = loss_style(vgg_style, style_layers_weights, vgg_target)
        total_loss = alpha * content_loss + beta * style_loss
        train_step = tf.train.AdamOptimizer(2.0).minimize(total_loss)

        init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Fix vgg layers of content and style
        for layer in content_layers_weights.keys():
            vgg_content[layer] = sess.run(vgg_content[layer])
        for layer in style_layers_weights.keys():
            vgg_style[layer] = sess.run(vgg_style[layer])

        for i in total_range:
            sess.run(train_step)
            if (i + 1) % snapshot_step == 0:
                save_image(output_dir + '/' + str(i) + '.png', sess.run(img_target))
