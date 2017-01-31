import numpy as np
import scipy as sp
import scipy.misc
import scipy.io
import tensorflow as tf
import .vgg
import .image
from functools import reduce


class VGGDiffModel:
    def __init__(self, vgg_A, vgg_B, layer_weights, unit_loss):
        self.vgg_A = vgg_A
        self.vgg_B = vgg_B
        self.layer_weights = layer_weights
        self.unit_loss = unit_loss

    def loss(self):
        return reduce(tf.add, (wl * self.unit_loss(self.vgg_A[l], self.vgg_B[l])
            for l, wl in self.layers_weights.items()))


class DeepArtModel:
    def __init__(self, vgg_factory, content_layers_weights, style_layers_weights, alpha, beta, target=None):
        content = tf.placeholder(tf.float32, shape=[None, None, 3])
        style = tf.placeholder(tf.float32, shape=[None, None, 3])

        if target is None:
            self._target = tf.Variable(target)
        else:
            ratio = 0.6
            noise = tf.Variable(tf.random_uniform(shape, -20, 20), dtype=tf.float32)
            self._target = ratio * noise + (1 - ratio) * content

        vgg_content = vgg_factory.build(content)
        vgg_style = vgg_factory.build(style)
        vgg_target = vgg_factory.build(self._target)

        vgg_diff_content = VGGDiffModel(vgg_content, vgg_target, content_layers_weights, _content_unit_loss)
        vgg_diff_style = VGGDiffModel(vgg_style, vgg_target, style_layers_weights, _style_unit_loss)

        self._content_loss = vgg_diff_content.loss()
        self._style_loss = vgg_diff_style.loss()
        self._loss = alpha * self._content_loss + beta * self._style_loss
        self._step = tf.train.AdamOptimizer(2.0).minimize(self._loss)

    def precompute_vgg(self, session, content, style):
        feed_dict = {
            content: content,
            style: style
        }
        for layer in content_layers_weights.keys():
            vgg_content[layer] = session.run(vgg_content[layer], feed_dict=feed_dict)
        for layer in style_layers_weights.keys():
            vgg_style[layer] = session.run(vgg_style[layer], feed_dict=feed_dict)

    def content_loss(self):
        return self._content_loss

    def style_loss(self):
        return self._style_loss

    def loss(self):
        return self._loss

    def train_step(self):
        return self._step

    def target(self):
        return self._target

    def _content_unit_loss(p, x):
        shape = tf.shape(p)
        N = shape[3]
        M = shape[1] * shape[2]
        Nf = tf.cast(N, tf.float32)
        Mf = tf.cast(M, tf.float32)
        ratio = 1. / (4. * Nf * Mf)
        return ratio * tf.reduce_sum(tf.pow(x - p, 2))

    def _style_unit_loss(a, x):
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


def execute(content, style,  output_dir, vgg_path, content_layers_weights, style_layers_weights, alpha, beta, total_range=range(10000), snapshot_step=100, target=None):
    if content.shape != style.shape:
        style = sp.misc.imresize(style[0,:,:,:], content.shape[1:], interp='bicubic')
        style = np.reshape(style, content.shape)

    model = DeepArtModel(vgg.factoryFromPath(vgg_path), content_layers_weights, style_layers_weights, alpha, beta, target=target)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        model.precompute_vgg(sess, content, style)

        for i in total_range:
            sess.run(model.train_step())
            if (i + 1) % snapshot_step == 0:
                img = sess.run(model.target())
                image.save(output_dir + '/' + str(i) + '.png', img)
