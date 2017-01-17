
# coding: utf-8

import numpy as np
import scipy as sp
import scipy.misc
import scipy.io
import tensorflow as tf
from tqdm import tqdm
from functools import reduce


# load_vgg_layers :: String -> Tensor -> (Layer -> Tensor)
def load_vgg_layers(path, image):
    vgg = sp.io.loadmat(path)
    vgg_layers = vgg['layers']

    def extract_weights(layer, expected_layer_name):
        layer_name = vgg_layers[0, layer][0, 0][0][0]
        assert layer_name == expected_layer_name

        W = vgg_layers[0, layer][0, 0][2][0, 0]
        b = vgg_layers[0, layer][0, 0][2][0, 1][:, 0]
        return tf.constant(W), tf.constant(b)

    def conv2d(layer, W, b):
        return tf.nn.conv2d(layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def relu_conv2d(prev_layer, layer_id, layer_name):
        W, b = extract_weights(layer_id, layer_name)
        return tf.nn.relu(conv2d(prev_layer, W, b))

    def avg_pool(layer):
        return tf.nn.avg_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    graph = {}
    graph['conv1_1'] = relu_conv2d(image, 0, 'conv1_1')
    graph['conv1_2'] = relu_conv2d(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = avg_pool(graph['conv1_2'])
    graph['conv2_1'] = relu_conv2d(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = relu_conv2d(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = avg_pool(graph['conv2_2'])
    graph['conv3_1'] = relu_conv2d(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = relu_conv2d(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = relu_conv2d(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = relu_conv2d(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = avg_pool(graph['conv3_4'])
    graph['conv4_1'] = relu_conv2d(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = relu_conv2d(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = relu_conv2d(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = relu_conv2d(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = avg_pool(graph['conv4_4'])
    graph['conv5_1'] = relu_conv2d(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = relu_conv2d(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = relu_conv2d(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = relu_conv2d(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = avg_pool(graph['conv5_4'])
    return graph


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

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def load_image(path):
    img = sp.misc.imread(path)
    img = np.reshape(img, ((1,) + img.shape))
    img = img - MEAN_VALUES
    return img


def dummy_image(content, ratio, shape):
    noise = tf.Variable(initial_value=np.random.uniform(-20, 20, size=shape), dtype=tf.float32)
    return ratio * noise + (1 - ratio) * content


def normalized_weights(weights):
    s = sum(weights.values())
    return {k: v * 1.0 / s for k, v in weights.items()}


def save_image(path, img):
    # Output should add back the mean.
    img = img + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)


np.random.seed(0)

content = load_image('./PNC.jpg')
# content = sp.misc.imresize(content[0,:,:,:], (128, 128, content.shape[-1]), interp='bilinear')
# content = np.reshape(content, ((1,) + content.shape))
style = load_image('./otto.jpg')
if content.shape != style.shape:
    style = sp.misc.imresize(style[0,:,:,:], content.shape[1:], interp='bilinear')
    style = np.reshape(style, content.shape)

device = '/gpu:0'

with tf.device(device):
    # Use constant to improve performance
    # http://stackoverflow.com/questions/37596333/tensorflow-store-training-data-on-gpu-memory
    img_content = tf.constant(content, dtype=tf.float32) # FIXME don't use shape of loaded content
    img_style = tf.constant(style, dtype=tf.float32) # FIXME don't use shape of loaded style

    img_target = dummy_image(img_content, ratio=0.6, shape=content.shape) # FIXME don't use shape of loaded content

    vgg_path = './imagenet-vgg-verydeep-19.mat'
    vgg_content = load_vgg_layers(vgg_path, img_content)
    vgg_style = load_vgg_layers(vgg_path, img_style)
    vgg_target = load_vgg_layers(vgg_path, img_target)

content_layers_weights = normalized_weights({ 'conv4_2': 1. })
# style_layers_weights = normalized_weights({
#     'conv1_1': 1,
#     'conv2_1': 1,
#     'conv3_1': 1,
#     'conv4_1': 1,
#     'conv5_1': 1
# })
style_layers_weights = normalized_weights({
    'conv1_1': 0.5,
    'conv2_1': 1.0,
    'conv3_1': 1.5,
    'conv4_1': 3.0,
    'conv5_1': 4.0
})

alpha = 100
beta = 5


with tf.device(device):
    content_loss = loss_content(vgg_content, content_layers_weights, vgg_target)
    style_loss = loss_style(vgg_style, style_layers_weights, vgg_target)
    total_loss = alpha * content_loss + beta * style_loss
    train_step = tf.train.AdamOptimizer(2.0).minimize(total_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    def sess_runner(tensor):
        return sess.run(tensor)

    # Fix vgg layers of content and style
    for layer in content_layers_weights.keys():
        vgg_content[layer] = sess_runner(vgg_content[layer])
    for layer in style_layers_weights.keys():
        vgg_style[layer] = sess_runner(vgg_style[layer])

    for i in tqdm(range(5001)):
        sess_runner(train_step)
        if i % 50 == 0:
            save_image('./output5/' + str(i) + '.png', sess_runner(img_target))
