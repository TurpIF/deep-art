import tensorflow as tf

class VGGModel:
    def __init__(self, graph):
        self.graph = graph

    def __getattr__(self, name):
        return graph[name]

    def __getitem__(self, item):
        return graph[item]


class VGGFactory:
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    def __init__(self, vgg):
        self.vgg = vgg
        self.vgg_layers = vgg['layers']

    def extract_weights(self, layer, expected_layer_name):
        layer_name = self.vgg_layers[0, layer][0, 0][0][0]
        assert layer_name == expected_layer_name

        W = self.vgg_layers[0, layer][0, 0][2][0, 0]
        b = self.vgg_layers[0, layer][0, 0][2][0, 1][:, 0]
        return tf.constant(W), tf.constant(b)

    def conv2d(self, layer, W, b):
        return tf.nn.conv2d(layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def relu_conv2d(self, prev_layer, layer_id, layer_name):
        W, b = self.extract_weights(layer_id, layer_name)
        return tf.nn.relu(self.conv2d(prev_layer, W, b))

    def avg_pool(self, layer):
        return tf.nn.avg_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME')

    ## tensor: image tensor with shape=(1, 1, 1, 3) between 0 and 255
    def build(self, tensor):
        source = tf.reshape(tensor, (1, 1, 1, 3)) - VGGModelFactory.MEAN_VALUES

        graph = {}
        graph['conv1_1'] = self.relu_conv2d(source, 0, 'conv1_1')
        graph['conv1_2'] = self.relu_conv2d(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = self.avg_pool(graph['conv1_2'])
        graph['conv2_1'] = self.relu_conv2d(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2'] = self.relu_conv2d(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = self.avg_pool(graph['conv2_2'])
        graph['conv3_1'] = self.relu_conv2d(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2'] = self.relu_conv2d(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3'] = self.relu_conv2d(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4'] = self.relu_conv2d(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = self.avg_pool(graph['conv3_4'])
        graph['conv4_1'] = self.relu_conv2d(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2'] = self.relu_conv2d(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3'] = self.relu_conv2d(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4'] = self.relu_conv2d(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = self.avg_pool(graph['conv4_4'])
        graph['conv5_1'] = self.relu_conv2d(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2'] = self.relu_conv2d(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3'] = self.relu_conv2d(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4'] = self.relu_conv2d(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = self.avg_pool(graph['conv5_4'])
        return VGGModel(graph)


def factoryFromPath(path):
    import scipy.io
    return VGGFactory(scipy.io.loadmat(path))
