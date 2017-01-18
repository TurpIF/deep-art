from .full import *

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

vgg_path = './imagenet-vgg-verydeep-19.mat'

paths = [
#    ('img1.png', 'style1.jpg'),
#    ('img2.png', 'style2.jpg'),
#    ('img3.png', 'style3.jpg'),
    ('img4.png', 'style4.jpg')
]

np.random.seed(0)

for i, (content_path, style_path) in enumerate(paths):
    print('Image 4: Content:', content_path, 'Style:', style_path)
    execute(content_path, style_path, './out/4/', vgg_path, content_layers_weights, style_layers_weights, alpha, beta)