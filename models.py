import tensorflow as tf
import numpy as np
from numpy import linalg as LA

slim = tf.contrib.slim

def vgg_16(inputs, num_cate = 50, num_attr = 1000, dropout_keep_prob = 0.5,
        spatial_squeeze = True, scope = 'vgg_16', padding = 'VALID'):

    """
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: 
    """

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections = end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope = 'conv1')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv2')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope = 'conv3')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope = 'conv4')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope = 'conv5')
            net = slim.max_pool2d(net, [2, 2], scope = 'pool5')

            net = slim.conv2d(net, 1024, [7, 7], padding = padding, scope = 'fc6')
            net = slim.dropout(net, dropout_keep_prob, scope='dropout6')
            net = slim.conv2d(net, 512, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, scope='dropout7')
            net1 = slim.conv2d(net, num_cate, [1, 1],
                    activation_fn = None, normalizer_fn = None, scope = 'fc8-c')
            net2 = slim.conv2d(net, num_attr, [1, 1],
                    activation_fn = None, normalizer_fn = None, scope = 'fc8-a')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net1 = tf.squeeze(net1, [1, 2], name = 'fc8-c/squeezed')
                net2 = tf.squeeze(net2, [1, 2], name = 'fc8-a/squeezed')

                end_points[sc.name + '/fc8-c'] = net1
                end_points[sc.name + '/fc8-a'] = net2
            return net1, net2, end_points
vgg_16.default_image_size = 224

def closest_l2_distance(feature_and_path, query):

    _dists = []
    for _tuple in feature_and_path:
        feature, path = _tuple
        dist = LA.norm(feature - query)
        _dists.append(dist)

    closests = np.argsort(_dists)[:5]

    paths = [feature_and_path[idx][1] for idx in closests]
    return paths
