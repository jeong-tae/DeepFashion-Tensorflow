from models import vgg_16
from data_loader import Data_loader
import tensorflow as tf

class Trainer(object):
    def __init__(self, image_size):

        self.input_images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        # sparse label
        self.input_cate = tf.placeholder(tf.int32, [None])
        self.input_attr = tf.placeholder(tf.float32, [None, 1000])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.g_step = tf.Variable(0)

    def build_model(self):

        pred_cate, pred_attr, end_points = vgg_16(self.input_images, num_cate = self.num_cate, num_attr = self.num_attr, dropout_keep_prob = self.dropout_keep_prob) 


