from models import vgg_16
from data_loader import Data_loader
import tensorflow as tf
import numpy as np
import os

root = './dataset/DeepFashion/'
cate_path = 'Anno/list_category_img.txt'
attr_path = 'Anno/list_attr_img.txt'
partition_path = 'Eval/list_eval_partition.txt'

class Trainer(object):
    def __init__(self, batch_size, image_size, lr, epoch):

        self.input_images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        self.pos_images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        self.neg_images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        # sparse label
        self.input_cate = tf.placeholder(tf.int32, [None])
        self.input_attr = tf.placeholder(tf.float32, [None, 1000])
        self.num_cate = 50
        self.num_attr = 1000
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.g_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(lr, self.g_step, 50000, 0.98)
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_epoch = epoch
        self.d_loader = Data_loader(root, cate_path, attr_path, partition_path, self.batch_size, image_size)

        self.trainX, self.train_pos, self.train_neg, self.trainY1, self.trainY2 = self.d_loader.get_queue(1000, 'train', epoch, True)
        self.valX, self.valY1, self.valY2 = self.d_loader.get_queue(1000, 'val', None, False)
        self.testX, self.testY1, self.testY2 = self.d_loader.get_queue(1000, 'test', None, False)

    def build_model(self):

        pred_triple, pred_attr, self.end_points = vgg_16(tf.concat([self.input_images, self.pos_images, self.neg_images], 0), 
                num_cate = self.num_cate, num_attr = self.num_attr, 
                dropout_keep_prob = self.dropout_keep_prob) 
        self.pred_cate, pos_cate, neg_cate = tf.split(pred_triple, 3)
        self.pred_attr, _, _ = tf.split(pred_attr, 3)

        self.cate_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.input_cate, logits = self.pred_cate))
        self.attr_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.input_attr, logits = self.pred_attr))
        pos_dist = tf.sqrt(tf.reduce_sum(tf.pow(self.pred_cate - pos_cate, 2), reduction_indices = 1))
        neg_dist = tf.sqrt(tf.reduce_sum(tf.pow(self.pred_cate - neg_cate, 2), reduction_indices = 1))
        # deprecated cost
        #triplet_cost = tf.reduce_mean(pos_dist - neg_dist)
        #triplet_cost = tf.clip_by_value(triplet_cost, -40., 40.)
        self.attr_loss = tf.clip_by_value(self.attr_loss, 0., 40.)
        # 0.5 for margin
        self.triplet_loss = tf.maximum(0.0, 0.5 + tf.reduce_mean(pos_dist - neg_dist))

        self.loss = self.cate_loss + self.triplet_loss + self.attr_loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step = self.g_step)

        self.sess = tf.Session()

        print(" [*] model ready")

    def top_k_acc(self, preds, sparse_labels, k):

        argsorted = np.argsort(-preds, 1)[:, :k]
        acc = 100.0 * np.sum([np.any(argsorted[i] == sparse_labels[i]) for i in range(preds.shape[0])]) / preds.shape[0]
        return acc

    def train(self):

        print(" [*] train start")
        prev_loss = 999.
        counting = 0.
        for i in range(self.max_epoch):
            batch_len = int(50000. / self.batch_size)
            for step in range(batch_len):
                batch_img, batch_pos, batch_neg, batch_cate, batch_attr = self.sess.run([self.trainX, 
                        self.train_pos, self.train_neg, self.trainY1, self.trainY2])
                _, c_preds, loss, g_step = self.sess.run([self.train_op, self.pred_cate, self.loss, self.g_step],
                            feed_dict = {
                                self.input_images: batch_img,
                                self.pos_images: batch_pos,
                                self.neg_images: batch_neg,
                                self.input_cate: batch_cate,
                                self.input_attr: batch_attr,
                                self.dropout_keep_prob: 0.5
                            })
                acc = 100.0 * np.sum(np.argmax(c_preds, 1) == batch_cate) / c_preds.shape[0]
                acc5 = self.top_k_acc(c_preds, batch_cate, k = 5)
                print("step: %d, acc: %.2f, top5 acc: %.2f, current loss: %.2f"%(batch_len*i + step, acc, acc5, loss))
                
                if g_step % 10 == 0 or g_step == 0:
                    val_loss = self.validation(g_step)
                    self.test(i)
                    if val_loss < prev_loss:
                        self.saver.save(self.sess, os.path.join("./data/",
                                    "deepfashion.ckpt"), global_step = g_step)
                        prev_loss = val_loss
                        counting = 0
                    else:
                        counting += 1
                
                if counting > 50:
                    print(" [*] Early stopping")
                    break
            self.test(i)
        print(" [*] train end")
            

    def validation(self, i):

        val_len = int(10000. / self.batch_size)
        val_preds = []
        val_cates = []
        val_loss = []
        for step in range(val_len):
            batch_img, batch_cate, batch_attr = self.sess.run([self.valX, self.valY1, self.valY2])
            c_preds, loss = self.sess.run([self.pred_cate, self.loss],
                        feed_dict = {
                            self.input_images: batch_img,
                            self.pos_images: batch_img, # dummy
                            self.neg_images: batch_img, # dummy
                            self.input_cate: batch_cate,
                            self.input_attr: batch_attr,
                            self.dropout_keep_prob: 1.0
                        })
            val_cates.append(batch_cate)
            val_preds.append(c_preds)
            val_loss.append(loss)
        val_preds = np.concatenate(val_preds, axis = 0)
        val_cates = np.concatenate(val_cates, axis = 0)
        acc = 100.0 * np.sum(np.argmax(val_preds, 1) == val_cates) / val_preds.shape[0]
        acc5 = self.top_k_acc(val_preds, val_cates, k = 5)

        mean_loss = sum(val_loss) / float(val_len)
        print("g_step: %d, validation acc: %.2f, top5 acc: %.2f, loss: %.2f"%(i, acc, acc5, mean_loss))

        return mean_loss

    def test(self, i):
        
        test_len = int(10000. / self.batch_size)
        test_preds = []
        test_cates = []
        for step in range(test_len):
            batch_img, batch_cate, batch_attr = self.sess.run([self.testX, self.testY1, self.testY2])
            c_preds, loss = self.sess.run([self.pred_cate, self.loss],
                        feed_dict = {
                            self.input_images: batch_img,
                            self.pos_images: batch_img, # dummy
                            self.neg_images: batch_img, # dummy
                            self.input_cate: batch_cate,
                            self.input_attr: batch_attr,
                            self.dropout_keep_prob: 1.0
                        })
            test_cates.append(batch_cate)
            test_preds.append(c_preds)
        test_preds = np.concatenate(test_preds, axis = 0)
        test_cates = np.concatenate(test_cates, axis = 0)
        acc = 100.0 * np.sum(np.argmax(test_preds, 1) == test_cates) / test_preds.shape[0]
        acc5 = self.top_k_acc(test_preds, test_cates, k = 5)
        print("epoch: %d, test acc: %.2f, top5 acc: %.2f"%(i, acc, acc5))

    def demo(self, test_img):

        test_img = np.reshape(test_img, [-1, self.image_size, self.image_size, 3])
        features = self.sess.run([self.end_points['vgg_16/fc7']],
                feed_dict = {
                    self.input_images: test_img,
                    self.pos_images: test_img, # dummy
                    self.neg_images: test_img, # dummy
                    self.dropout_keep_prob: 1.0
                })

        return np.reshape(features, [-1, 4096])
