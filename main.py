from trainer import Trainer
from models import closest_l2_distance
import tensorflow as tf
import pickle, os
import numpy as np

batch_size = 25
image_size = 224
lr = 0.005
epoch = 10
checkpoint_dir = './data/'

flags = tf.app.flags
flags.DEFINE_boolean("feature_learning", True, "True, if you want to train feature extractor, otherwise False")
flags.DEFINE_boolean("retrieval", False, "True, if you want to search similar images by query image")
flags.DEFINE_boolean("fine_tune", False, "True, if you want to train from existing model")
FLAGS = flags.FLAGS

def main(_):
    
    if FLAGS.feature_learning == True:
        train_module = Trainer(batch_size, image_size, lr, epoch)

        train_module.build_model()

        train_module.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and FLAGS.fine_tune:
            train_module.saver.restore(train_module.sess, ckpt)
        else:
            print(" [!] Not found checkpoint")

        train_module.sess.run([tf.global_variables_initializer(), 
                tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess = train_module.sess, coord = coord)

        train_module.train()

        train_module.test()
        tf.reset_default_graph()
        train_module.sess.close()

    if FLAGS.retrieval == True:
        
        demo_module = Trainer(batch_size, image_size, 0., 1)
        demo_module.build_model()

        demo_module.saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            demo_module.saver.restore(demo_module.sess, ckpt)
            print(" [*] Parameter restored from %s"%ckpt)
        else:
            print(" [!] Not found checkpoint, process terminates")
            return -1

        demo_module.sess.run([tf.global_variables_initializer(), 
                tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess = demo_module.sess, coord = coord)
        from scipy import misc

        if os.path.exists("./data/retrieval.pkl"):
            fin = open('./data/retrieval.pkl', 'rb')
            feature_and_paths = pickle.load(fin)
            fin.close()
        else:
            test_files = demo_module.d_loader._d['test_files']

            feature_and_paths = []
            for f in test_files:
                img = misc.imread(f)
                img = misc.imresize(img, (image_size, image_size)) * (1. / 255) - 0.5
                
                feature = demo_module.demo(test_img = img)
                feature_and_paths.append( (feature, f) )

            fout = open("./data/retrieval.pkl", "wb")
            pickle.dump(feature_and_paths, fout)
            fout.close()
            print(" [*] retrieval data saved at ./data/retrieval.pkl")
        print(" [*] retrieval ready")

        val_files = demo_module.d_loader._d['test_files']
        index = np.random.randint(10000, size = 1)

        f = val_files[index[0]]
        print(" example: %s"%f)
        img = misc.imread(f)
        img = misc.imresize(img, (image_size, image_size)) * (1. / 255) - 0.5
        feature = demo_module.demo(test_img = img)

        paths = closest_l2_distance(feature_and_paths, feature)
        print(paths)


if __name__ == '__main__':
    tf.app.run()
