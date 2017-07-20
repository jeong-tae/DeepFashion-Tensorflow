from trainer import Trainer
import tensorflow as tf

batch_size = 50
image_size = 224
lr = 0.005
epoch = 10
ckpt_path = './data/deepfashion.ckpt'

flags = tf.app.flags
flags.DEFINE_boolean("feature_learning", True, "True, if you want to train feature extractor, otherwise False")
flags.DEFINE_boolean("retrieval", True, "True, if you want to search similar images by query image")

def main(_):
    
    train_module = Trainer(batch_size, image_size, lr, epoch)

    train_module.build_model()

    train_module.saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(train_module.sess, ckpt.model_checkpoint_path)
    else:
        print(" [!] Not found checkpoint")

    train_module.sess.run([tf.global_variables_initializer(), 
            tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess = train_module.sess, coord = coord)

    train_module.train()

    train_module.test()



if __name__ == '__main__':
    tf.app.run()
