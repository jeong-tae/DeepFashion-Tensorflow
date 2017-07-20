import os, random
import numpy as np
import tensorflow as tf
import pickle

def read_and_decode(root, input_queue):    
    image = tf.read_file(input_queue[0])
    example = tf.image.decode_png(image, channels = 3)
    category = input_queue[1]
    attr = input_queue[2]
    return example, category, attr

def queue_ready(files, categories, attrs, nepochs = 10):
    f = tf.convert_to_tensor(files, dtype=tf.string)
    c = tf.convert_to_tensor(categories, dtype=tf.int32)
    a = tf.convert_to_tensor(attrs, dtype=tf.int32)
    return tf.train.slice_input_producer([f, c, a], num_epochs = nepochs)

class Data_loader(object):
    def __init__(self, root, cate_path, attr_path, partition_path,
            batch_size, scale_size, pkl_path = './data/fashion.pkl'):

        self.batch_size = batch_size
        self.scale_size = scale_size

        self.get_data(root, cate_path, attr_path, partition_path,
                batch_size, pkl_path)

    def get_queue(self, min_queue_examples, split = 'train', nepochs = 10, shuffle = True):
        split = split.lower()
        _queue_ready = queue_ready(self._d[split + '_files'],
                self._d[split + '_category'], self._d[split + '_attr'], nepochs = nepochs)

        content = tf.read_file(_queue_ready[0])
        image = tf.image.decode_png(content, channels = 3)
        category = _queue_ready[1]
        attr = _queue_ready[2]

        resized_image = tf.image.resize_images(image, [self.scale_size, self.scale_size])
        # convert from [0, 255] to [-0.5, 0.5] float.
        resized_image = tf.cast(resized_image, tf.float32) * (1. / 255) - 0.5

        num_preprocess_threads = 4

        if not shuffle:
            feature, cate, att = tf.train.batch([resized_image, category, attr],
                    batch_size = self.batch_size,
                    num_threads = num_preprocess_threads,
                    capacity = min_queue_examples + 10 * self.batch_size,
                    allow_smaller_final_batch = True)
        else:
            feature, cate, att = tf.train.shuffle_batch([resized_image, category, attr],
                    batch_size = self.batch_size,
                    num_threads = num_preprocess_threads,
                    capacity = min_queue_examples + 10 * self.batch_size,
                    min_after_dequeue = min_queue_examples)

        return feature, cate, att

    def get_data(self, root, cate_path, attr_path, partition_path, 
            batch_size, scale_size, pkl_path = './data/fashion.pkl'):

        if pkl_path != None and os.path.exists(pkl_path):
            f = open(pkl_path, 'rb')
            self._d = pickle.load(f)
            f.close()
            print(" [*] Number of train files: %d"%len(self._d['train_files']))
            print(" [*] Number of val files: %d"%len(self._d['val_files']))
            print(" [*] Number of test files: %d"%len(self._d['test_files']))
            print(" [*] file loaded at %s"%pkl_path)
        else:

            list_category_img = open(root + cate_path).readlines()
            list_attr_img = open(root + attr_path).readlines()

            train_val_test_idx = open(root + partition_path).readlines()

            train_files, val_files, test_files = [], [], []
            train_category, val_category, test_category = [], [], []
            train_attr, val_attr, test_attr = [], [], []

            for idx in range(2, len(train_val_test_idx)):
                path, split = train_val_test_idx[idx].split()
                category = list_category_img[idx].split()[1]
                attr = list_attr_img[idx].split()[1:]
                path = root + path[0].upper() + path[1:]
                attr = [n if n != -1 else 0 for n in attr]
                if split.lower() == 'train':
                    train_files.append(path)
                    train_category.append(category)
                    train_attr.append(attr)
                elif split.lower() == 'val':
                    val_files.append(path)
                    val_category.append(category)
                    val_attr.append(attr)
                else:
                    test_files.append(path)
                    test_category.append(category)
                    test_attr.append(attr)

            train_N, val_N, test_N = 50000, 10000, 10000
            train_indices, val_indices, test_indices = np.arange(len(train_files)), np.arange(len(val_files)), np.arange(len(test_files))
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            train_indices = train_indices[:train_N]
            val_indices = val_indices[:val_N]
            test_indices = test_indices[:test_N]

            train_files = np.array(train_files)[train_indices]
            val_files = np.array(val_files)[val_indices]
            test_files = np.array(test_files)[test_indices]
            train_category = np.array(train_category)[train_indices]
            val_category = np.array(val_category)[val_indices]
            test_category = np.array(test_category)[test_indices]
            train_attr = np.array(train_attr)[train_indices]
            val_attr = np.array(val_attr)[val_indices]
            test_attr = np.array(test_attr)[test_indices]

            print(" [*] Number of train files: %d"%len(train_files))
            print(" [*] Number of val files: %d"%len(val_files))
            print(" [*] Number of test files: %d"%len(test_files))

            self._d = {'train_files': train_files,
                'val_files': val_files,
                'test_files': test_files,
                'train_category': train_category,
                'val_category': val_category,
                'test_category': test_category,
                'train_attr': train_attr,
                'val_attr': val_attr,
                'test_attr': test_attr,
            }

            f = open(pkl_path, 'wb')
            pickle.dump(self._d, f)
            f.close()
            print(" [*] filed saved at %s"%pkl_path)
