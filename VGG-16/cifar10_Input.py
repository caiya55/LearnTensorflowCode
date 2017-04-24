#!/usr/bin/env python3
# --*-- encoding:utf-8 --*--

import tensorflow as tf
import numpy as np
import os

def read_cifar10(data_dir,is_training,batch_size,shuffle):
    """

    :param data_dir:
    :param is_training:
    :param batch_size:
    :param shuffle:
    :return:
    """
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    img_bytes = img_height * img_width *img_depth



    with tf.name_scope("input") as scope:

        train_filenames = [os.path.join(data_dir,'data_batch_%d.bin'%ii) for ii in np.arange(1,6)]

        val_filenames = [os.path.join(data_dir,'test_batch.bin')]


        train_queue = tf.train.string_input_producer(train_filenames)
        val_queue = tf.train.string_input_producer(val_filenames)

        queue_select = tf.cond(is_training,
                               lambda :tf.constant(0),
                               lambda :tf.constant(1) )
        queue = tf.QueueBase.from_list(queue_select,[train_queue,val_queue])

        reader = tf.FixedLengthRecordReader(label_bytes+img_bytes)
        key,value = reader.read(queue)
        recode_bytes = tf.decode_raw(value,tf.uint8)

        label = tf.slice(recode_bytes,[0],[label_bytes])
        label = tf.cast(label,tf.int32)

        image_raw = tf.slice(recode_bytes,[label_bytes],[img_bytes])
        image_raw = tf.reshape(image_raw,[img_depth, img_height, img_width])
        image = tf.transpose(image_raw,[1,2,0])

        image = tf.cast(image,tf.float32)

        image = tf.image.per_image_standardization(image)

        if shuffle:
            images, label_batch= tf.train.shuffle_batch([image,label],
                                                   batch_size=batch_size,
                                                   num_threads=16,
                                                   capacity=512+3*batch_size,
                                                   min_after_dequeue=512,
                                                   allow_smaller_final_batch=True)
        else:
            images, label_batch = tf.train.batch([image, label],
                                            batch_size=batch_size,
                                            num_threads=16,
                                            capacity=512 + 3*batch_size,
                                            allow_smaller_final_batch=True)
        label_batch = tf.cast(label_batch,tf.int32)

        return images,label_batch


