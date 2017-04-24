#!/user/bin/env python3
# --*-- encoding:utf-8 --*--

import tensorflow as tf
import cifar10_Input

def inference(images,n_classes,droprate):
    """
    实现vgg16的网络结构
    :param images:
    :param c_classes:
    :return:
    """
    with tf.variable_scope('conv1') as scope:
        conv1_1 = tf.layers.conv2d(images, filters=64, kernel_size=[3, 3],
                                   strides=(1, 1), padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv1/1")
        conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=[3, 3],
                                   strides=[1, 1], padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv1/2")
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=[2, 2], strides=[2, 2],
                                        padding="SAME", name="conv1/pool")
        tf.summary.histogram("conv1", pool1)
        #tf.summary.image('conv1',pool1[0])

    with tf.variable_scope('conv2') as scope:
        conv2_1 = tf.layers.conv2d(pool1, filters=128, kernel_size=[3, 3],
                                   strides=(1, 1), padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv2/1")
        conv2_2 = tf.layers.conv2d(conv2_1, filters=128, kernel_size=[3, 3],
                                   strides=(1, 1), padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv2/2")
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=[2, 2], strides=(1, 1),
                                        padding='SAME', name="conv2/pool")
        tf.summary.histogram('conv2', pool2)
        #tf.summary.image('conv2',pool2[0])

    with tf.variable_scope('conv3') as scope:
        conv3_1 = tf.layers.conv2d(pool2, filters=256, kernel_size=[3, 3],
                                   strides=(1, 1), padding='SAME',
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv3/1')
        conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=[3, 3],
                                   strides=(1, 1), padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv3/2')
        conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=[3, 3],
                                   strides=(1, 1), padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv3/3")
        pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=[2, 2], strides=[2, 2],
                                        padding="SAME", name="conv3/pool")
        tf.summary.histogram('conv3', pool3)
        #tf.summary.image('conv3',pool3[0])

    with tf.variable_scope('conv4') as scope:
        conv4_1 = tf.layers.conv2d(pool3, filters=512, kernel_size=[3, 3],
                                   strides=(1, 1), padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv4/1')
        conv4_2 = tf.layers.conv2d(conv4_1, filters=512, kernel_size=[3, 3],
                                   strides=(1, 1), padding='SAME',
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv4/2")
        conv4_3 = tf.layers.conv2d(conv4_2, filters=512, kernel_size=[3, 3],
                                   strides=(1, 1), padding='SAME',
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv4/3")
        pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=[2, 2], strides=[2, 2],
                                        padding='SAME', name="conv4/pool")
        tf.summary.histogram('conv4', pool4)
        #tf.summary.image('conv5',pool4[0])

    with tf.variable_scope('conv5') as scope:
        conv5_1 = tf.layers.conv2d(pool4, filters=512, kernel_size=[3, 3],
                                   strides=(1, 1), padding="SAME",
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv5/1")
        conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size=[3, 3],
                                   strides=(1, 1), padding='SAME',
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv5/2')
        conv5_3 = tf.layers.conv2d(conv5_2, filters=512, kernel_size=[3, 3],
                                   strides=(1, 1), padding='SAME',
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv5/3')
        pool5 = tf.layers.max_pooling2d(conv5_3, pool_size=[2, 2], strides=2, padding="SAME", name="conv5/pool")
        tf.summary.histogram('conv5', pool5)
        #tf.summary.image('conv5',pool5[0])

    with tf.variable_scope('fc') as scope:
        pool5_flatten = tf.contrib.layers.flatten(pool5)
        fc6 = tf.layers.dense(inputs=pool5_flatten, units=4096, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc6")
        dropout6 = tf.layers.dropout(fc6, rate=droprate, name="fc/drop6")

        fc7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc7")

        dropout7 = tf.layers.dropout(fc7, rate=droprate, name="fc/drop7")
        logit = tf.layers.dense(dropout7, units=n_classes)
        tf.summary.histogram("logit", logit)


    return logit


def losses(logit , labels, n_classes):
    """

    :param logit:
    :param labels:
    :param n_classes:
    :return:
    """
    with tf.name_scope('loss') as scope:
        labels = tf.one_hot(labels, n_classes)
        labels = tf.cast(labels,tf.int32)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=labels)

        loss = tf.reduce_mean(cross_entropy,name="loss")

        tf.summary.scalar(scope+"/loss",loss)

    return loss

def training(loss,learning_rate):
    """

    :param loss:
    :param learning_rate:
    :return:
    """
    with tf.name_scope('optimizer') as scope:
        starter_learning_tate = learning_rate
        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_tate,
                                                   global_step=global_step,
                                                   decay_steps=1000, decay_rate=0.9, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                   global_step=global_step,
                                                   learning_rate=learning_rate,
                                                   optimizer='SGD')
        tf.summary.scalar("global_step",global_step)


    return train_op

def evaluation(logits, labels,n_classes):
    """

    :param logits:
    :param labels:
    :param n_classes:
    :return:
    """
    with tf.name_scope('accuracy') as scope:
        # labels = tf.argmax(labels,axis=1)
        labels = tf.squeeze(labels)
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope + '/accuracy', accuracy)
    return accuracy

