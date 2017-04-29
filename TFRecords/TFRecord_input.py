#!/usr/bin/env python3
# --*-- encoding:utf-8 --*--

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io

def get_file(file_dir):
    """
    get full image directory and correspond labels
    :param file_dir: 
    :return: 
    """
    images =[]
    temp =[]
    for root ,sub_folders,files in os.walk(file_dir):
        #image directories
        for name in files:
            images.append(os.path.join(root,name))
        #get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root,name))

    labels =[]
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]

        if letter =='A':
            labels = np.append(labels,n_img*[1])
        elif letter =="B":
            labels = np.append(labels,n_img*[2])
        elif letter =='C':
            labels = np.append(labels,n_img*[3])
        elif letter =="D":
            labels = np.append(labels,n_img*[4])
        elif letter =="E":
            labels = np.append(labels,n_img*[5])
        elif letter =="F":
            labels = np.append(labels,n_img*[6])
        elif letter =="G":
            labels = np.append(labels,n_img*[7])
        elif letter =="H":
            labels = np.append(labels,n_img*[8])
        elif letter =="I":
            labels =np.append(labels,n_img*[9])
        else:
            labels = np.append(labels,n_img*[10])

    #shuffle
    temp = np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list,label_list

def int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list= tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images,labels,save_dir,name):
    """
    convert all images and labels to one tfrecord file
    :param images: 
    :param labels: 
    :param save_dir: 
    :param name: 
    :return: 
    """
    filename = os.path.join(save_dir,name+".tfrecords")
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Image size %d does not '
                         'match label size %d'%(images.shape[0],n_samples))

    #wait some time
    writer = tf.python_io.TFRecordWriter(filename)
    print("\n Transform start....")
    for i in np.arange(0,n_samples):
        try:
            image = io.imread(images[i])
            image_raw = image.tostring()
            label= int(labels[i])
            example = tf.train.Example(features =tf.train.Features(feature={'label':int64_feature(label),
                                                                             "image_raw":bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print("could not read :",images[i])
            print("error:%s"%e)
            print('Skip it')
    writer.close()
    print("Transform done!")

def read_and_decode(tfrecords_file,batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example,
                                           features={"label":tf.FixedLenFeature([],tf.int64),
                                                     "image_raw":tf.FixedLenFeature([],tf.string),})
    image = tf.decode_raw(img_features['image_raw'],tf.uint8)
    ################################################################
    #
    #put dataaugmentation here
    ################################################################

    image = tf.reshape(image,[28,28])
    label = tf.cast(img_features['label'],tf.int32)
    image_batch, label_batch = tf.train.batch([image,label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity=2000)
    return image_batch,tf.reshape(label_batch,[batch_size])




def plot_images(images,labels,number):
    """
    plot one batch size
    :param images: 
    :param labels: 
    :return: 
    """
    for i in np.arange(0,number):
        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.title(chr(ord('A')+labels[i]-1),fontsize = 14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

if __name__=="__main__":
    test_dir ="/data/dataset/notMNIST_large"
    save_dir ="./"
    BATCH_SIZE = 25

    #create the data: you just need to run it ONCE
    if  not os.path.exists("./test.tfrecords"):
        name_test = "test"
        images, labels = get_file(test_dir)
        convert_to_tfrecord(images, labels, save_dir, name_test)




    tfrecords_file = "./test.tfrecords"
    image_batch, label_batch = read_and_decode(tfrecords_file,batch_size=BATCH_SIZE)

    #显示一个batch中的图片
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1:
                images,labels = sess.run([image_batch,label_batch])
                plot_images(images,labels,BATCH_SIZE)
                i+=1
        except tf.errors.OutOfRangeError:
            print('Done')
        finally:
            coord.request_stop()
        coord.join(threads)
