#!/usr/bin/env python3
# --*--encoding:utf-8 --*--

import tensorflow as tf
import os
import numpy as np

import model
import cifar10_Input

MAX_STEP=20000
learning_rate =0.1
BATCH_SIZE = 64
#每隔step步打印一下
STEP = 50
#每隔VAL_STEP测试一下
VAL_STEP = 500
VAL_SIZE= 10000
#每隔SNAPSTEP步存储一下模型
SNAPSTEP =5000

data_dir = "/data/dataset/CIFAR-10/cifar-10-batches-bin"
log_dir= "./logs/"
checkpoint_path = os.path.join(log_dir, "train/model.ckpt")



def run_training():

    is_training = tf.placeholder_with_default(True,None)
    image_batch, label_batch = cifar10_Input.read_cifar10(data_dir=data_dir,
                                                          is_training=is_training,
                                                          batch_size=BATCH_SIZE, shuffle=True)
    droprate= tf.placeholder_with_default(0.5,None)

    train_logits = model.inference(image_batch, 10, droprate=droprate)
    train_loss = model.losses(train_logits, label_batch, 10)
    train_op = model.training(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, label_batch, 10)

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()


    with tf.Session(graph=tf.get_default_graph()) as sess:



        train_writer = tf.summary.FileWriter(logdir=os.path.join(log_dir,"train"),graph=sess.graph)
        val_writer = tf.summary.FileWriter(logdir=os.path.join(log_dir,"test"))

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.initialize_local_variables())



        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(os.path.join(log_dir,"train/"))
        #if checkpoint exits ,restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc],feed_dict={is_training:True,
                                                                                         droprate:0.5})


                if step% STEP ==0:
                    print('Step %d,train_loss=%.2f,train_acc=%.2f'
                          %(step,tra_loss,tra_acc))


                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,global_step=step)

                if step % VAL_STEP==0 or(step+1)==MAX_STEP:
                    sum_acc,sum_lo=0,0
                    for i in range(int(VAL_SIZE/BATCH_SIZE)):
                        val_accuracy,val_lo=sess.run([train_acc,train_loss],
                                                     feed_dict={is_training:False,droprate:0})
                        sum_acc +=val_accuracy
                        sum_lo += val_lo


                    print("STEP%d,val_loss=%.2f,val_acc=%.2f"%(step,sum_lo/int(VAL_SIZE/BATCH_SIZE)
                                                               ,sum_acc/int(VAL_SIZE/BATCH_SIZE)))

                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str,global_step=step)

                if step % SNAPSTEP ==0 or (step+1)==MAX_STEP:


                    saver.save(sess, checkpoint_path, global_step=step)



        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

if __name__ =="__main__":
    run_training()