import tensorflow as tf
import numpy as np
import scipy
import os, sys
from tensorflow.contrib import rnn
from tensorflow.core.protobuf import saver_pb2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
from provider import  DVR_Provider

def weight_variable(shape, stddev=0.1):
    init = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(init)

def bias_variable_const(shape, const=0.1):
    init = tf.constant(const, shape=shape)
    return tf.Variable(init)

def bias_variable(shape, stddev=0.1):
    init = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(init)

def conv2d(x, W, stride, padding='VALID'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

class Model(object):

    def __init__(self):
        #self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
        self.y = None
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        self.time_steps = 28
        self.features = 28
        self.lstm_hidden = 512

        self.keep_prob = tf.placeholder(tf.float32)

    def loss(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, \
                name='global_step')

        L2Const = 0.0001
        train_vars = tf.trainable_variables()
#        loss = tf.reduce_mean(tf.square(tf.subtract(self.y_, self.y)) + \
#                tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2Const)
        loss = tf.reduce_mean(tf.square(tf.subtract(self.y_, self.y)))
        #print self.y.shape
        #print self.y_.shape
        ac1_a = tf.abs(tf.subtract(self.y_[:,1], self.y[:,1])) < 0.1
        ac1_a = tf.reduce_mean(tf.cast(ac1_a, tf.float32))
        ac2_a = tf.abs(tf.subtract(self.y_[:,1], self.y[:,1])) < 0.06
        ac2_a = tf.reduce_mean(tf.cast(ac2_a, tf.float32))
        ac3_a = tf.abs(tf.subtract(self.y_[:,1], self.y[:,1])) < 0.04
        ac3_a = tf.reduce_mean(tf.cast(ac3_a, tf.float32))
        ac4_a = tf.abs(tf.subtract(self.y_[:,1], self.y[:,1])) < 0.02
        ac4_a = tf.reduce_mean(tf.cast(ac4_a, tf.float32))
        ac5_a = tf.abs(tf.subtract(self.y_[:,1], self.y[:,1])) < 0.01
        ac5_a = tf.reduce_mean(tf.cast(ac5_a, tf.float32))
        ac6_a = tf.abs(tf.subtract(self.y_[:,1], self.y[:,1])) < 0.001
        ac6_a = tf.reduce_mean(tf.cast(ac6_a, tf.float32))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("angle_6", ac1_a)
        tf.summary.scalar("angle_3", ac2_a)
        tf.summary.scalar("angle_2", ac3_a)
        tf.summary.scalar("angle_1", ac4_a)
        tf.summary.scalar("angle_0.6", ac5_a)
        tf.summary.scalar("angle_0.06", ac6_a)

        ac1_s = tf.abs(tf.subtract(self.y_[:,0], self.y[:,0])) < 0.5
        ac1_s = tf.reduce_mean(tf.cast(ac1_s, tf.float32))
        ac2_s = tf.abs(tf.subtract(self.y_[:,0], self.y[:,0])) < 0.25
        ac2_s = tf.reduce_mean(tf.cast(ac2_s, tf.float32))
        ac3_s = tf.abs(tf.subtract(self.y_[:,0], self.y[:,0])) < 0.15
        ac3_s = tf.reduce_mean(tf.cast(ac3_s, tf.float32))
        ac4_s = tf.abs(tf.subtract(self.y_[:,0], self.y[:,0])) < 0.05
        ac4_s = tf.reduce_mean(tf.cast(ac4_s, tf.float32))

        tf.summary.scalar("speed_10", ac1_s)
        tf.summary.scalar("speed_5", ac2_s)
        tf.summary.scalar("speed_3", ac3_s)
        tf.summary.scalar("speed_1", ac4_s)

        self.merged_summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

        return loss, ac1_a, ac2_a, ac3_a, ac4_a, ac5_a, ac6_a, ac1_s, ac2_s, ac3_s, ac4_s

    def cnn(self):
        # Conv 1
        W1 = weight_variable([5, 5, 3, 24])
        b1 = bias_variable_const([24])
        h1 = tf.nn.relu(conv2d(self.x, W1, 2) + b1)

        # Conv 2
        W2 = weight_variable([5, 5, 24, 36])
        b2 = bias_variable_const([36])
        h2 = tf.nn.relu(conv2d(h1, W2, 2) + b2)

        # Conv 3
        W3 = weight_variable([5, 5, 36, 48])
        b3 = bias_variable_const([48])
        h3 = tf.nn.relu(conv2d(h2, W3, 2) + b3)

        # Conv 4
        W4 = weight_variable([3, 3, 48, 64])
        b4 = bias_variable_const([64])
        h4 = tf.nn.relu(conv2d(h3, W4, 1) + b4)

        # Conv 5
        W5 = weight_variable([3, 3, 64, 64])
        b5 = bias_variable_const([64])
        h5 = tf.nn.relu(conv2d(h4, W5, 1) + b5)

        return h5

    def mlp(self, h):
        # FCL 1
        W_fc1 = weight_variable([1152, 1164])
        b_fc1 = bias_variable_const([1164])
        h_flat = tf.reshape(h, [-1, 1152])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # FCL 2
        W_fc2 = weight_variable([1164, 100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # FCL 3
        W_fc3 = weight_variable([100, 50])
        b_fc3 = bias_variable([50])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob)

        # FCL 4
        W_fc4 = weight_variable([50, 10])
        b_fc4 = bias_variable([10])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
        h_fc4_drop = tf.nn.dropout(h_fc4, self.keep_prob)

        # Output
        W_fc5 = weight_variable([10, 2])
        b_fc5 = bias_variable([2])

        return tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)

    def cnn_mlp(self):
        h = self.cnn()
        self.y = self.mlp(h)
        return self.y

    def train(self, epochs, batch_size, lr = 1e-4, save_path = 'save/cnn_dvr' \
        , logs_path = 'log/cnn_dvr'):
        self.summary_writer = tf.summary.FileWriter(logs_path, \
                graph=tf.get_default_graph())

        loss, ac1_a, ac2_a, ac3_a, ac4_a, ac5_a, ac6_a, ac1_s, ac2_s, ac3_s, ac4_s = self.loss()
        train_step = tf.train.AdamOptimizer(lr).minimize(loss, self.global_step)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        data_input = DVR_Provider()

        for epoch in range(epochs):
            loss_sum = ac1_sum_a = ac2_sum_a = ac3_sum_a = ac4_sum_a = ac5_sum_a = ac6_sum_a = ac1_sum_s = ac2_sum_s = ac3_sum_s = ac4_sum_s = count = 0
            for i in range(int(data_input.num_images/batch_size)):
                xs, ys = data_input.load_one_batch(batch_size, 'train')
                train_step.run(feed_dict={self.x:xs, self.y_:ys, self.keep_prob: 0.8})
                if i % 10 == 0:
                    xs, ys = data_input.load_one_batch(batch_size, 'val')

                    loss_v = loss.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac1_v_a = ac1_a.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac2_v_a = ac2_a.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac3_v_a = ac3_a.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac4_v_a = ac4_a.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac5_v_a = ac5_a.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac6_v_a = ac6_a.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac1_v_s = ac1_s.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac2_v_s = ac2_s.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac3_v_s = ac3_s.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})
                    ac4_v_s = ac4_s.eval(feed_dict={self.x:xs, self.y_:ys, \
                            self.keep_prob: 1.0})

                    loss_sum += loss_v
                    ac1_sum_a += ac1_v_a
                    ac2_sum_a += ac2_v_a
                    ac3_sum_a += ac3_v_a
                    ac4_sum_a += ac4_v_a
                    ac5_sum_a += ac5_v_a
                    ac6_sum_a += ac6_v_a
                    ac1_sum_s += ac1_v_s
                    ac2_sum_s += ac2_v_s
                    ac3_sum_s += ac3_v_s
                    ac4_sum_s += ac4_v_s

                    count += 1
                    print ("Epoch: %d, Step: %d" % (epoch, \
                            epoch * batch_size + i))
                    print ("loss: " + str(loss_v))
                    print ("ac1_a: " + str(ac1_v_a))
                    print ("ac2_a: " + str(ac2_v_a))
                    print ("ac1_s: " + str(ac1_v_s))
                    print ("ac2_s: " + str(ac2_v_s))

                self.summary = self.merged_summary_op.eval(feed_dict={\
                        self.x:xs, self.y_:ys, self.keep_prob:1.0})
                self.summary_writer.add_summary(self.summary, \
                        epoch * data_input.num_images / batch_size + i)

            if i == int(data_input.num_images/batch_size) - 1:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                checkpoint_path = os.path.join(save_path, "model.ckpt")
                filename = self.saver.save(sess, checkpoint_path,\
                        self.global_step)
                print("Model saved in file: %s" % filename)
                print "-------------------------------"
                loss_total = loss_sum / float(count)
                ac1_total_a = ac1_sum_a / float(count)
                ac2_total_a = ac2_sum_a / float(count)
                ac3_total_a = ac3_sum_a / float(count)
                ac4_total_a = ac4_sum_a / float(count)
                ac5_total_a = ac5_sum_a / float(count)
                ac6_total_a = ac6_sum_a / float(count)

                ac1_total_s = ac1_sum_s / float(count)
                ac2_total_s = ac2_sum_s / float(count)
                ac3_total_s = ac3_sum_s / float(count)
                ac4_total_s = ac4_sum_s / float(count)
                print ("loss: " + str(loss_total))
                print ("ac1_a: " + str(ac1_total_a))
                print ("ac2_a: " + str(ac2_total_a))
                print ("ac3_a: " + str(ac3_total_a))
                print ("ac4_a: " + str(ac4_total_a))
                print ("ac5_a: " + str(ac5_total_a))
                print ("ac6_a: " + str(ac6_total_a))
                print "-------------------------------"
                print ("ac1_s: " + str(ac1_total_s))
                print ("ac2_s: " + str(ac2_total_s))
                print ("ac3_s: " + str(ac3_total_s))
                print ("ac4_s: " + str(ac4_total_s))
                print "-------------------------------"

        xs, ys = data_input.load_val_all(16)
        for i in range(len(xs)):
            loss_v += loss.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac1_v_a += ac1_a.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac2_v_a += ac2_a.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac3_v_a += ac3_a.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac4_v_a += ac4_a.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac5_v_a += ac5_a.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac6_v_a += ac6_a.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})

            ac1_v_s += ac1_s.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac2_v_s += ac2_s.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac3_v_s += ac3_s.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
            ac4_v_s += ac4_s.eval(feed_dict={self.x:xs[i], self.y_:ys[i], \
                    self.keep_prob: 1.0})
        print "-----------------------------"
        print "------------Final------------"
        print ("loss: " + str(loss_v / len(xs)))
        print ("ac1_a: " + str(ac1_v_a / len(xs)))
        print ("ac2_a: " + str(ac2_v_a / len(xs)))
        print ("ac3_a: " + str(ac3_v_a / len(xs)))
        print ("ac4_a: " + str(ac4_v_a / len(xs)))
        print ("ac5_a: " + str(ac5_v_a / len(xs)))
        print ("ac6_a: " + str(ac6_v_a / len(xs)))

        print ("ac1_s: " + str(ac1_v_s / len(xs)))
        print ("ac2_s: " + str(ac2_v_s / len(xs)))
        print ("ac3_s: " + str(ac3_v_s / len(xs)))
        print ("ac4_s: " + str(ac4_v_s / len(xs)))

        print "-----------------------------"

#        print("Run the command line:\n" \
#                "--> tensorboard --logdir=./logs " \
#                "\nThen open http://0.0.0.0:6006/ into your web browser")

if __name__ == "__main__":
    test = Model()
    result = test.cnn_mlp()
    print result.shape
    #test.train(100, 64)
