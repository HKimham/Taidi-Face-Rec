import cv2
import os
from os.path import join
import numpy as np
import random
# import logging as log
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# gpu 版本的配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

SIZE = 64
x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 1])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob_5_percent = tf.placeholder(tf.float32)
keep_prob_75_percent = tf.placeholder(tf.float32)


def weightVariable(shape):
    ''' build weight variable'''
    # init = tf.random_normal(shape, stddev=0.01)
    init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    ''' build bias variable'''
    # init = tf.random_normal(shape)
    init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def conv2d(x, W):
    ''' conv2d by 1, 1, 1, 1'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    ''' max pooling'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    ''' drop out'''
    return tf.nn.dropout(x, keep)

def cnnLayer(classnum):
    ''' create cnn layer'''
    # 第一层
    W1 = weightVariable([3, 3, 1, 32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5_percent) # 32 * 32 * 32 多个输入channel 被filter内积掉了

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5_percent) # 16 * 16 * 64

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5_percent) # 8 * 8 * 64

    # 全连接层
    Wf = weightVariable([8 * 8 * 64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8 * 8 * 64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75_percent)

    # 输出层
    Wout = weightVariable([512, classnum])
    bout = weightVariable([classnum])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def train(train_x, train_y, tfsavepath):
    ''' train'''
    # log.debug('train')
    # out = cnnLayer(train_y.shape[1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 200
        Epoch = 100
        num_batch = len(train_x) // batch_size + 1
        for n in range(Epoch):
            # r = np.random.permutation(len(train_x))
            # train_x = train_x[r, :]
            # train_y = train_y[r, :]

            for i in range(num_batch):
                if i == num_batch - 1 :
                    batch_x = train_x[i*batch_size : :]                   
                    batch_y = train_y[i*batch_size : :]
                else:
                    batch_x = train_x[i*batch_size : (i+1)*batch_size]
                    batch_y = train_y[i*batch_size : (i+1)*batch_size]
                _, loss = sess.run([train_step, cross_entropy],\
                                   feed_dict={x_data:batch_x, y_data:batch_y,
                                              keep_prob_5_percent:0.5, keep_prob_75_percent:0.75})
            print('Epoch: {: <2d}  Loss: {}'.format(n,loss))
        saver.save(sess, tfsavepath)



def validate(test_x, classnum, tfsavepath):
    ''' validate '''
    # out = cnnLayer(classnum)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, tfsavepath)
        result = sess.run(tf.argmax(out, 1),
                       feed_dict={x_data: test_x,
                                  keep_prob_5_percent:1.0, keep_prob_75_percent: 1.0})
        # 返回分类结果的编号
        return result 

def acc(x, labels, tfsavepath):
    # out = cnnLayer(labels.shape[1])
    predict = np.array([])
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:    
        saver.restore(sess, tfsavepath)
        batch_size = 200
        num_batch = len(x) // batch_size + 1
        for i in range(num_batch): 
            if i == num_batch - 1 :
                batch_x = x[i*batch_size : :]
            else:
                batch_x = x[i*batch_size : (i+1)*batch_size]
            result = sess.run(tf.argmax(out, 1),                        
                                feed_dict={x_data: batch_x,                                  
                                             keep_prob_5_percent:1.0, keep_prob_75_percent: 1.0})
            predict = np.r_[predict,result]

    y = np.argmax(labels,1)
    acc = np.sum(np.equal(predict,y)) / labels.shape[0]
    return acc 
   
    

if __name__ == '__main__':
    npy_path = './faceImage_npy'
    tfsavepath = './model/face.ckpt'
    if not os.path.exists('./model'):
        os.mkdir('./model')

    data = np.load(join(npy_path,'data.npy'))
    data = data.reshape(-1,SIZE,SIZE,1).astype(np.uint8)
    print("load data.npy done")
    labels = np.load(join(npy_path,'labels.npy'))
    print("load labels.npy done")

    # One-Hot 编码
    label_encoder= LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_encoder.fit_transform(labels)
    labels = onehot_encoder.fit_transform(integer_encoded.reshape(-1,1))

    num = data.shape[0]
    train_index = int(num*0.8)
    train_data = data[0:train_index,:,:,:]
    train_labels =  labels[0:train_index,:]

    test_data = data[train_index::,:,:,:]
    test_labels =  labels[train_index::,:]

    out = cnnLayer(10) # 不能放在函数内，否则每运行一次函数会重新定义一次变量，这样行的变量名就会改变

    # 训练
    # log.debug('generateface')
    train(train_data, train_labels, tfsavepath)
    # log.debug('training is over, please run again')

    # 训练集精确度
    tr_acc = acc(train_data, train_labels, tfsavepath = tfsavepath)
    print('---------------------------------------------')
    print('训练集精确度为 {:.2f}%'.format(tr_acc*100))
    print('---------------------------------------------')


    # 测试集精确度
    te_acc = acc(test_data, test_labels, tfsavepath = tfsavepath)
    print('---------------------------------------------')
    print('测试集精确度为 {:.2f}%'.format(te_acc*100))
    print('---------------------------------------------')

