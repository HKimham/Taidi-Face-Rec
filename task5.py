import cv2
import os
from mtcnn.mtcnn import MTCNN
from os.path import join
import tensorflow as tf
import numpy as np
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


npy_path = './faceImage_npy'
labels = np.load(join(npy_path,'labels.npy'))
print("load labels.npy done")

# One-Hot 编码
label_encoder= LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = label_encoder.fit_transform(labels)
labels = onehot_encoder.fit_transform(integer_encoded.reshape(-1,1))


tfsavepath = './model/face.ckpt'
if not os.path.exists('./model'):
    os.mkdir('./model')
detector = MTCNN()
cap = cv2.VideoCapture(1) 

out = cnnLayer(10)
saver = tf.train.Saver()
with tf.Session(config=config) as sess:  
    saver.restore(sess, tfsavepath)
    while True:
        ret, frame = cap.read()
        if ret:  # 若不出错则显示图像
            faces = detector.detect_faces(frame)
            if len(faces) == 0:
                continue
            x, y, width, height = faces[0]['box']
            gray_img = cv2.cvtColor(frame[y : y + height, x : x + width,:], cv2.COLOR_BGR2GRAY)
            if gray_img is None:
                continue
            gray_img = cv2.resize(gray_img,(64,64),interpolation=cv2.INTER_LINEAR)
            gray_img = gray_img.reshape(1,64,64,1)

            name_index = validate(gray_img, classnum = 10, tfsavepath=tfsavepath)
            name = label_encoder.inverse_transform([name_index])

            cv2.putText(frame, name[0], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
            frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.imshow("myface", frame)  

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                print("手动退出!")
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

