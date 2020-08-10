#_*_coding:utf-8_*_
import tensorflow as tf # 导入tensorflow模块
from tensorflow.examples.tutorials.mnist import input_data # 导入mnist数据集
from PIL import Image # 从PIL库中导入Image模块
import numpy as np # 导入numpy科学计算模块，取个别名
import matplotlib.pyplot as plt # 导入matplotlib.pyplot画图模块，取个别名
import warnings # 导入警告，不影响程序的正常运行
warnings.filterwarnings('ignore')

# 1.数据加载
mnist = input_data.read_data_sets("./data",one_hot=True)
# 训练集
train_img = mnist.train.images # [55000,784]
# print(train_img.shape)
train_label = mnist.train.labels # [55000,10]
# print(train_label.shape)
# 测试集
test_img = mnist.test.images # [10000,784]
# print(test_img.shape)
test_label = mnist.test.labels # [10000,10]
# print(test_label.shape)
print("mnist data ready")

# 2.模型构建
# (1)超参数
# 输入大小：28X28=784
input_size = train_img.shape[1] # 28X28=784
# print(input_size)
# 输出大小：10
output_size = train_label.shape[1] # 10
# print(output_size)
# 占位符
X = tf.placeholder(dtype=tf.float32,shape=[None,input_size],name="X")
Y = tf.placeholder(dtype=tf.float32,shape=[None,output_size],name="Y")
# 学习率
learning_rate = 1e-3
# 批数量大小
batch_size = 128
# 迭代次数
epochs = 50
# 迭代多少次，显示一下相关信息
display_step = 10
# 设置一个阈值
good_acc = 0.99
# 训练集和测试集的准确率连续5次达到0.99以上，模型训练停止。
successive_limit = 5

# (2)构建lenet
def lenet():
    # X的shape：[None,784]
    with tf.name_scope('Inputlayer'):
        # 输入的是四维的
        input = tf.reshape(X,shape=[-1,28,28,1])
        tf.summary.image('input',input)
    # 第一层卷积 + 激活 + 池化
    with tf.name_scope('Conv_1_layer'):
        # wc1：卷积核 [filter_height,filter_width,input_channel,output_channel]
        wc1 = tf.Variable(tf.random_normal([5,5,1,20],stddev=0.1),name='wc1')
        tf.summary.histogram('wc1', wc1)
        # wb1：偏置项
        wb1 = tf.Variable(tf.random_normal([20],stddev=0.1),name='wb1')
        tf.summary.histogram('wb1', wb1)
        # 卷积：tf.nn.conv2d(input,filter,strides,padding)
        # input：指定需要做卷积的输入图像，输入是一个tensor，形状为[batch_size,height,width,inchannel] 数据类型为float32或float64
        # filter：卷积核，要求输入的是一个tensor，形状为[filter_height,filter_width,in_channel,out_channel] 数据类型与input一致，并且第三位参数和input的第四位参数一致。
        # strides：卷积核移动的步长，是一个一维的向量，长度为4，[batch_step,height_step,width_step,in_channel_step]。一般情况下，第一个和最后一个都为1。
        # padding：string类型的数据，只能是SAME和VALID中的一种，可以做填充，输入的
        conv1 = tf.nn.conv2d(input=input,filter=wc1,strides=[1,1,1,1],padding='VALID',name='conv1')
        # tf.summary.image('conv1', conv1)
        # print(conv1)
        # conv2 = tf.nn.conv2d(input=input,filter=wc1,strides=[1,1,1,1],padding='SAME')
        # print(conv2)
        # Tensor("Conv_1_layer/conv1:0", shape=(?, 24, 24, 20), dtype=float32)
        # Tensor("Conv_1_layer/Conv2D:0", shape=(?, 28, 28, 20), dtype=float32)
        # 激活：修正线性relu
        conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1,wb1),name='conv1_relu')
        # tf.summary.image('conv1_relu', conv1_relu)
        # 最大值池化
        # ksize：[batch,height,width,in_channel]
        conv1_maxpooling = tf.nn.max_pool(value=conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='conv1_maxpooling')
        # tf.summary.image('conv1_pooling', conv1_maxpooling)

    # 第二层卷积 + 激活 + 池化
        # 第一层卷积 + 激活 + 池化
    with tf.name_scope('Conv_2_layer'):
        # wc2卷积核 [height,width,input_channel,output_channel]
        wc2 = tf.Variable(tf.random_normal([5, 5, 20, 50], stddev=0.1), name='wc2')
        tf.summary.histogram('wc2', wc2)
        wb2 = tf.Variable(tf.random_normal([50], stddev=0.1), name='wb2')
        tf.summary.histogram('wb2', wb2)
        conv2 = tf.nn.conv2d(input=conv1_maxpooling, filter=wc2, strides=[1, 1, 1, 1], padding='VALID', name='conv2')
        # tf.summary.image('conv2', conv2)
        conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2, wb2), name='conv2_relu')
        # tf.summary.image('conv2_relu', conv2_relu)
        conv2_maxpooling = tf.nn.max_pool(value=conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID', name='conv2_maxpooling')
        # tf.summary.image('conv2_pooling', conv2_maxpooling)
    # 因为卷积计算的是四维的数据，而BP计算的是二维的数据，所以需要把四维的数据拉伸成二维的数据。
    # 拉伸 flatten
    with tf.name_scope('flatten'):
        conv_shape = conv2_maxpooling.get_shape().as_list()
        # print(conv_shape) # [None, 4, 4, 50]
        flatten_shape = conv_shape[1]*conv_shape[2]*conv_shape[3]
        # print(flatten_shape) # 800
        flatten = tf.reshape(tensor=conv2_maxpooling,shape=[-1,flatten_shape])
        # print(flatten.shape) # (-1, 800)
    # FC + 激活
    with tf.name_scope('fc1'):
        wfc_1 = tf.Variable(tf.random_normal([flatten_shape,500]),name='wfc_1')
        tf.summary.histogram('wfc_1', wfc_1)
        bfc_1 = tf.Variable(tf.random_normal([500]),name='bfc_1')
        tf.summary.histogram('bfc_1', bfc_1)
        fc_1 = tf.nn.relu(tf.add(tf.matmul(flatten,wfc_1),bfc_1),name='fc1')
        tf.summary.histogram('fc_1', fc_1)
    # FC
    with tf.name_scope('outLayer'):
        wfc_2 = tf.Variable(tf.random_normal([500,output_size]),name='wfc_2')
        tf.summary.histogram('wfc_2', wfc_2)
        bfc_2 = tf.Variable(tf.random_normal([output_size]),name='bfc_2')
        tf.summary.histogram('bfc_2', bfc_2)
        output = tf.add(tf.matmul(fc_1,wfc_2),bfc_2,name='output')
        # print(output) # Tensor("outLayer/output:0", shape=(?, 10), dtype=float32)
    return output

# 3.模型训练
def train():
    y_ = lenet()
    print('lenet ready')
    # (1)损失
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))
    tf.summary.scalar('loss', cost)

    # (2)准确率
    corr=tf.equal(tf.argmax(y_,axis=1), tf.argmax(Y,axis=1))
    acc = tf.reduce_mean(tf.cast(corr, dtype=tf.float32))
    tf.summary.scalar('accuary', acc)

    # (3)优化器
    global_step=tf.Variable(0.0)
    optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)

    # (4)可视化，已在模型构建时加入。
    # (5) 会话执行图
    print('run ready')
    with tf.Session() as sess:
        # 变量初始化
        tf.global_variables_initializer().run()
        # 构建持久化模型的对象
        saver=tf.train.Saver()
        # 构建summary文件写的对象
        writer=tf.summary.FileWriter(logdir='log',graph=sess.graph,session=sess)
        # 收集summary的信息
        merge=tf.summary.merge_all()
        # 达到符合要求的次数
        successive_count = 0
        #  迭代训练
        for i in range(epochs):
            # 循环完一个训练集，再进入下一次迭代
            # mnist.train.num_examples 训练集的样本数
            for _ in range(mnist.train.num_examples//batch_size):
                batch_train_x,batch_train_y=mnist.train.next_batch(batch_size)
                # 执行优化器，优化参数
                sess.run(optm,feed_dict={X:batch_train_x,Y:batch_train_y})
            # 每display_step 迭代，输出一下测试集和训练各自的损失和准确率，并进行持久化模型操作以及summary记录操作
            if i%display_step==0:
                # 使用训练集，输出损失和准确率
                train_acc, train_loss,train_merge = sess.run([acc, cost,merge], feed_dict={X: batch_train_x, Y: batch_train_y})
                print('Train,Epoch:{}/{},Cost:{},Acc:{}'.format(i,epochs,train_loss,train_acc))
                # 使用测试集，输出损失和准确率
                test_acc,test_loss=sess.run([acc,cost],feed_dict={X:test_img,Y:test_label})
                print('Test,Epoch:{}/{},Cost:{},Acc:{}'.format(i,epochs,test_loss,test_acc))
                print('*'*100)
                writer.add_summary(train_merge,global_step=i)
                # 并且记录保存每display_step的模型结果，方便下一次断点续训
                saver.save(sess, 'model/lenet.ckpt',global_step=i)
                if test_acc>good_acc and train_acc>good_acc:
                    successive_count += 1
                    if successive_count > successive_limit:
                        # 执行持久化模型操作
                        saver.save(sess,'model/lenet.ckpt')
                        # 一般来说，当准确率达标的时候，就可以不再训练
                        break
        writer.close()
        print('train end')
train()
# 4.模型预测
def predict():
    # 加载模型
    with tf.Session() as sess:
        # 判断模型文件是否存在
        ckpt = tf.train.get_checkpoint_state('model')
        if ckpt and ckpt.model_checkpoint_path:
            # 加载图
            saver=tf.train.import_meta_graph('model/lenet.ckpt-150.meta')
            # 加载模型参数
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('模型文件不存在')
        # 获取默认的图
        graph=tf.get_default_graph()
        # 加载output张量
        predict_out=graph.get_tensor_by_name('outLayer/output:0')
        # 加载X的占位符
        X=graph.get_tensor_by_name('X_1:0')
        # 选择一个训练集的图片和一个测试集的图片，来看一下预测的效果
        img1,label1=train_img[0],train_label[0] # img1 shape=[784,]
        img2, label2 = test_img[0], test_label[0]
        img_data=[img1,img2] # [2,784]
        # print(np.array(img_data).shape)
        predict_=sess.run(predict_out,feed_dict={X:np.array(img_data)})
        predict_label=np.argmax(predict_,axis=1)
        true_label=np.argmax([label1,label2],axis=1)

        # 画一下这两个图，并且输出预测的结果和真实的结果
        img1_np=np.reshape(img1,(28,28))
        # 矩阵转换为图片对象
        new_img1=Image.fromarray(img1_np)
        plt.matshow(new_img1)
        plt.show()
        print("train_img")
        print("True_label:{},Predict label:{}".format(true_label[0],predict_label[0]))

        img2_np = np.reshape(img2, (28, 28))
        # 矩阵转换为图片对象
        new_img2 = Image.fromarray(img2_np)
        plt.matshow(new_img2)
        plt.show()
        print("test_img")
        print("True_label:{},Predict label:{}".format(true_label[1], predict_label[1]))

if __name__ == '__main__':
    is_train=True   # True执行训练；False执行预测。
    if is_train:
        train()
    else:
        predict()
