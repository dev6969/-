import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.layers import dense,conv2d,batch_normalization,conv2d_transpose
import numpy as np
import timeit as T

class UNET():        
    # crop_acd_concat FROM_TOP_LEFT_CROP
    def crop_and_concat(self,x1,x2):
        with tf.name_scope("crop_and_concat"):
            x1_shape = tf.shape(x1)
            x2_shape = tf.shape(x2)
            x_crop = tf.slice(x1,[0,0,0,0],x2_shape)
            return tf.concat([x_crop,x2],-1)

    
    
    def conv_net(self,inputs,filters,
                 kernel_size=[3,3],strides=[1,1],
                 activation=tf.nn.relu,name=None):
        
        conv2 = conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, name=name)
#         tensor =  batch_normalization(tensor)
        if name !=None : print("{}:{}".format(name,conv2.shape))
        return conv2 
        
    def deconv_net(self,input1,input2,filters,
                 kernel_size=[4,4],strides=[2,2],
                 activation=tf.nn.relu,name=None):
        
        concat = self.crop_and_concat(input1,input2)        
        deconv2 = conv2d_transpose(inputs=concat, filters=filters,kernel_size=kernel_size, strides=strides,activation=activation ,data_format="channels_last", name=name)
#         tensor = batch_normalization(tensor)
        if name !=None : print("{}:{}".format(name,deconv2.shape))
        return deconv2
    
    def loss(self,pred):        
        learning_rate=0.0001        
#         loss = tf.metrics.mean_absolute_error(labels=self.target,predictions=pred)
        loss = tf.cast(tf.equal(pred,self.target),tf.float32)
        ops = tf.train.AdamOptimizer(learning_rate)
        train_op = ops.minimize(loss)     
        return loss, train_op

    def build_model(self):        
        tensor = self.inputs
        print("==========Encoder Start=============")    
        start = T.default_timer()
        conv0 = self.conv_net(tensor, filters=32) 
        conv1 = self.conv_net(conv0, filters=64,kernel_size=[4,4],strides=[2,2]) 
        conv2 = self.conv_net(conv1, filters=64) 
        conv3 = self.conv_net(conv2, filters=128,kernel_size=[4,4],strides=[2,2])
        conv4 = self.conv_net(conv3, filters=128) 
        conv5 = self.conv_net(conv4, filters=256,kernel_size=[4,4],strides=[2,2])
        conv6 = self.conv_net(conv5, filters=256)
        conv7 = self.conv_net(conv6, filters=512,kernel_size=[4,4],strides=[2,2])
        conv8 = self.conv_net(conv7, filters=512) 
        print("[Encoder build_time : {}ms]".format(T.default_timer() - start))  
        
        print("==========Decoder Start=============")      
        #512 256 256 128 128 64 64 32 3
        deconv8 = self.deconv_net(conv7,conv8,filters=512)
        deconv7 = self.conv_net(deconv8,256)
        deconv6 = self.deconv_net(conv6,deconv7,filters=256)
        deconv5 = self.conv_net(deconv6,128)
        deconv4 = self.deconv_net(conv4,deconv5,filters=128)
        deconv3 = self.conv_net(deconv4,64)
        deconv2 = self.deconv_net(conv2,deconv3,filters=64)
        deconv1 = self.conv_net(deconv2,32)
        
        deconv0 = self.crop_and_concat(conv0,deconv1)
        deconv0 = conv2d(deconv0,3,kernel_size=[3,3],strides=[1,1],activation=None)                        

        print("[Encoder build_time : {}ms]".format(T.default_timer() - start))
        
        pre = tf.cast(deconv0,tf.uint8)
        print("{}".format(pre.shape))        
        
        return pre , deconv0
    
    def __init__(self,sess,inputs,target):
        self.sess = sess
        self.inputs = inputs
        self.target = target

def main():
    inputs = tf.placeholder(shape = [None,512,512,3], dtype=tf.float32)
    target = tf.placeholder(shape = [None,414,414,3], dtype=tf.float32)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    unet = UNET(sess,inputs,target)
    (pred ,logits) = unet.build_model()
    loss = tf.metrics.mean_squared_error(labels=target,predictions=logits)
    tarin_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

if __name__ == '__main__':
    main()



# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=unet.target))
# loss = tf.metrics.mean_absolute_error(labels=target,predictions=logits)
# tarin_op = tf.train.AdamOptimizer(0.0001).minimize(loss)