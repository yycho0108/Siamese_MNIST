import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

batch_size = 32
input_size = 10
feat_size = 16
margin = 0.5

def contrastive_loss(diff,label,m=0.5):
    # assumes vectors of shape [batch_size, feature_length]
    label = tf.cast(label, tf.float32)
    d = tf.square(tf.reduce_sum(diff, -1))
    d_sqrt = tf.sqrt(d)
    pd = d
    nd = tf.square(tf.maximum(0.0, m - d_sqrt)) # per-feature margin?

    return 0.5 * (label*pd + (1.0 - label)*nd)

def logistic_loss():
    pass

def cosine_loss(a,b,label):
    an = tf.norm(a, axis=-1, keep_dims=True)
    bn = tf.norm(b, axis=-1, keep_dims=True)
    d = (a * b) / (an*bn)

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

l = tf.placeholder(tf.float32, [batch_size])
a = tf.placeholder(tf.float32, [batch_size, input_size])
b = tf.placeholder(tf.float32, [batch_size, input_size])
W = slim.variable('W', shape=[input_size, feat_size])

am = tf.matmul(a,W)
bm = tf.matmul(b,W)
d = tf.abs(am - bm)

loss = contrastive_loss(d,l,m=margin)
opt = tf.train.AdamOptimizer(1e-2).minimize(loss)

#neg_loss = tf.exp(tf.multiply(1.0 - f_label_t, tf.square(diff)))
#mx = tf.square(tf.maximum(margin - diff, 0))
#pos_loss = tf.multiply(f_label_t, mx)
#loss = 1.0 * (neg_loss + pos_loss)

def gen_pair(bs=2, n=10, noise=0.01):
    a = np.random.random((bs,n))
    b = np.random.random((bs,n))

    label = np.random.choice([0.0,1.0], (batch_size,1))

    b = (label*a) + (1.0-label)*(b)

    noise = np.random.normal(size=(bs,n), scale=noise)
    return a, b + noise, np.squeeze(label, 1)


with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        a_in, b_in, l_in = gen_pair(bs=batch_size, n=input_size)
        loss_val, _ = sess.run([loss,opt], feed_dict={l : l_in, a : a_in, b : b_in})
        loss_val *= 1000
        if i % 10 == 0:
            print 'labels', l_in 
            print 'predictions', (loss_val > 1e-1) * 1
        #print 'labels', l_in

    #a_in, b_in = gen_pair(same=True)
    #print sess.run(loss, feed_dict={l : [DIFF], a : a_in, b : b_in}), 'same-wrong'

    #a_in, b_in = gen_pair(same=False)
    #print sess.run(loss, feed_dict={l : [DIFF], a : a_in, b : b_in}), 'diff-right'

    #a_in, b_in = gen_pair(same=False)
    #print sess.run(loss, feed_dict={l : [SAME], a : a_in, b : b_in}), 'diff-wrong'

    #print sess.run(loss, feed_dict={l : [DIFF], a : [1.0], b : [0.0]}), 'wrong'
    #print sess.run(loss, feed_dict={l : [DIFF], a : [1.0], b : [1.0]}), 'match'
