import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.examples.tutorials.mnist import input_data
slim = tf.contrib.slim
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, reshape=False)
num_features = 64 
train = False 

if not os.path.exists('/tmp/siam'):
    os.makedirs('/tmp/siam')
#run_id = '%02d' % len(os.walk('/tmp/siam').next()[1])
run_id = '13'
print 'Running : %s' % run_id
log_root = os.path.join('/tmp','siam', run_id)
ckpt_path = os.path.join(log_root, 'model.ckpt')

def contrastive_loss(dsq, d, label, m=0.5, pn_ratio=3.0):
    with tf.name_scope('contrastive_loss'):
        # assumes vectors of shape [batch_size, feature_length]
        label = tf.cast(label, tf.float32)
        pos = dsq
        neg = tf.square(tf.maximum(0.0, m - d)) # per-feature margin?

        # negative mining ...
        k = tf.cast(tf.reduce_sum(label)*pn_ratio, tf.int32)
        k = tf.maximum(k,1)
        k = tf.minimum(k,tf.shape(label)[0])
        n_v, n_i = tf.nn.top_k(neg,k)
        n_mask = tf.cast(neg > n_v[-1], tf.float32)

        loss = tf.reduce_mean(0.5 * (label*pos + (1.0 - label)*neg*n_mask))

    return loss

def reformat_data(x,y,diag=False):
    if diag:
        iu = np.triu_indices(len(y))
    else:
        iu = list(np.triu_indices(len(y)-1))
        iu[1] += 1
    return x[iu[0]], x[iu[1]], y[iu[0]], y[iu[1]]

def dwc(inputs, num_out, scope, stride=1, padding='SAME', activation_fn=tf.nn.elu):
    dc = slim.separable_conv2d(inputs,
            num_outputs=None,
            stride=stride,
            padding=padding,
            activation_fn=activation_fn,
            depth_multiplier=1,
            kernel_size=[3, 3],
            scope=scope+'/dc')
    pc = slim.conv2d(dc,
            num_out,
            kernel_size=[1, 1],
            activation_fn=activation_fn,
            scope=scope+'/pc')
    return pc

def net(input, is_training=True, reuse=None):
    with tf.variable_scope('Siamese', reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected],
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(4e-5),
                normalizer_fn = slim.batch_norm,
                normalizer_params={
                    'is_training' : is_training,
                    'decay' : 0.99,
                    'fused' : True,
                    'reuse' : reuse,
                    'scope' : 'BN'
                    }
                ):
            # input = (None,28,28,1)
            input -= 0.5
            input *= 2.0 # normalize to -1~1 

            logits = input
            logits = dwc(logits, num_features, scope='conv1')
            logits = dwc(logits, num_features*2, stride=2, scope='conv2')
            logits = dwc(logits, num_features*2, stride=2, scope='conv3')
            logits = dwc(logits, num_features*4, stride=2, scope='conv4')
            logits = slim.conv2d(logits, num_features, kernel_size=[1,1], stride=1, scope='conv5')
            logits = tf.space_to_depth(logits, block_size=4)
            logits = tf.squeeze(logits, [1,2])
            logits = tf.nn.l2_normalize(logits, dim=-1) # embedding
            cls = slim.fully_connected(logits, 10, scope='fc_cls', activation_fn=None)
    return logits, cls

def generate_metadata_file(n):
    d = os.path.join(log_root, 'projector')

    if not os.path.exists(d):
        os.makedirs(d)
    f = os.path.join(d, 'metadata.tsv')

    with open(f, 'w') as f:
        for i in range(n):
            c = mnist.test.labels[i]
            f.write('{}\n'.format(c))

def generate_embeddings(sess, input_tensor, embed_tensor, N):
    emb = []
    for i in range(10):
        s = i*N / 10
        e = (i+1)*N / 10
        emb.append(sess.run(embed_tensor, feed_dict={input_tensor : mnist.test.images[s:e]}))
    
    print emb[0].shape
    emb = np.concatenate(emb, axis=0)
    emb = tf.Variable(emb, name='embedding')

    print emb.shape

    sess.run(tf.variables_initializer([emb]))

    d = os.path.join(log_root, 'projector')

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(d, sess.graph)

    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = emb.name
    embed.metadata_path = os.path.join(d, 'metadata.tsv')
    embed.sprite.image_path = '/home/yoonyoungcho/Miscellaneous/test/contrastive_loss/MNIST_data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(writer, config)
    
    saver.save(sess, os.path.join(d, 'model.ckpt'), global_step=N)


def main():

    with tf.name_scope('inputs'):
        x1_t = tf.placeholder(tf.float32, (None,28,28,1))
        x2_t = tf.placeholder(tf.float32, (None,28,28,1))
        y1_t = tf.placeholder(tf.int32, (None))
        y2_t = tf.placeholder(tf.int32, (None))
        m_t = tf.equal(y1_t, y2_t)
        is_training = tf.placeholder_with_default(False, [])

    margin = 0.5
    logits_1, cls_1 = net(x1_t, is_training=is_training, reuse=None)
    logits_2, cls_2 = net(x2_t, is_training=is_training, reuse=True)

    with tf.name_scope('verification'):
        diff = logits_1 - logits_2
        dsq  = tf.reduce_sum(tf.square(diff), -1) # euclidean distance, squared
        d    = tf.sqrt(dsq) # euclidean distance
        v_loss = contrastive_loss(dsq, d, m_t, m=margin)
        v_pred = tf.less(d, margin, name='pred')

    with tf.name_scope('classification'):
        c_pred_1 = tf.cast(tf.argmax(cls_1, -1),tf.int32)
        c_pred_2 = tf.cast(tf.argmax(cls_2, -1),tf.int32)
        c_loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_1, labels=y1_t))
        c_loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_2, labels=y2_t))

    tf.losses.add_loss(v_loss)
    tf.losses.add_loss(c_loss_1)
    tf.losses.add_loss(c_loss_2)

    tf.summary.scalar('v_loss', v_loss)
    tf.summary.scalar('c_loss', c_loss_1 + c_loss_2)

    with tf.name_scope('accuracy'):
        v_acc = tf.reduce_mean(tf.cast(tf.equal(v_pred, m_t), tf.float32))
        c_acc_1 = tf.reduce_mean(tf.cast(tf.equal(c_pred_1, y1_t), tf.float32))
        c_acc_2 = tf.reduce_mean(tf.cast(tf.equal(c_pred_2, y2_t), tf.float32))

    tf.summary.scalar('v_acc', v_acc)
    tf.summary.scalar('c_acc', tf.reduce_mean([c_acc_1, c_acc_2]))
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    net_loss = tf.losses.get_total_loss()
    with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(1e-3).minimize(net_loss)

    merged = tf.summary.merge_all()
    
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if train:
            t_writer = tf.summary.FileWriter(os.path.join(log_root, 'train'), sess.graph)
            v_writer = tf.summary.FileWriter(os.path.join(log_root, 'valid'), sess.graph)

            for i in range(12000):
                xs, ys = mnist.train.next_batch(15)
                x1, x2, y1, y2 = reformat_data(xs, ys)
                _,s = sess.run([opt,merged], feed_dict={x1_t:x1, x2_t:x2, y1_t:y1, y2_t:y2, is_training : True})
                t_writer.add_summary(s, i)
                if i % 10 == 0: # validation
                    xs,ys = mnist.validation.next_batch(15)
                    x1, x2, y1, y2 = reformat_data(xs, ys)
                    l,a,s = sess.run([net_loss,v_acc,merged], feed_dict={x1_t:x1, x2_t:x2, y1_t:y1, y2_t:y2, is_training : False})
                    print '%d) Loss : %.2E; Acc : %.2f' % (i,l,a)
                    v_writer.add_summary(s, i)
                    saver.save(sess, ckpt_path)
        else:
            saver.restore(sess, ckpt_path) # -- only restores mobilenet weights

        #cnt = 0
        #m_cnt = 0
        #for i in range(1000):
        #    xs, ys = mnist.test.next_batch(2)
        #    x1, x2, y1, y2 = reformat_data(xs, ys)
        #    p = sess.run(v_pred, feed_dict={x1_t:x1, x2_t:x2, y1_t:y1, y2_t:y2, is_training : False})
        #    x1,x2,y1,y2,p = [np.squeeze(e) for e in [x1,x2,y1,y2,p]]
        #    m = (y1 == y2)
        #    if m:
        #        m_cnt += 1
        #        print m_cnt
        #    cv2.imshow('x1', x1)
        #    cv2.imshow('x2', x2)
        #    if(p == m):
        #        continue
        #    cnt += 1
        #    print 'wrong :', cnt
        #    print 'pred : %d; label : %d' % (p, m)
        #    if cv2.waitKey(0) == 27:
        #        break

        generate_metadata_file(10000)
        generate_embeddings(sess, x1_t, logits_1, 10000)


    #for x1,x2,y in zip(b_x1,b_x2,b_y):
    #    cv2.imshow('x1', x1)
    #    cv2.imshow('x2', x2)
    #    if cv2.waitKey(0) == 27:
    #        break

if __name__ == "__main__":
    main()


