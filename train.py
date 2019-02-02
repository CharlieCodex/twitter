import tensorflow as tf
import numpy as np
import os
from dataset import dataset, x_data, all_data
from model_rethink import Model
from tqdm import tqdm
from constants import BATCHSIZE, SEQLEN, VECLEN, DECODER
from time import time

timestamp = int(time())

it = dataset.make_initializable_iterator()

next_batch = it.get_next('Xin')

model = Model(next_batch[0])

if os.path.exists('/Volumes/Space/'):
    tesorboard_path = '/Volumes/Space/twitter-rethink/tensorboard/{}'.format(timestamp)
else:
    tesorboard_path = 'tensorboard/{}'.format(timestamp)

fw = tf.summary.FileWriter(tesorboard_path)

for variable in tf.global_variables():
    tf.summary.histogram(variable.name, variable)

histo_sum_op = tf.summary.merge_all()

scalar_sum_op = tf.summary.merge((
    tf.summary.scalar('Loss', tf.reduce_mean(model.loss)),
    tf.summary.scalar('Accuracy', model.accuracy),)
)

save_dir = 'saves-rethink-quick-lr-no-bias/'

epochs = 10
iterations = ( 1e6 // SEQLEN // BATCHSIZE) * BATCHSIZE

first = True
# first_offset = 3 * all_data.shape[0] // 4
first_offset = 0

with tf.Session() as sess:
    fw.add_graph(sess.graph)
    saver = tf.train.Saver()
    latest = tf.train.latest_checkpoint(save_dir)
    sess.run(tf.global_variables_initializer())
    if latest:
        saver.restore(sess, latest)
    
    first_offset = (sess.run(model.global_step) * BATCHSIZE) % all_data.shape[0]
    for epoch in range(epochs):
        print('Start of epoch {}'.format(epoch))
        try:
            epoch_data = all_data
            if first:
                epoch_data = epoch_data[first_offset:]
                first = False
            sess.run(it.initializer, feed_dict={
                x_data: epoch_data
            })
            print('Epoch data shape', epoch_data.shape)
            while True:
                # Keep running next_batch till the Dataset is exhausted
                with tqdm(total=iterations, unit='samples') as pbar:
                    step = 0
                    while True:
                        acc, _, gstep, s = sess.run([model.accuracy, model.step, model.global_step, scalar_sum_op])
                        pbar.update(BATCHSIZE)
                        step += 1
                        fw.add_summary(s, global_step=gstep)
                        if step == pbar.total//BATCHSIZE//2:
                            r = np.random.random(size=(1,20,))
                            sample_len = 255
                            Rin = np.repeat(r, sample_len, axis=-2)
                            lg = sess.run(model.f_Yout, feed_dict={
                                model.r: r,
                                model.batchsize: 1,
                                model.seqlen: sample_len
                            })
                            Ryo = np.argmax(lg[0,:,:], axis=-1)
                            buff = DECODER[Ryo].reshape((-1,))
                            text = ''.join([chr(x) for x in buff])
                            with open('sample_{}_{}.txt'.format(timestamp, gstep), 'w+') as f:
                                f.write(text)
                                f.write(repr(Ryo))

                        if step == pbar.total//BATCHSIZE:
                            print('Saving...')
                            sp = saver.save(sess, os.path.join(save_dir, 'checkpoint'), global_step=gstep)
                            print('\tSaved to {}'.format(sp))
                            break

        except tf.errors.OutOfRangeError:
            print('Epoch complete, data exhausted')
            pass