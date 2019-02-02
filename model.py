import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import SEQLEN, BATCHSIZE, VECLEN

class Model():
    def __init__(self, Xin_):
        Xin  = tf.one_hot(Xin_, VECLEN, dtype=tf.float32)

        self.global_step = tf.train.create_global_step()

        h_height = 5
        h_units = 100

        h_cells = [rnn.GRUCell(h_units, name='h_cell_{}'.format(i)) for i in range(h_height)]

        self.h = rnn.MultiRNNCell(h_cells, state_is_tuple=False)
        self.h_Hin = tf.placeholder_with_default(self.h.zero_state(dtype=tf.float32, batch_size=tf.shape(Xin)[0]), shape=(None, self.h.state_size,))

        h_Yout, self.h_Hout = tf.nn.dynamic_rnn(self.h, Xin, initial_state=self.h_Hin)

        _h_Yout = h_Yout[:,-1,:]

        self.r = tf.layers.dense(_h_Yout, 20, name='r', activation=tf.nn.sigmoid)

        self.Rin = tf.tile(self.r[:,None,:], (1, tf.shape(Xin)[-2], 1))

        tf.summary.histogram('Rin', self.Rin)

        f_height = 4
        f_units = 200

        f_cells = [rnn.GRUCell(f_units, name='f_cell_{}'.format(i)) for i in range(f_height)]

        self.f = rnn.MultiRNNCell(f_cells, state_is_tuple=False)
        self.f_Hin = tf.placeholder_with_default(self.f.zero_state(dtype=tf.float32, batch_size=tf.shape(Xin)[0]), shape=(None, self.f.state_size,))


        f_Yout, self.f_Hout = tf.nn.dynamic_rnn(self.f, self.Rin, initial_state=self.f_Hin)

        self.logits = tf.layers.dense(f_Yout, VECLEN, name='FinalDense')

        labels = tf.cast(Xin_, tf.int32)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits) 

        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(
                tf.argmax(self.logits, axis=-1, output_type=tf.int32),
                labels),
            tf.float32))

        optim = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )

        self.step = optim.minimize(self.loss, global_step=self.global_step)
