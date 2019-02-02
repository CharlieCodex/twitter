import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import SEQLEN, BATCHSIZE, VECLEN

def self_feeding_rnn(cell, seqlen, Hin, Xin, processing=tf.identity):
    '''Unroll cell by feeding output (hidden_state) of cell back into in as input.
       Outputs are passed through `processing`. It is up to the caller to ensure that the processed
       outputs have suitable shape to be input.'''
    veclen = tf.shape(Xin)[-1]
    # this will grow from [ BATCHSIZE, 0, VELCEN ] to [ BATCHSIZE, SEQLEN, VECLEN ]
    buffer = tf.TensorArray(dtype=tf.float32, size=seqlen)
    initial_state = (0, Hin, Xin, buffer)
    condition = lambda i, *_: i < seqlen
    print(initial_state)
    def do_time_step(i, state, xo, ta):
        Yt, Ht = cell(xo, state)
        Yro = processing(Yt)
        return (1+i, Ht, Yro, ta.write(i, Yro))
    
    _, Hout, _, final_ta = tf.while_loop(condition, do_time_step, initial_state)

    ta_stack = final_ta.stack()
    Yo = tf.reshape(ta_stack,shape=((-1, seqlen, veclen)))
    return Yo, Hout


class Model():
    def __init__(self, Xin_):
        Xin  = tf.one_hot(Xin_, VECLEN, dtype=tf.float32)

        self.batchsize = tf.placeholder_with_default(tf.shape(Xin_)[0], shape=(()))
        self.seqlen = tf.placeholder_with_default(tf.shape(Xin_)[1], shape=(()))

        self.global_step = tf.train.create_global_step()

        h_height = 5
        h_units = 100

        h_cells = [rnn.GRUCell(h_units, name='h_cell_{}'.format(i)) for i in range(h_height)]

        self.h = rnn.MultiRNNCell(h_cells, state_is_tuple=False)
        self.h_Hin = tf.placeholder_with_default(self.h.zero_state(dtype=tf.float32, batch_size=self.batchsize), shape=(None, self.h.state_size,))

        h_Yout, self.h_Hout = tf.nn.dynamic_rnn(self.h, Xin, initial_state=self.h_Hin)

        _h_Yout = h_Yout[:,-1,:]

        self.r = tf.layers.dense(_h_Yout, 20, name='r', activation=tf.nn.sigmoid)

        f_height = 4
        f_units = 200

        f_cells = [rnn.GRUCell(f_units, name='f_cell_{}'.format(i)) for i in range(f_height)]

        self.f = rnn.MultiRNNCell(f_cells, state_is_tuple=False)
        self.f_Hin = tf.layers.dense(self.r, self.f.state_size)

        # f_Xin: [BATCH X VEC]

        self.f_Xin = tf.placeholder_with_default(
            tf.zeros(
                dtype=tf.float32,
                shape=tf.TensorShape((1, VECLEN))),
            shape=tf.TensorShape((None, VECLEN)))

        self.f_Xin_ = tf.tile(self.f_Xin, (self.batchsize,1))

        processing = tf.layers.Dense(VECLEN)

        self.f_Yout, self.f_Hout = self_feeding_rnn(self.f, self.seqlen, self.f_Hin, self.f_Xin_, processing=processing)

        labels = tf.cast(Xin_, tf.int32)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.f_Yout) 

        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(
                tf.argmax(self.f_Yout, axis=-1, output_type=tf.int32),
                labels),
            tf.float32))

        optim = tf.train.AdamOptimizer(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-3
        )

        self.step = optim.minimize(self.loss, global_step=self.global_step)

if __name__ == '__main__':
    Xin = tf.placeholder(dtype=tf.uint8, shape=(BATCHSIZE, SEQLEN))
    model = Model(Xin)