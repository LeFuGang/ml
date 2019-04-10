import tensorflow as tf



class TextLSTM(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.lstm()

    def lstm(self):
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
            cells = [lstm_cell for _ in range(self.config.num_layers)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

            # dynamic返回的是两个参数：outputs,last_states，其中outputs是[2,20,128]，也就是每一个迭代隐状态的输出,last_states是由(c,h)组成的tuple，均为[batch,128]。
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]

        with tf.name_scope("score"):
            fc1 = tf.layers.dense(last, self.config.hidden_dim, name="fc1")
            fc = tf.layers.dropout(fc1, self.keep_prob)     # dropout只能用在训练中，测试的时候不能dropout，要用完整的网络测试哦。
            fc = tf.nn.relu(fc)

            self.logitis = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logitis), 1) # 1 表示列，返回下标

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logitis, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuarcy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    # 类型转换

