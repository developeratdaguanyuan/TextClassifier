import tensorflow as tf

class InputConfig(object):
  def __init__(self, _vocab_size, _class_size):
    self.vocab_size = _vocab_size
    self.class_size = _class_size

class ModelConfig(object):
  hidden_size = 256
  batch_size = 100
  learning_rate = 1e-3
  num_epoch = 1000

class RNNClassification(object):
  def __init__(self, model_config, input_config):
    self.num_epoch = model_config.num_epoch
    self.batch_size = model_config.batch_size

    size = model_config.hidden_size
    learning_rate = model_config.learning_rate
    vocab_size = input_config.vocab_size
    num_classes = input_config.class_size

    # Placeholders
    self.x = tf.placeholder(tf.int32, [self.batch_size, None]) # [batch_size, ?]
    self.seqlen = tf.placeholder(tf.int32, [self.batch_size])
    self.y = tf.placeholder(tf.int32, [self.batch_size])

    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x) # [batch_size, ?, size]
    # RNN
    init_state_fw = tf.get_variable('init_state_fw', [1, size],
                                    initializer=tf.constant_initializer(0.0))
    init_state_fw = tf.tile(init_state_fw, [self.batch_size, 1])
    init_state_bw = tf.get_variable('init_state_bw', [1, size],
                                    initializer=tf.constant_initializer(0.0))
    init_state_bw = tf.tile(init_state_bw, [self.batch_size, 1])

    cell_fw = tf.contrib.rnn.GRUCell(size)
    cell_bw = tf.contrib.rnn.GRUCell(size)
    #(rnn_outputs_fw, rnn_outputs_bw), final_state \
    rnn_outputs, final_state \
      = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, self.seqlen, init_state_fw, init_state_bw)
    rnn_outputs_fw, rnn_outputs_bw = rnn_outputs
    last_output = tf.transpose(rnn_outputs_fw, perm=[1, 0, 2])[-1]
    
    # Softmax layer
    with tf.variable_scope('softmax'):
      W = tf.get_variable('W', [size, num_classes])
      b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits_t = tf.matmul(last_output, W) + b
    preds = tf.nn.softmax(logits_t)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), self.y)
    self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_t, labels = self.y))
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

  def train(self, train_data_producer, valid_data_producer):
    self.saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      max_valid_accuracy, train_accuracy = 0, 0
      num_iteration = self.num_epoch * train_data_producer.size / self.batch_size
      for i in range(num_iteration):
        curr_data, curr_label, curr_seqlen = train_data_producer.next_batch(self.batch_size)
        feed = {self.x: curr_data, self.y: curr_label, self.seqlen: curr_seqlen}
        train_accuracy += sess.run([self.accuracy, self.optimizer, self.loss], feed_dict=feed)[0]

        if i > 0 and i % (train_data_producer.size / self.batch_size) == 0:
          valid_accuracy = 0
          valid_iteration = valid_data_producer.size/self.batch_size
          for j in range(valid_iteration):
            valid_data, valid_label, valid_seqlen = valid_data_producer.next_batch(self.batch_size)
            valid_feed = {self.x: valid_data, self.y: valid_label, self.seqlen: valid_seqlen}
            valid_accuracy += sess.run([self.accuracy], feed_dict=valid_feed)[0]

          valid_accuracy /= valid_iteration
          train_accuracy = train_accuracy / train_data_producer.size * self.batch_size
          print("[train accuracy] %5.3f [valid accuracy] %5.3f" % (train_accuracy, valid_accuracy))
          train_accuracy = 0
          if valid_accuracy > max_valid_accuracy:
            max_valid_accuracy = valid_accuracy
            save_path = self.saver.save(sess, "/tmp/bidirectional_rnn")
            print("Model saved in file: %s" % save_path)
