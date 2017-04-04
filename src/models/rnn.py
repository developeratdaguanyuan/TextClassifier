import tensorflow as tf
import reader


class ModelConfig(object):
  def __init__(self, hidden_size, batch_size, learning_rate, vocab_size, class_size):
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.vocab_size = vocab_size
    self.class_size = class_size
    #self.keep_prob = keep_prob

class RNNClassifier(object):
  def __init__(self, model_config, is_training):
    self.batch_size = model_config.batch_size

    size = model_config.hidden_size
    learning_rate = model_config.learning_rate
    vocab_size = model_config.vocab_size
    num_classes = model_config.class_size

    # Placeholders
    self.x = tf.placeholder(tf.int32, [self.batch_size, None]) # [batch_size, ?]
    self.seqlen = tf.placeholder(tf.int32, [self.batch_size])
    self.y = tf.placeholder(tf.int32, [self.batch_size])
    #keep_prob = tf.constant(model_config.keep_prob)

    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x) # [batch_size, ?, size]
    # rnn_inputs = tf.nn.dropout(rnn_inputs, keep_prob)
    # RNN
    cell = tf.contrib.rnn.GRUCell(size)
    init_state = tf.get_variable('init_state', [1, size],
                                 initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [self.batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=self.seqlen,
                                                 initial_state=init_state) # [batch_size, size]
    # rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    idx = tf.range(self.batch_size) * tf.shape(rnn_outputs)[1] + (self.seqlen - 1) # very important
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, size]), idx) # [batch_size, size]

    # Softmax layer
    with tf.variable_scope('softmax'):
      W = tf.get_variable('W', [size, num_classes])
      b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits_t = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits_t)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), self.y)
    self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_t, labels = self.y))
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

  def train(self, train_data_producer, valid_data_producer, num_epoch):
    self.saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      max_valid_accuracy, train_accuracy = 0, 0
      num_iteration = num_epoch * train_data_producer.size / self.batch_size
      for i in range(num_iteration):
        curr_data, curr_label, curr_seqlen = train_data_producer.next_batch(self.batch_size)
        feed = {self.x: curr_data, self.y: curr_label, self.seqlen: curr_seqlen}
        train_accuracy += sess.run([self.accuracy, self.optimizer, self.loss], feed_dict=feed)[0]

        if i > 0 and i % (train_data_producer.size / self.batch_size) == 0:
          valid_accuracy = 0
          valid_iteration = valid_data_producer.size / self.batch_size
          for j in range(valid_iteration):
            valid_data, valid_label, valid_seqlen = valid_data_producer.next_batch(self.batch_size)
            valid_feed = {self.x: valid_data, self.y: valid_label, self.seqlen: valid_seqlen}
            valid_accuracy += sess.run([self.accuracy], feed_dict=valid_feed)[0]

          valid_accuracy /= valid_iteration
          train_accuracy = train_accuracy / train_data_producer.size * self.batch_size
          print("[train accuracy] %5.3f [valid accuracy] %5.3f" % (train_accuracy, valid_accuracy))
          train_accuracy = 0
          if valid_accuracy >= max_valid_accuracy:
            max_valid_accuracy = valid_accuracy
            save_path = self.saver.save(sess, "/tmp/rnn/rnn_" + str(i))
            print("Model saved in file: %s" % save_path)
