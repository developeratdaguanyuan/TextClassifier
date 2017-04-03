import reader
import tensorflow as tf
from models import bidirectional_rnn as birnn

def main(_):
    test_data = reader.ptb_raw_test_data("/home/dzhou/tfprojects/TextClassifier/data/20Newsgroups/")
    test_data_producer = reader.DataProducer(test_data)

    model_config = birnn.ModelConfig(256, 1, 1e-3, 8328, 4)
    graph = birnn.RNNClassification(model_config, False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/bidirectional_rnn/bidirectional_rnn_1998")
        accuracy = 0
        for i in range(test_data_producer.size):
            data, label, seqlen = test_data_producer.next_batch(1)
            feed = {graph.x: data, graph.y: label, graph.seqlen: seqlen}
            accuracy += sess.run([graph.accuracy], feed_dict=feed)[0]
        print(accuracy)

if __name__ == "__main__":
    tf.app.run()