import reader
import tensorflow as tf
from models import bidirectional_rnn as birnn

def main(_):
    train_data, valid_data, vocabulary, word_to_id, label_to_id \
        = reader.ptb_raw_data("/home/dzhou/tfprojects/TextClassifier/data/20Newsgroups/")
    test_data = reader.ptb_raw_test_data("/home/dzhou/tfprojects/TextClassifier/data/20Newsgroups/")
    test_data_producer = reader.DataProducer(test_data)

#    saver = tf.train.import_meta_graph("/tmp/bidirectional_rnn.meta")
    model_config = birnn.ModelConfig(256, 1, 1e-3, 1000)
    input_config = birnn.InputConfig(vocabulary, len(label_to_id))
    graph = birnn.RNNClassification(model_config, input_config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/bidirectional_rnn")
        accuracy = 0
        for i in range(test_data_producer.size):
            data, label, seqlen = test_data_producer.next_batch(1)
            feed = {graph.x: data, graph.y: label, graph.seqlen: seqlen}
            accuracy += sess.run([graph.accuracy], feed_dict=feed)[0]
            #print(accuracy / (i + 1))
        print(accuracy)

if __name__ == "__main__":
    tf.app.run()