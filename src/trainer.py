import reader
import tensorflow as tf
from models import bidirectional_rnn as birnn

def main(_):
    # train_data, valid_data, vocabulary, word_to_id, label_to_id = reader.ptb_raw_data("DSL-Task-master/data/DSLCC-v2.0/train-dev/")
    train_data, valid_data, vocabulary, word_to_id, label_to_id \
        = reader.ptb_raw_data("/home/dzhou/tfprojects/TextClassifier/data/20Newsgroups/")
    train_data_producer = reader.DataProducer(train_data)
    valid_data_producer = reader.DataProducer(valid_data)

    print("vocabulary: " + str(vocabulary))
    model_config = birnn.ModelConfig(256, 100, 1e-3, vocabulary, len(label_to_id))
    graph = birnn.RNNClassification(model_config, True)

    graph.train(train_data_producer, valid_data_producer, 1000)


if __name__ == "__main__":
    tf.app.run()