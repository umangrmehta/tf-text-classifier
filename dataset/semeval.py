import tensorflow as tf
from dataset.preprocessors import preprocess_tweet

features = {
	'text': tf.io.FixedLenFeature([], tf.string, default_value=''),
	'polarity': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}
classes = 2


def parse_record(example_proto):
	example_dict = tf.io.parse_single_example(example_proto, features=features)
	return example_dict["text"], example_dict["polarity"]


def load_data(path):
	data = tf.data.TFRecordDataset([path])
	return data.map(parse_record)


def get_datasets(validation_percent=None):
	train_data = load_data("data/SemEval2017/train.tfrecords")
	validation_data = load_data("data/SemEval2017/validation.tfrecords")
	test_data = load_data("data/SemEval2017/test.tfrecords")

	return train_data, validation_data, test_data


def get_preprocessed_input_layer(input_layer):
	return tf.keras.layers.Lambda(preprocess_tweet, output_shape=(1, ), name='tweet_preprocessor')(input_layer)
