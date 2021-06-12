import tensorflow_datasets as tfds
import tensorflow as tf
from dataset.preprocessors import preprocess_tweet

classes = 2


def get_datasets(validation_percent):
	train_percent = 100 - validation_percent

	train_data, validation_data, test_data = tfds.load(
		name="sentiment140",
		split=(f'train[{train_percent}%:]', f'train[:{validation_percent}%]', tfds.Split.TEST),
		as_supervised=True,
		download=True
	)

	return train_data, validation_data, test_data


def get_preprocessed_input_layer(input_layer):
	return tf.keras.layers.Lambda(preprocess_tweet, output_shape=(1, ), name='tweet_preprocessor')(input_layer)
