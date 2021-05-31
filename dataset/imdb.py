import tensorflow_datasets as tfds
import tensorflow as tf


def get_datasets(validation_percent):
	train_percent = 100 - validation_percent

	if tf.__version__.startswith('1'):
		train_validation_split = tfds.Split.TRAIN.subsplit([train_percent, validation_percent])

		(train_data, validation_data), test_data = tfds.load(
			name="imdb_reviews",
			split=(train_validation_split, tfds.Split.TEST),
			as_supervised=True,
			download=True
		)
	else:
		train_data, validation_data, test_data = tfds.load(
			name="imdb_reviews",
			split=(f'train[{train_percent}%:]', f'train[:{validation_percent}%]', tfds.Split.TEST),
			as_supervised=True,
			download=True
		)

	return train_data, validation_data, test_data
