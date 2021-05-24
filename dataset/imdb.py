import tensorflow_datasets as tfds


def get_datasets(validation_percent):
	train_percent = 100 - validation_percent

	train_validation_split = tfds.Split.TRAIN.subsplit([train_percent / 10, validation_percent / 10])

	(train_data, validation_data), test_data = tfds.load(
		name="imdb_reviews",
		split=(train_validation_split, tfds.Split.TEST),
		as_supervised=True,
		download=True
	)

	return train_data, validation_data, test_data
