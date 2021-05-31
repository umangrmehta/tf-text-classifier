import tensorflow_datasets as tfds


def get_datasets(validation_percent):
	train_percent = 100 - validation_percent

	train_data, validation_data, test_data = tfds.load(
		name="sentiment140",
		split=(f'train[{train_percent}%:]', f'train[:{validation_percent}%]', tfds.Split.TEST),
		as_supervised=True,
		download=True
	)

	return train_data, validation_data, test_data
