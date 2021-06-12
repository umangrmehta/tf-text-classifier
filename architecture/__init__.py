import tensorflow as tf


def get_output_layer(hidden_layer, classes):
	if classes > 2:
		pred = tf.keras.layers.Dense(classes, activation='sigmoid')(hidden_layer)
		loss = "categorical_crossentropy"
	else:
		pred = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)
		loss = "binary_crossentropy"

	return pred, loss
