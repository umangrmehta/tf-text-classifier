import tensorflow as tf
import tensorflow_hub as hub

# Setting outside to let embedding() to be used in Lambda Layer
trainable = False
pooled = False


def embeddings(x):
	elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=trainable)
	if pooled:
		return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
	return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]


def embedding_layer(input_layer):
	return tf.keras.layers.Lambda(embeddings, output_shape=(1024, ))(input_layer)
