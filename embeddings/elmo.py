import tensorflow as tf
import tensorflow_hub as hub


def embeddings(x, trainable: bool = False, pooled: bool = False):
	elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=trainable)
	if pooled:
		return elmo(tf.cast(x, tf.string), signature="default", as_dict=True)["default"]
	else:
		return elmo(tf.cast(x, tf.string), signature="default", as_dict=True)["elmo"]
