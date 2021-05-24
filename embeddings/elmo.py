import tensorflow as tf
import tensorflow_hub as hub

# Setting outside to let embedding() to be used in Lambda Layer
trainable = False
pooled = False


def embeddings(x):
	elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=trainable)
	if pooled:
		return elmo(tf.cast(x, tf.string), signature="default", as_dict=True)["default"]
	else:
		return elmo(tf.cast(x, tf.string), signature="default", as_dict=True)["elmo"]
