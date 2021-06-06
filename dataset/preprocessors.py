import tensorflow as tf


def preprocess_tweet(x):
	x = tf.strings.regex_replace(x, '(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', name="remove_urls")
	x = tf.strings.regex_replace(x, '@\s*.+', 'tweeter', name="replace_mentions")
	x = tf.strings.regex_replace(x, '\s#', ' ', name="replace_hashtags")
	return tf.squeeze(tf.cast(x, tf.string))
