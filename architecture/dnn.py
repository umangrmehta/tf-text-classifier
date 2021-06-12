import tensorflow as tf


def get_model(input_layer, embedding_layer, model_name, classes=2):
	embedding_dropout = tf.keras.layers.Dropout(0.1)(embedding_layer)
	dense = tf.keras.layers.Dense(64, activation='relu')(embedding_dropout)
	if classes > 2:
		pred = tf.keras.layers.Dense(classes, activation='sigmoid')(dense)
		loss = "categorical_crossentropy"
	else:
		pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
		loss = "binary_crossentropy"
	model = tf.keras.Model(inputs=[input_layer], outputs=pred, name=model_name)
	model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
	return model
