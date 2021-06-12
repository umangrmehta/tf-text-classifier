import tensorflow as tf
import architecture


def get_model(input_layer, embedding_layer, model_name, classes=2):
	embedding_dropout = tf.keras.layers.Dropout(0.1)(embedding_layer)
	dense = tf.keras.layers.Dense(64, activation='relu')(embedding_dropout)
	pred, loss = architecture.get_output_layer(dense, classes)
	model = tf.keras.Model(inputs=[input_layer], outputs=pred, name=model_name)
	model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
	return model
