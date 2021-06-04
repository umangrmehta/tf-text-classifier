import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, choices=["IMDB", "SemEval", "Sentiment140"], default="Sentiment140", help="Dataset to use for Model")
parser.add_argument("-e", "--embedding", type=str, choices=["ELMo", "BERT"], default="BERT", help="Embeddings to use for Model")
args = parser.parse_args()

if args.dataset == "IMDB":
	import dataset.imdb as ds
else:
	if not tf.__version__.startswith('2'):
		raise Exception("Sentiment140 Dataset works on TensorFlow 2.x only. Please install Tensorflow 1.15(preferably) to use ELMo")
	import dataset.sentiment140 as ds

if args.embedding == "ELMo":
	if not tf.__version__.startswith('1'):
		raise Exception("ELMo Embeddings works on TensorFlow 1.x only. Please install Tensorflow 1.15(preferably) to use ELMo")
	import embeddings.elmo as emb
else:
	import embeddings.bert as emb

# Load Dataset
BATCH_SIZE = 256
train_data, validation_data, test_data = ds.get_datasets(20)
train_data = train_data.shuffle(5000).batch(BATCH_SIZE)
validation_data = validation_data.batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# Build Model
input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentences')
emb.pooled = True
embedding = emb.embedding_layer(input_text)
embedding_dropout = tf.keras.layers.Dropout(0.1)(embedding)
dense = tf.keras.layers.Dense(256, activation='relu')(embedding_dropout)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
model = tf.keras.Model(inputs=[input_text], outputs=pred, name=f'{args.embedding}-Dense')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if args.embedding == "ELMo":
	# Train Model
	with tf.Session() as session:
		tf.keras.backend.set_session(session)
		session.run(tf.global_variables_initializer())
		session.run(tf.tables_initializer())
		history = model.fit(train_data, validation_data=validation_data, epochs=100)

	# Evaluate Model
	evaluation = model.evaluate(test_data)
else:
	# Train Model
	history = model.fit(train_data, validation_data=validation_data, epochs=100)

	# Evaluate Model
	evaluation = model.evaluate(test_data, return_dict=True)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()

plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()

print("Test Results:")
print(evaluation)
