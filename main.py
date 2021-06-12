import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, choices=["IMDB", "SemEval", "Sentiment140"], default="Sentiment140", help="Dataset to use for Model")
parser.add_argument("-e", "--embedding", type=str, choices=["ELMo", "BERT"], default="BERT", help="Embeddings to use for Model")
parser.add_argument("-t", "--trainable", action="store_true", help="Are Embeddings Trainable?")
parser.add_argument("-a", "--architecture", type=str, choices=["DNN", "LSTM", "CNN", "LSTM-CNN", "CNN-LSTM"], default="DNN", help="Model Architecture")
args = parser.parse_args()

if args.dataset == "IMDB":
	import dataset.imdb as ds
elif args.dataset == "SemEval":
	import dataset.semeval as ds
else:
	if not tf.__version__.startswith('2'):
		raise Exception("Sentiment140 Dataset works on TensorFlow 2.x only. Please install latest TensorFlow 2.x to use the dataset")
	import dataset.sentiment140 as ds

if args.embedding == "ELMo":
	if not tf.__version__.startswith('1'):
		raise Exception("ELMo Embeddings works on TensorFlow 1.x only. Please install TensorFlow 1.15(preferably) to use ELMo")
	import embeddings.elmo as emb
else:
	import embeddings.bert as emb

if args.trainable:
	emb.trainable = True

if args.architecture == "LSTM":
	import architecture.lstm as arc
else:
	import architecture.dnn as arc
	emb.pooled = True

# Load Dataset
BATCH_SIZE = 256
train_data, validation_data, test_data = ds.get_datasets(20)
train_data = train_data.shuffle(5000).batch(BATCH_SIZE)
validation_data = validation_data.batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# Build Model
input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentences')
input_layer = ds.get_preprocessed_input_layer(input_text)
embedding = emb.embedding_layer(input_layer)
model = arc.get_model(input_text, embedding, f'{args.embedding}-{args.architecture}')

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
	history = model.fit(train_data, validation_data=validation_data, epochs=1)
	json.dump(history.history, open(f'results/{args.embedding}-{args.architecture}-{args.dataset}-training.json', 'w'))
	model.save(f'models/{args.embedding}-{args.architecture}-{args.dataset}')

	# Evaluate Model
	evaluation = model.evaluate(test_data, return_dict=True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
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
