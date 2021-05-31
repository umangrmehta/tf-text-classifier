import embeddings
import tensorflow_hub as hub
import tensorflow_text as text

bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'
tfhub_handle_encoder = embeddings.map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = embeddings.map_model_to_preprocess[bert_model_name]
trainable = False
pooled = False


def embedding_layer(input_layer):
	preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
	encoder_inputs = preprocessing_layer(input_layer)
	encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=trainable, name='BERT_encoder')
	outputs = encoder(encoder_inputs)
	if pooled:
		return outputs['pooled_output']
	return outputs['sequence_output']
