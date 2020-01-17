from flask import Flask, request
from flask_cors import CORS
import torch
import numpy as np
from pytorch_transformers import BertTokenizer, BertForMaskedLM
import nltk

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()

@app.route('/fillblanks', methods=['POST'])
def predict():
	sentence_orig = request.form.get('text')
	if '____' not in sentence_orig:
		return sentence_orig

	sentence = sentence_orig.replace('____', 'MASK')
	tokens = nltk.word_tokenize(sentence)
	sentences = nltk.sent_tokenize(sentence)
	sentence = " [SEP] ".join(sentences)
	sentence = "[CLS] " + sentence + " [SEP]"
	tokenized_text = tokenizer.tokenize(sentence)
	masked_index = tokenized_text.index('mask')
	tokenized_text[masked_index] = "[MASK]"
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

	segments_ids = []
	sentences = sentence.split('[SEP]')
	for i in range(len(sentences)-1):
		segments_ids.extend([i]*len(sentences[i].strip().split()))
		segments_ids.extend([i])

	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	with torch.no_grad():
	    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
	    predictions = outputs[0]
	    attention = outputs[-1]

	dim = attention[2][0].shape[-1]*attention[2][0].shape[-1]
	a = attention[2][0].reshape(12, dim)
	b = a.mean(axis=0)
	c = b.reshape(attention[2][0].shape[-1],attention[2][0].shape[-1])
	avg_wgts = c[masked_index]
	#print (avg_wgts, tokenized_text)
	focus = [tokenized_text[i] for i in avg_wgts.argsort().tolist()[::-1] if tokenized_text[i] not in ['[SEP]', '[CLS]', '[MASK]']][:5]

	# for layer in range(12):
	# 	weights_layer = np.array(attention[0][0][layer][masked_index])
	# 	print (weights_layer, tokenized_text)
	# 	print (weights_layer.argsort()[-3:][::-1])
	# 	print ()
	predicted_index = torch.argmax(predictions[0, masked_index]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
	for f in focus:
		sentence_orig = sentence_orig.replace(f, '<font color="blue">'+f+'</font>')
	return sentence_orig.replace('____', '<font color="red"><b><i>'+predicted_token+'</i></b></font>')

if __name__=='__main__':
	app.run(debug=False)
