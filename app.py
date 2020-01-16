from flask import Flask, request
from flask_cors import CORS
import torch
from pytorch_transformers import BertTokenizer, BertForMaskedLM
import nltk

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
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

	predicted_index = torch.argmax(predictions[0, masked_index]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
	return sentence_orig.replace('____', '<font color="red"><b>'+predicted_token+'</b></font>')

if __name__=='__main__':
	app.run(debug=True)