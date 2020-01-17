# Fill-in-the-BERT
Fill-in-the-BERT is a fill-in-the-blanks model that is trained to predict the missing word in the sentence. For the purpose of this demo we will be using pre-trained bert-base-uncased as our prediction model. If you are new to BERT, please read [BERT-Explained](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) and [Visual Guide to BERT](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

## Steps to Run
````
$> python3 app.py
````
Open index.html in the browser and start typing :speech_balloon:



## Demo
![Fill in the blanks - BERT demo](https://github.com/prakhar21/Fill-in-the-BERT/blob/master/demo.gif)

## Details
* __Model__ - bert-base-uncased
* __Pre-trained Task__ - MaskedLM

_P.S. The attention visualisation is done for layer 3 across all attention heads by taking their average._ Read more about heads and what they mean at [Visualizing inner workings of attention](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)

## Technologies Used
1. PyTorch
2. HTML/Bootstrap
3. Flask
