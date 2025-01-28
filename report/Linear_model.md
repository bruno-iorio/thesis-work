# Linear_model.ipynb
## Data: 

 - Daily Dialog

## preprocessing:

$$
D_1 = [[utt, utt,\cdots], [utt, utt, \cdots], \cdots] \longrightarrow [[word,word, \cdots], [word, word, \cdots],\cdots] 
$$

This is essentially unwrapping each utterance into a sequence of words, and flattening them.
We tokenize each word and emotion. We proceed by adding a padding (token = 0),so every 
dialog/emotion-flow has the same dimension.


In addition, we use a pre-trained embedding model of dimension 300 and a vocab size of 
roughly 1,000,000.


## The model:
2 input channels: emotion tokens and text tokens
 - For each input channel:
  1) 1 Embedding layer;
  2) 3 Linear Layers,
 - Fusion network:
  1) Concatenation
  2) Linear Layer 

2 Output channels: for predicted emotion, and predicted text
  - For each output channel:
  1) concatenation with the fusion layer and the input channel 
  2) 1 or more linear layers.
  3) Softmax function


## Metrics: 

           -   precision    recall  f1-score   support

           0       0.24      0.73      0.36      3735
           1       0.23      0.19      0.21      2403
           2       0.00      0.00      0.00      3587
           3       0.15      0.11      0.13      2003
           4       0.34      0.27      0.30      2300
           5       0.03      0.01      0.02      1273
           6       0.37      0.26      0.30      2053
           7       0.00      0.00      0.00      1245

        accuracy     -        -        0.25     18599
       macro avg   0.17      0.20      0.16     18599
    weighted avg   0.18      0.25      0.19     18599


# Diagnose of problems:
 
  The main issue of this model might be its simplicity. In fact, such a simple model is not enough to describe all the components we need.

The model incurs in the mistake of consistently over-fitting whenever it can:
 1) It over-fits the emotion prediction;
 2) It over-fits the word prediction (chooses always "unk" token or "," token which are very common)

Why the model does it:
 1) We can easily justify it for the emotions, because the model just learns when to change the emotions i.e. whenever it sees a "sep" token, it understands it is time to change the emotion, which is literally not the kind of behaviour we are looking for 
 2) For the words it gets slightly more complicated, but this might be an issue related to the loss. I presume it must be due to the fact that the model can only find an easy local minimum in the optimization process, which is just choosing the most frequent tokens, instead of learning how to predict properly the next token;

# What I tried and did not work:

I tried giving more priority to the loss of the words, instead of the emotions. I also tried making the prediction of the emotion more dependent on the word prediction; Both of these approaches reduced the over-fitting for the emotion prediction, but increased significantly the over-fitting for the word prediction.

Consequently, we couldn't have a decent strategy for this task using this simple model;
