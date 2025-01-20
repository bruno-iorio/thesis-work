# SimpleModel2.ipynb
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

Combining with the text loss:  

              precision    recall  f1-score   support

    (padding)0       0.13      1.00      0.23      2553
             1       1.00      0.35      0.52     42974
             2       0.07      0.17      0.10       554
             3       0.00      0.00      0.00        10
             4       0.00      0.00      0.00        67
             5       0.75      0.08      0.14      2433
             6       0.11      0.06      0.08       758
             7       0.00      0.00      0.00       651

          accuracy     -         -       0.36     50000
          macro avg  0.26      0.21      0.13     50000
       weighted avg  0.90      0.36      0.47     50000

## hyperparameters: 
 - epochs = 3
 - 16M trainable parameters
 - 316M parameters


## Issues: 
- preprocessing requires refinement (as always)
- adding the loss for the text token actually slows down the training (impossible to train in the cpu)

