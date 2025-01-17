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

## Issues: 
- preprocessing requires refinement (as always)
- adding the loss for the text token actually slows down the training (impossible to train in the cpu)

