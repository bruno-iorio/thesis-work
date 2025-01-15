# SimpleModel.ipynb

## preprocessing:
$$
D_1 = [[utt, utt,\cdots], [utt, utt, \cdots], \cdots] \longrightarrow [[word,word, \cdots], [word, word, \cdots],\cdots] 
$$
 
This is essentially unwrapping each utterance into a sequence of words, and flattening them.

After I rearrange the data into: 
$$
D_2 = [[[word_{1,1},word_{1,2},\cdots], [word_{2,1},\cdots],\cdots]]
$$

Where $word_{i,k}$ is the $i$-th word of the $k$-th conversation.

## The model:
The model consists only of linear layers, with an input channel for the emotions encoding and another 
for the words encoding. The only special part is that there is a layer that is composed a list of Linear
layers. We, then, pass the i-th, and j-th word in different layers of this list.

## Why this idea?
 Although it doesn't work (as expected), it was a "somewhat" (disputed claim) straightforward way of
 having some kind of matrix update when we analyse a sequence of words.

## Issues:
- Doesn't converge properly during training.
- Takes a bit too long to train, which shouldn't be the case given.



