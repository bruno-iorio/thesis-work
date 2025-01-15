# SimpleModel.ipynb

## preprocessing:
$$
D_1 = [[utt, utt,\cdots], [utt, utt, \cdots], \cdots] \longrightarrow [[word,word, \cdots], [word, word, \cdots],\cdots] 
$$
 
This is essentially unwrapping each utterance into a sequence of words, and flattening them.

After I rearrange the data into: 

$$
D_2 = [word, word, \cdots]
$$

We just extend the list of words.

## The model:
As it stands right now, It is just a NN model with no weight updating (yet). Just to test and fix the 
bad complexity, and preprocessing of SimpleModel (the previous one)

## Issues:
- Too simple 
- preprocessing requires refinement

