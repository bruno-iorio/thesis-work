# SimpleModel1.ipynb

## Data:
 - Daily Dialog

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


At the end we have the following inputs: 

$$
I_1 = [word_1, word_2, word_3, \cdots, word_{n-1}] 
$$

$$
I_2 = [emo_1, emo_2, emo_3, \cdots , emo_{n-1}] 
$$

And we have our targets:

$$
T_1 = [word_2, word_3, word_4,\cdots, word_{n}] 
$$

$$
T_2 = [emo_2, emo_3, emo_4, \cdots , emo_{n}] 
$$

In this case we use "context = 1": we only use $word_k$ and $emo_k$ to predict
$word_{k+1}$ and $emo_{k+1}$.

## The model:
As it stands right now, It is just a NN model with no weight updating (yet). Just to test and fix the 
bad complexity, and preprocessing of SimpleModel (the previous one)

## Loss:
           -   precision    recall  f1-score   support
           1       0.89      0.96      0.93     82011
           2       0.00      0.00      0.00      1083
           3       0.00      0.00      0.00        41
           4       0.00      0.00      0.00       126
           5       0.90      0.00      0.01      6231
           6       0.00      0.00      0.00      1085
           7       0.00      0.00      0.00      1033
     accuracy        -         -       0.86     91610
     macro avg     0.26      0.14      0.13     91610
     weighted avg  0.86      0.86      0.83     91610

## Issues:
- Too simple 
- preprocessing requires refinement

