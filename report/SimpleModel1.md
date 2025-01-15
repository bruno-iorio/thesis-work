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



              precision    recall  f1-score   support

           0       0.77      0.24      0.36     82011
           1       0.92      0.80      0.85      1083
           2       0.00      0.07      0.00        41
           3       0.24      0.55      0.33       126
           4       0.00      0.00      0.00      6231
           5       0.93      0.75      0.83      1085
           6       0.90      0.90      0.90      1033

    accuracy                           0.24     91610
   macro avg       0.54      0.47      0.47     91610
weighted avg       0.72      0.24      0.36     91610



