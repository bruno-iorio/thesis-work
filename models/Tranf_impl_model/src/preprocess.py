
##############################################################
##############################################################
###### Preprocess functions for ERC task #####################
###### Author: Bruno Iorio, École Polytechnique ##############
###### Supervisor: Gaël Guibon ###############################
###### Project: Bachelor Thesis ##############################
###### Date: February, 2025 ##################################
##############################################################
##############################################################

from include.include import *
from include.include_preprocess import *

def get_encoder():
    ## simple to get the encoder embedding vectors
    return gensim.models.KeyedVectors.load_word2vec_format("./data/wiki-news-300d-1M.vec", binary = False)

def get_embeddings(encoder_model):
    ## We create the embeddings and find the vocab
    pad_token, sep_token, unk_token = '<pad>', '<sep>', '<unk>'
    embedding_vectors = torch.from_numpy(encoder_model.vectors)
    pretrained_vocab = copy.deepcopy(encoder_model.index_to_key)
    pretrained_vocab[:0] = [pad_token, sep_token ,unk_token]
    stoi = {word: i for i, word in enumerate(pretrained_vocab)}
    itos = {i: word for i, word in enumerate(pretrained_vocab)}
    pretrained_embeddings = torch.cat((torch.zeros(1,embedding_vectors.shape[1]),embedding_vectors))
    pretrained_embeddings = torch.cat((torch.ones(1,embedding_vectors.shape[1]),embedding_vectors))
    pretrained_embeddings = torch.cat((-torch.ones(1,embedding_vectors.shape[1]),embedding_vectors))
    return pretrained_embeddings,embedding_vectors, stoi, itos

def get_tokenizer():
    ## simple to create a tokenizer
    return TweetTokenizer()

def tokenize_text_extend_emotions(text, emotion, stoi, tok): 
    ## utteration : string -> list of tokenized words : [int]
    text = tok.tokenize(text)
    text = [stoi[word] if word in stoi else stoi['<unk>'] for word in text]
    emotion = [emotion for _ in range(len(text))]
    return text, emotion

## done / to review
def concat_utt(dialog, emotions, stoi,tok,max_size=max_size):
    ## list of utterations : [string] -> list of list of tokenized words : [int]
    tokenized_and_extended = [tokenize_text_extend_emotions(t, e, stoi, tok) for t,e in zip(dialog,emotions)]
    dialog = [i[0] for i in tokenized_and_extended]
    emotions = [i[1] for i in tokenized_and_extended]
    dialog_flat = []
    emotions_extended = ['NE']
    for i in range(len(dialog) - 1):
        dialog[i].append(stoi["<sep>"])
        emotions[i].append('NE') # number of emotions
    for i in range(len(dialog)):
        dialog_flat.extend(dialog[i])
        emotions_extended.extend(emotions[i])
    if len(emotions_extended) > max_size:
        dialog_flat = dialog_flat[:max_size]
        emotions_extended = emotions_extended[:max_size]
    else:
        dialog_flat.extend([0]*(max_size - len(dialog_flat)))
        emotions_extended.extend(['NE']*(max_size - len(emotions_extended)))
    return dialog_flat,emotions_extended

def preprocess_data(X,Y,stoi,tok): 
    ## list of lists of utterations : [[string]] -> list of lists of tokenized words : [[int]]
    X_processed = []
    Y_processed = []
    for i in range(len(X)):
        x,y = concat_utt(X[i],Y[i],stoi,tok)
        X_processed.append(x)
        Y_processed.append(y)
    return X_processed, Y_processed

def get_target(X,Y): 
    ## generates the target values and input values
    text_input     = [i[:-1] for i in X]
    text_target    = [i[1:] for i in X]
    emotion_input  = [i[:-1] for i in Y]
    emotion_target = [i[1:] for i in Y]
    return text_input, text_target, emotion_input, emotion_target

def label_to_index(Y,unk='NE',lookup=None):
    found = (lookup is None) ## True if not passed (train) False else
    lookup = {} if (lookup is None) else lookup # {} if not passed {train} False else
    k = 0
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if (Y[i][j] not in lookup) and found: # if train and Y is not in lookup then add Y to lookup.keys()
                lookup[Y[i][j]] = k
                Y[i][j] = k
                k += 1
                continue
            elif Y[i][j] in lookup: # if Y in lookup then assign different value..
                Y[i][j] = lookup[Y[i][j]]
                continue
            elif not found: # if it is passed but Y not in
                Y[i][j] = lookup[unk]  
    return Y, lookup

##############################################################
####################### For MELD #############################
##############################################################

def parse_meld(df,lookup = None):
    ## parses the MELD dataset
    X, Y = {}, {}
    for _, row in df.iterrows():
        dialog_id = row['Dialogue_ID']
        if dialog_id not in X:
            X[dialog_id] = []
            Y[dialog_id] = []
        X[dialog_id].append(row['Utterance'])
        Y[dialog_id].append(row['Emotion'])
    X = list(X.values())
    Y = list(Y.values())
    Y, lookup = label_to_index(Y,lookup)
    return X, Y, lookup

##############################################################
####################### For EmoryNLP #########################
##############################################################
def parse_seasons(episodes): 
    ## annoying parsing
    X = []
    Y = []
    for episode in episodes:
        for scene in episode['scenes']:
            dialog = []
            emotion = []
            for utterance in scene['utterances']:
                if utterance['transcript'] != '':
                    dialog.append(utterance['transcript'])
                    emotion.append(utterance['emotion'])
            X.append(dialog)
            Y.append(emotion)
    return X, Y

def parse_emory():
    ## getting from the web
    json_train = 'https://raw.githubusercontent.com/emorynlp/emotion-detection/refs/heads/master/json/emotion-detection-trn.json'
    json_test = 'https://raw.githubusercontent.com/emorynlp/emotion-detection/refs/heads/master/json/emotion-detection-dev.json'
    json_val = 'https://raw.githubusercontent.com/emorynlp/emotion-detection/refs/heads/master/json/emotion-detection-tst.json'
    train = requests.get(json_train)
    test = requests.get(json_test)
    val = requests.get(json_val)

    train = json.loads(train.text)['episodes']
    test = json.loads(test.text)['episodes']
    val = json.loads(val.text)['episodes']
    X_train, Y_train = parse_seasons(train)
    X_test, Y_test = parse_seasons(test) 
    X_val, Y_val = parse_seasons(val)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def save_dict_json(path,dic):
    with open(path, 'w+') as file:
        json.dump(dic,file,indent=4)

def load_dict_json(path):
    loaded = dict()
    with open(path,'r') as file:
        loaded = json.load(file)
    return loaded

