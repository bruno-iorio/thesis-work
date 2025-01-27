import copy
from include.include import *
from include.include_datasets import *

def get_encoder():
    return encoder_model = gensim.models.KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec", binary = False)

def get_embeddigs():
    ## We create the embeddings and find the vocab
    unk_token, sep_token = '<unk>', '<sep>'
    embedding_vectors = torch.from_numpy(encoder_model.vectors) ## TODO: remove least frequent
    pretrained_vocab = copy.deepcopy(encoder_model.index_to_key)
    pretrained_vocab[:0] = [unk_token,sep_token]

    stoi = {word: i for i, word in enumerate(pretrained_vocab)}
    itos = {i: word for i, word in enumerate(pretrained_vocab)}

    pretrained_embeddings = torch.cat((torch.ones(1,embedding_vectors.shape[1]),embedding_vectors))
    pretrained_embeddings = torch.cat((-torch.ones(1,embedding_vectors.shape[1]),embedding_vectors))
    return pretrained_embeddings, stoi, itos

def get_tokenizer():
    return TweetTokenizer()

def tokenize_text_extend_emotions(text, emotion, stoi, tok): ## utteration : string -> list of tokenized words : [int]
  text = tok.tokenize(text)
  text = [stoi[word] if word in stoi else stoi['<unk>'] for word in text]
  return text, [emotion]*len(text)

def concat_utt(dialog, emotions, stoi): ## list of utterations : [string] -> list of list of tokenized words : [int]
  tokenized_and_extended = [tokenize_text_extend_emotions(t,e,stoi) for t,e in zip(dialog,emotions)]
  dialog = [i[0] for i in tokenized_and_extended]
  emotions = [i[1] for i in tokenized_and_extended]
  dialog_flat = []
  emotions_extended = []
  for i in range(len(dialog) - 1):
    dialog[i].append(stoi["<sep>"])
    emotions[i].append(7) ## number of emotions
  for i in range(len(dialog)):
    dialog_flat.extend(dialog[i])
    emotions_extended.extend(emotions[i])
  return dialog_flat,emotions_extended

def preprocess_data(X,Y,stoi): ## list of lists of utterations : [[string]] -> list of lists of tokenized words : [[int]]
  X_processed = []
  Y_processed = []
  for i in tqdm(range(len(X))):
    X_processed.append(concat_utt(X[i],Y[i],stoi)[0])
    Y_processed.append(concat_utt(X[i],Y[i],stoi)[1])
  return X_processed, Y_processed

def get_target(X,Y): ## generates the target values and input values
  text_input = [i[:-1] for i in X]
  text_target = [i[1:] for i in X]
  emotion_input = [i[:-1] for i in Y]
  emotion_target = [i[1:] for i in Y]
  return text_input, text_target, emotion_input, emotion_target

## For EmoryNLP:
def parse_seasons(episodes,lookup=None): ## annoying parsing
  lookup = {} if lookup is None else lookup
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
  Y,lookup = change_Y(Y)
  return X, Y, lookup

def parse_emory(): ## getting from the web
  json_train = 'https://raw.githubusercontent.com/emorynlp/emotion-detection/refs/heads/master/json/emotion-detection-trn.json'
  json_test = 'https://raw.githubusercontent.com/emorynlp/emotion-detection/refs/heads/master/json/emotion-detection-dev.json'
  json_val = 'https://raw.githubusercontent.com/emorynlp/emotion-detection/refs/heads/master/json/emotion-detection-tst.json'
  train = requests.get(json_train)
  test = requests.get(json_test)
  val = requests.get(json_val)

  train = json.loads(train.text)['episodes']
  test = json.loads(test.text)['episodes']
  val = json.loads(val.text)['episodes']
  X_train, Y_train, lookup = parse_seasons(train)
  X_test, Y_test, _ = parse_seasons(test)
  X_val, Y_val, _ = parse_seasons(val)
  return X_train, Y_train, X_test, Y_test, X_val, Y_val, lookup


## to get top n most frequent words:
def get_topk(X,k): ## get the top k most frequent words - has to get a flatten version of the input
  X = [j for i in X for j in i]
  flat_X = [] 
  for i in X:
    flat_X.extend(tok.tokenize(i))
  c = Counter(flat_X)
  top_k = heapq.nlargest(k, c.items(), key=lambda x: x[1])
  new_stoi = {'<unk>' : 0, '<sep>' : 1}
  res = []
  k = 2
  for (word,_) in top_k:
    if word in stoi:
      new_stoi[word] = k
      k += 1
      res.append(pretrained_embeddings[stoi[word]])
  res[:0] = [torch.ones(embedding_vectors.shape[1]),-torch.ones(embedding_vectors.shape[1])]
  new_itos = {index : word for (word,index) in new_stoi.items()}
  return torch.stack(res), new_stoi, new_itos

