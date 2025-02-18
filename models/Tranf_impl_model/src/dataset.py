from include.include import *
from include.include_datasets import *

class CustomedDataset(Dataset):
  def __init__(self, texts, emotions,decoded_emotions,target_texts,target_emotions,target_decoded_emotions):
  # Dataset object for Daily Dialog dataset
    self.texts = texts                       ## tokenized text
    self.emotions = emotions                 ## tokenized emotions
    self.decoded_emotions = decoded_emotions 
    self.target_texts = target_texts         ## target text for loss computation
    self.target_emotions = target_emotions   ## target emotions for loss computation
    self.target_decoded_emotions = target_decoded_emotions

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    item = {
                'texts': np.array(self.texts[idx]),
             'emotions': np.array(self.emotions[idx]),
         'target_texts': np.array(self.target_texts[idx]),
      'target_emotions': np.array(self.target_emotions[idx]),
                  'dec': np.array(self.decoded_emotions[idx]),
           'target_dec': np.array(self.target_decoded_emotions[idx])
    }
    return item

