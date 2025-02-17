from include.include import *
from include.include_datasets import *

class CustomedDataset(Dataset):
  def __init__(self, texts,texts_previous,emotions,emotions_previous,target_texts,target_emotions):
  # Dataset object for Daily Dialog dataset
    self.texts = texts                       ## tokenized text
    self.emotions = emotions                 ## tokenized emotions
    self.target_texts = target_texts         ## target text for loss computation
    self.target_emotions = target_emotions
    self.target_emotions_previous = emotions_previous
    self.target_texts_previous = texts_previous
  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    item = {
                         'texts': np.array(self.texts[idx]),
                      'emotions': np.array(self.emotions[idx]),
                  'target_texts': np.array(self.target_texts[idx]),
               'target_emotions': np.array(self.target_emotions[idx]),
      'target_emotions_previous': np.array(self.target_emotions_previous[idx]),
         'target_texts_previous': np.array(self.target_texts_previous[idx]),
    }
    return item

