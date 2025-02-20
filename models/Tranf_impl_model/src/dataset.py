from include.include import *
from include.include_datasets import *

class CustomedDataset(Dataset):
    def __init__(self, texts, emotions,target_texts,target_emotions):
    # Dataset object for Daily Dialog dataset
        self.texts = texts                       ## tokenized text
        self.emotions = emotions                 ## tokenized emotions
        self.target_texts = target_texts         ## target text for loss computation
        self.target_emotions = target_emotions   ## target emotions       ===

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
                   'texts': np.array(self.texts[idx]),
                'emotions': np.array(self.emotions[idx]),
            'target_texts': np.array(self.target_texts[idx]),
         'target_emotions': np.array(self.target_emotions[idx]),
        }
        return item

