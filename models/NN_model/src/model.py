from include.include import *
from include.include_model import *

class SimpleModel(nn.Module):
    def __init__(self, embeddings, emo_dim, n_emotion, n_vocab):
        super(SimpleModel,self).__init__()
        ## word_dim = 300
        self.embedding_layer_text = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.embedding_layer_emotion = nn.Embedding(n_emotion, emo_dim)
        ## Channel for utterances/words:
        self.Linear_utt1 = nn.Linear(300,80)
        self.Linear_utt2 = nn.Linear(80,80)
        self.Linear_utt3 = nn.Linear(80,80)

        ## Channel for emotions:
        self.Linear_emo1 = nn.Linear(emo_dim,80)
        self.Linear_emo2 = nn.Linear(80,80)
        self.Linear_emo3 = nn.Linear(80,80)
        # self.Linear_emo3.requires_grad = False

        ## fusion by concatenation and Linear layer:
        self.Linear_fus = nn.Linear(160,300)

    ## We concatenate and do linear again (2 different concatenations)
        self.Linear_utt_final1 = nn.Linear(80 + 80  + 300, 180)
        self.Linear_utt_final2 = nn.Linear(180, 100)
        self.Linear_utt_final3 = nn.Linear(100, 200)
        self.Linear_utt_final = nn.Linear(200, n_vocab - 2) ## test remove unk and sep from teh predictions


        self.Linear_emo_final = nn.Linear(160, n_emotion)

        self.softmax = nn.Softmax(dim=0)


    def forward(self, text, emotion, hidden):
        with torch.no_grad():
            text = self.embedding_layer_text(text)
        text = self.Linear_utt1(text)
        text = self.Linear_utt2(text)
        text = self.Linear_utt3(text)

        emotion = self.embedding_layer_emotion(emotion)
        emotion = self.Linear_emo1(emotion)
        emotion = self.Linear_emo2(emotion)
        emotion = self.Linear_emo3(emotion)

        hidden = hidden +  emotion*0.6 + text*0.4
        z = torch.cat((text,emotion),-1)
        z = self.Linear_fus(z)
        
        emotion = torch.cat((text,hidden),-1)
        emotion = self.Linear_emo_final(emotion)
        emotion = self.softmax(emotion)

        text = torch.cat((z,hidden,text),-1)
        text = self.Linear_utt_final1(text)
        text = self.Linear_utt_final2(text)
        text = self.Linear_utt_final3(text)
        text = self.Linear_utt_final(text)
        text = self.softmax(text)
        
        return text, emotion, hidden
        #return emotion
