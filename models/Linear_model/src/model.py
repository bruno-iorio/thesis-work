from include.include import *
from include.include_model import *

class SimpleModel(nn.Module):
    def __init__(self, embeddings,emo_dim, n_emotion, n_vocab,n_dec):
        super(SimpleModel,self).__init__()
        ## word_dim = 300
        self.embedding_layer_text = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.embedding_layer_emotion = nn.Embedding(n_emotion, emo_dim)
        ## Channel for utterances/words:
        self.Linear_utt1 = nn.Linear(300,600)
        self.Linear_utt2 = nn.Linear(600,800)
        self.Linear_utt3 = nn.Linear(800,400)

        ## Channel for emotions:
        self.Linear_emo1 = nn.Linear(emo_dim,600)
        self.Linear_emo2 = nn.Linear(600,800)
        self.Linear_emo3 = nn.Linear(800,400)
        # self.Linear_emo3.requires_grad = False

        ## fusion by concatenation and Linear layer:
        self.Linear_fus = nn.Linear(800,500)
        self.Hidden_weight = nn.Linear(400,400)

        ## We concatenate and do linear again (2 different concatenations)
        self.Linear_utt_final1 = nn.Linear(400 + 400  + 500, 500)
        self.Linear_utt_final2 = nn.Linear(500, 600)
        self.Linear_utt_final3 = nn.Linear(600, 1000)
        self.Linear_utt_final = nn.Linear(1000, n_vocab) ## test remove unk and sep from teh predictions to see if it improves somethings (answer: not really)


        self.Linear_emo_final1 = nn.Linear(400 + 500 , 600)
        self.Linear_emo_final2 = nn.Linear(600, 700)
        self.Linear_emo_final3 = nn.Linear(700, n_emotion)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.Dropout = nn.Dropout(p = 0.5)
      
    def forward(self, text, emotion, hidden):
        text1 = self.embedding_layer_text(text)
        text2 = self.Linear_utt1(text1)
        text3 = self.relu(text2)
        text4 = self.Linear_utt2(text3)
        text5 = self.relu(text4)
        text6 = self.Linear_utt3(text5)

        emotion1 = self.embedding_layer_emotion(emotion)
        emotion2 = self.Linear_emo1(emotion1)
        emotion3 = self.Linear_emo2(emotion2)
        emotion4 = self.Linear_emo3(emotion3)

        hidden1 = hidden + self.Hidden_weight(emotion4*0.7 + text6*0.3)
        hidden2 = self.Dropout(hidden1)
        hidden3 = self.tanh(hidden2)

        z = torch.cat((text6,emotion4),-1)
        z1 = self.Linear_fus(z)
    
        emotion5 = torch.cat((text6,z1),-1)
        emotion6 = self.Linear_emo_final1(emotion5)
        emotion7 = self.Dropout(emotion6)
        emotion8 = self.Linear_emo_final2(emotion7)    
        emotion9 = self.Dropout(emotion8)
        emotion10 = self.relu(emotion9)
        emotion11 = self.Linear_emo_final3(emotion10)
        emotion12 = self.softmax(emotion11)

        text7 = torch.cat((z1,hidden3,text6),-1)
        text8 = self.Linear_utt_final1(text7)
        text9 = self.relu(text8)
        text10 = self.Linear_utt_final2(text9)
        text11 = self.Dropout(text10)
        text12 = self.Linear_utt_final3(text11)
        text13 = self.Dropout(text12)
        text14 = self.Linear_utt_final(text13)
        text15 = self.softmax(text14)
        return text15, emotion12, hidden3
    #return emotion
