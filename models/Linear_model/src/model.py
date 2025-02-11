from include.include import *
from include.include_model import *



class SentenceSimpleModel(nn.Module):
    def __init__(self, embeddings,text_dim,emo_dim, n_emotion, n_vocab):
        super(SentenceSimpleModel,self).__init__()
        ## word_dim = 300
        self.embedding_layer_emotion = nn.Embedding(n_emotion, emo_dim)
        ## Channel for utterances/words:
        self.Linear_utt1 = nn.Linear(text_dim,200)
        self.Linear_utt2 = nn.Linear(200,300)

        ## Channel for emotions:
        self.Linear_emo1 = nn.Linear(emo_dim,200)
        self.Linear_emo2 = nn.Linear(200,300)
        # self.Linear_emo3.requires_grad = False

        ## fusion by concatenation and Linear layer:
        self.Linear_fus = nn.Linear(600,300)
        self.Hidden_weight = nn.Linear(300,300)

        ## We concatenate and do linear again (2 different concatenations)
        self.Linear_utt_final1 = nn.Linear(300 + 300  + 300, 300)
        self.Linear_utt_final2 = nn.Linear(300, text_dim)

        self.Linear_emo_final1 = nn.Linear(300 + 300 , 200)
        self.Linear_emo_final2 = nn.Linear(200, n_emotion)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0) ## Chat said this is wrong ## TODO: Debug to check if it is actually wrong
        self.Dropout = nn.Dropout(p = 0.5)
      
    def forward(self, text, emotion, hidden):
        text = self.Linear_utt1(text)
        text = self.relu(text)
        text = self.Dropout(text)
        text = self.Linear_utt2(text)
        text = self.relu(text)

        emotion = self.embedding_layer_emotion(emotion)
        emotion = self.Linear_emo1(emotion)
        emotion = self.relu(emotion)
        emotion = self.Dropout(emotion)
        emotion = self.Linear_emo2(emotion)
        emotion = self.relu(emotion)
        
        hidden = hidden + self.Hidden_weight(emotion*0.7 + text*0.3)
        hidden = self.tanh(hidden)

        z = torch.cat((text,emotion),-1)
        z = self.Linear_fus(z)
        z = self.relu(z)
        
        emotion = torch.cat((text,z),-1)
        emotion = self.Linear_emo_final1(emotion)
        emotion = self.relu(emotion)
        emotion = self.Dropout(emotion)
        emotion = self.Linear_emo_final2(emotion)
        emotion = self.relu(emotion)

        text = torch.cat((z,hidden,text),-1)
        text = self.Linear_utt_final1(text)
        text = self.relu(text)
        text = self.Dropout(text)
        text = self.Linear_utt_final2(text)
        return text, emotion, hidden
    #return emotion 
    def infer(self, text,emotion, hidden):
        text,emotion,hidden = self.forward(text,emotion,hidden)
        text = self.softmax(text)
        emotion = self.softmax(emotion) 
        return text, emotion, hidden
    def init_hidden(self,batch_size,hidden_size,device):
        return torch.zeros((batch_size,hidden_size), requires_grad=True).to(device)


class SimpleModel(nn.Module): ## Model unnecessarily complicated! ## TODO:  remove the sequential Linear Layers to see if becomes faster
    def __init__(self, embeddings,emo_dim, n_emotion, n_vocab):
        super(SimpleModel,self).__init__()
        ## word_dim = 300
        self.embedding_layer_text = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.embedding_layer_emotion = nn.Embedding(n_emotion, emo_dim)
        ## Channel for utterances/words:
        self.Linear_utt1 = nn.Linear(300,600)
        self.Linear_utt2 = nn.Linear(600,400)

        ## Channel for emotions:
        self.Linear_emo1 = nn.Linear(emo_dim,600)
        self.Linear_emo2 = nn.Linear(600,400)
        # self.Linear_emo3.requires_grad = False

        ## fusion by concatenation and Linear layer:
        self.Linear_fus = nn.Linear(800,500)
        self.Hidden_weight = nn.Linear(400,400)

        ## We concatenate and do linear again (2 different concatenations)
        self.Linear_utt_final1 = nn.Linear(400 + 400  + 500, 500)
        self.Linear_utt_final2 = nn.Linear(500, n_vocab)

        self.Linear_emo_final1 = nn.Linear(400 + 500 , 600)
        self.Linear_emo_final2 = nn.Linear(600, n_emotion)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0) ## Chat said this is wrong ## TODO: Debug to check if it is actually wrong
        self.Dropout = nn.Dropout(p = 0.5)
      
    def forward(self, text, emotion, hidden):
        text = self.embedding_layer_text(text)
        text = self.Linear_utt1(text)
        text = self.relu(text)
        text = self.Dropout(text)
        text = self.Linear_utt2(text)
        text = self.relu(text)

        emotion = self.embedding_layer_emotion(emotion)
        emotion = self.Linear_emo1(emotion)
        emotion = self.relu(emotion)
        emotion = self.Dropout(emotion)
        emotion = self.Linear_emo2(emotion)
        emotion = self.relu(emotion)
        
        hidden = hidden + self.Hidden_weight(emotion*0.7 + text*0.3)
        hidden = self.tanh(hidden)

        z = torch.cat((text,emotion),-1)
        z = self.Linear_fus(z)
        z = self.relu(z)
        
        emotion = torch.cat((text,z),-1)
        emotion = self.Linear_emo_final1(emotion)
        emotion = self.relu(emotion)
        emotion = self.Dropout(emotion)
        emotion = self.Linear_emo_final2(emotion)
        emotion = self.relu(emotion)

        text = torch.cat((z,hidden,text),-1)
        text = self.Linear_utt_final1(text)
        text = self.relu(text)
        text = self.Dropout(text)
        text = self.Linear_utt_final2(text)
        text = self.relu(text)
        return text, emotion, hidden
    #return emotion 
    def infer(self, text,emotion, hidden):
        text,emotion,hidden = self.forward(text,emotion,hidden)
        text = self.softmax(text)
        emotion = self.softmax(emotion) 
        return text, emotion, hidden
    def init_hidden(self,batch_size,hidden_size,device):
        return torch.zeros((batch_size,hidden_size), requires_grad=True).to(device)

def load_model(path,embeddings,emo_dim, n_emotions, n_vocab):
    model = SimpleModel(embeddings,emo_dim,n_emotions,n_vocab)
    model.load_state_dict(torch.load(path,weights_only=False,map_location=torch.device('cpu')))
    return model

