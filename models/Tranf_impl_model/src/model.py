from include.include import *
from include.include_model import *

class AdditiveAttention(nn.Module): ## Bahdanau attention // Taken from sooftware/attentionsa git
    def __init__(self, hidden_dim: int):
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.Wa = nn.Linear(hidden_dim, 1)

    def forward(self, query, key):
        ## query : [B, hidden_dim]
        ## key   : [B, seq_len , hidden_dim]

        score = self.Wa(torch.tanh(self.key_proj(key) + self.query_proj(query).unsqueeze(1) + self.bias)).squeeze(-1) ## [B,seq_len]
        attn = F.softmax(score, dim=-1) ## [B, seq_len]
        context = torch.bmm(attn.unsqueeze(1), key) ## [B, 1, hidden_dim]
        return context, attn

class AttentionModel(nn.Module):
    def __init__(self,embeddings,encoded_text, encoded_emo, n_vocab,n_emotion):
        super(AttentionModel,self).__init__()
        self.attention = AdditiveAttention(encoded_text)
        self.multi_head_attention1 = nn.MultiheadAttention(embed_dim=encoded_text,num_heads=20,dropout = 0.2, batch_first=True)
        self.multi_head_attention2 = nn.MultiheadAttention(embed_dim=encoded_text,num_heads=20,dropout = 0.2, batch_first=True)
        
        self.encoder_emo = nn.Embedding(n_emotion,encoded_text)
        self.encoder_text = nn.Embedding.from_pretrained(embeddings, freeze=True)
        #self.encoder_text = nn.Embedding(n_vocab,encoded_text)
        

        self.Linear_text1 = nn.Linear(encoded_text,encoded_text)
        self.Linear_text2 = nn.Linear(encoded_text, encoded_text)
        self.Linear_emo1 = nn.Linear(encoded_text,encoded_text)
        ## Final layers

        self.Final_emo = nn.Linear(encoded_text,n_emotion)
        self.Final_text1 = nn.Linear(encoded_text,encoded_text)
        self.Final_text2 = nn.Linear(encoded_text,n_vocab)
        self.dropout = nn.Dropout(0.2)
    def forward(self,input_text,target_emotions=None):
        ## input_text : [B, seq_len]
        ## target_emotions: [B, seq_len] 
        
        text = self.encoder_text(input_text).squeeze(-2) # [B,seq_len, encoded_text]
        text = self.Linear_text1(text)
        text = self.dropout(text)
        text = F.relu(text) ## [B,seq_len,encoded_text]
        
        text_out, _ = self.multi_head_attention1(text,text,text) ## self attentiln on the text [ B,seq_len, encoded_text]
        text_out = self.Linear_text2(text_out) ## [B,seq_len,encoded_text]
        
        query = torch.zeros(text_out.size(0),1,dtype=torch.long)
        emotion_out = []
        for i in range(text_out.size(1)):
            value = text_out
            key = text_out
            if target_emotions is not None:
                query = target_emotions[:,i] ## [B,1]

            query = self.encoder_emo(query) #[B, encoded_text]
            emo, _ = self.multi_head_attention2(query,key,value) #[B,encoded_text]
            
            emo = self.Linear_emo1(emo)
            emo = self.dropout(emo)
            emo = F.relu(emo)
            
            query = torch.argmax(emo,2)
            emotion_out.append(emo)
        
        
        emotion_out = torch.cat(emotion_out,dim=1)
        emotion_out = self.Final_emo(emotion_out)
        emotion_out = self.dropout(emotion_out)
        emotion_out = F.relu(emotion_out)
        
        text_out = self.Final_text1(text_out)
        text_out = self.dropout(text_out)
        text_out = F.relu(text_out)

        text_out = self.Final_text2(text_out)
        text_out = self.dropout(text_out)
        text_out = F.relu(text_out)
        return emotion_out, text_out
        


def load_model(path,embeddings,device, n_emotions, n_vocab):
    model = TransfModel(embeddings,n_vocab,n_emotions,device)
    model.load_state_dict(torch.load(path,weights_only=False,map_location=torch.device('cpu')))

