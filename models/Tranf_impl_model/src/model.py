from include.include import *
from include.include_model import *


## Attention module:
class AttentionModule(nn.Module): # Bahdanau attention module but not recursive
    def __init__(self,encode_dim,device):
        super(AttentionModule,self).__init__()
        self.Wk = nn.Linear(encode_dim,encode_dim)
        self.Wq = nn.Linear(encode_dim,encode_dim)
        self.Wv = nn.Linear(encode_dim, 1)
        self.device = device
    def forward(self,query,key):
        ## query: [B, encode_dim]
        ## key: [B,seq_len,encode_dim]


        scores = self.Wv(torch.tanh(self.Wk(query) + self.Wq(key))) # [B, seq_len ,1]
        scores = scores.squeeze(2).unsqueeze(1) # [B, 1, seq_len]
        weights = F.softmax(scores,dim=-1) # [B, 1, seq_len]
        context = torch.bmm(weights, key) # [B, 1, encode_dim]
        return context, weights ## Have to add the context vector also


## Encoder Module
class Encoder(nn.Module):
    def __init__(self, decode_dim, encode_dim, device,sequence_size = 200, embeddings = None):
        super(Encoder,self).__init__()
        self.embedding = nn.Embeddings(decode_dim,encode_dim) if embeddings is None else nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.encoder = nn.Linear(encode_dim,encode_dim)
        self.sequence_size = sequence_size
        self.device = device
    def forward(self, encoder_input): 
        ## We remove the use of hidden states...
        ## encoder_input: [B, seq_len, decode_dim]
        out = self.embedding(encoder_input) ## [B, seq_len ,encode_dim]
        out = self.encoder(out) # [B, seq_len, encode_dim]
        out = F.relu(out)
        out = F.Softmax(out)
        return out
        
# Attention Decoder:
class Decoder(nn.Module):
    def __init__(self,encode_dim, decode_dim,device,sequence_size=200):
        super(Decoder,self).__init__()
        self.embedding = nn.Embedding(decode_dim,encode_dim)
        self.decoder = nn.Linear(encode_dim,decode_dim)
        self.attention = AttentionModule()
        self.dropout = nn.Dropout(0.5)
        self.sequence_size = sequence_size
        self.device = device

    def forward(self,encoder_outputs, target_tensor=None):
        ## encoder_ouputs [B, seq_len, encode_dim]
        ## target_tensor [B, seq_len]
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros(batch_size,1).to(self.device) ## [B, 1]
        decoder_output = []
        attentions = []

        for i in range(self.sequence_size):
            decoder_output, attn_weights = self.forward_step(decoder_input,encoder_outputs) ## [B,1,encode_dim],[B,1,seq_len]
            decoder_outputs.append(decoder_output) 
            attentions.append(attn_weights)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) ## [B, 1]
            else:
                _ , topi = decoder_output.topk(1) ## [B ,1 , 1]
                decoder_input = topi.squeeze(-1).detach() ## [B, 1]
        decoder_outputs = torch.cat(decoder_outputs,dim=1) ## [B, seq_len]
        decoder_outputs = F.log_softmax(decoder_ouputs,dim=-1) 
        attentions = torch.cat(attentions,dim=1) ## [B,seq_len,seq_len]
        return decoder_outputs, attentions

    def forward_step(self,decoder_input,encoder_outputs):
        ## decode input: [B,1]
        ## encoder_outputs: [B,seq_len,encode_dim]
        embedd = self.embedding(decoder_input) ##[B,encode_dim]
        context, weights = self.attention.forward(embedd,decoder_input) ## [B, 1,encode_dim], [B,1,seq_len]
        out = F.relu(context) ## [B,1,encode_dim]
        out = self.decoder(out)
        out = F.relu(out)
        return out, weights


class TransfModel(nn.Module):
    def __init__(self, embeddings, n_vocab, n_emotions, device,sequence_size=200):
        super(TransfModel,self).__init__()
        self.device = device
        self.sequence_size = sequence_size
        
        self.encoder_text = Encoder(300,300,device,embeddings=embeddings)
        self.encoder_emo = Encoder(n_emotions,300,device)


        self.Linear_text = nn.Linear(300,300) 
        self.Linear_emo = nn.Linear(300,300)

        ## for the Fusion layer // attention-based
        self.attention = AttentionModule(300,device)
        self.Linear_fusion = nn.Linear(300,300) ## TODO: get dimensions

        self.Linear_text_final = nn.Linear(300,300)  ## Decide dimensions
        self.Linear_emo_final = nn.Linear(300,300)  ## Decide dimenstions

        self.decoder_text = Decoder(300, n_vocab,device)
        self.decoder_emo = Decoder(300, n_emotions,device)

    def forward(self,text,emotion,target_tensor):
        ## text: [B,seq_len, 1]
        ## emotion: [B,seq_len, 1]
        text = self.encoder_text(text) ## [B,seq_len,300]
        text = self.Linear_text(text) ## [B,seq_len,300]
        text = F.relu(text) ## [B,seq_len,300]

        emotion = self.encoder_emo(emotion) ## [B, seq_len, 300]
        emotion = self.Linear_emo(emotion) ## [B, seq_len, 300]
        emotion = F.relu(emotion) ## [B, seq_len, 300]

        decoder_inp = []
        for i in range(text.size(1)): ## text[:,i, :]: [B, 300]
            context, _ = self.attention(text[:,i,:]) ## [B, 1, 300], [B,1,seq_len]
            context = self.Linear_fusion(context) ## [B, 1, 300]
            context = F.relu(context) ## [B,1,300]
            decoder_inp.append(context)
        decoder_inp = torch.cat(decoder_inp, dim=1) ## [B,seq_len,300]
        out_emotion, _ = self.decoder_emo(context,target_tensor) ## [B,seq_len]
        return out_emotion


def load_model(path,embeddings,device, n_emotions, n_vocab):
    model = TransfModel(embeddings,n_vocab,n_emotions,device)
    model.load_state_dict(torch.load(path,weights_only=False,map_location=torch.device('cpu')))
    return model
