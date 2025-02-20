from include.include import *
from include.include_model import *


## Attention module:
class AttentionModule(nn.Module): # Bahdanau attention module but not recursive
    def __init__(self,encode_dim,device):
        super(AttentionModule,self).__init__()
        self.Wk = nn.Linear(encode_dim,encode_dim)
        self.Wq = nn.Linear(encode_dim,encode_dim)
        self.Wv = nn.Linear(encode_dim, 1)
        self.layer_norm = nn.LayerNorm(encode_dim)
        self.dropout = nn.Dropout(0.2)
        self.device = device

    def forward(self,query,key):
        ## query: [B, encode_dim]
        ## key: [B,seq_len,encode_dim]
        query = self.layer_norm(query)
        key = self.layer_norm(key)
        scores = self.Wv(torch.tanh(self.Wk(query).unsqueeze(1) + self.Wq(key))) ## [B, seq_len ,1]
        scores = self.dropout(scores)
        scores = scores / math.sqrt(self.Wk.weight.size(0))
        weights = scores.squeeze(2).unsqueeze(1) ## [B, 1, seq_len]
        context = torch.bmm(weights, key)        ## [B, 1, encode_dim]
        return context, weights

## Encoder Module
class Encoder(nn.Module):
    def __init__(self, decode_dim, encode_dim, device,sequence_size = 199, embeddings = None):
        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(decode_dim,encode_dim) if embeddings is None else nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.encoder = nn.Linear(encode_dim,encode_dim)
        self.sequence_size = sequence_size
        self.device = device
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(encode_dim)

    def forward(self, encoder_input): 
        ## We remove the use of hidden states...
        ## encoder_input: [B, seq_len, decode_dim]
        out = self.embedding(encoder_input) ## [B, seq_len , encode_dim, 1] || [B,encode_dim,1]
        out = out.squeeze(-2)               ## [B, seq_len ,encode_dim]     || [B,encode_dim]
        out = self.layer_norm(out)
        out = self.encoder(out)             ## [B, seq_len, encode_dim]     || [B,encode_dim]
        out = F.relu(out)
        out = self.dropout(out)
        return out
        
# Attention Decoder:
class Decoder(nn.Module):
    def __init__(self,encode_dim, decode_dim,device,sequence_size=199):
        super(Decoder,self).__init__()
        self.embedding = nn.Embedding(decode_dim,encode_dim)
        self.decoder = nn.Linear(encode_dim,decode_dim)
        self.attention = AttentionModule(encode_dim,device)
        self.dropout = nn.Dropout(0.2)
        self.sequence_size = sequence_size
        self.device = device

    def forward(self,encoder_outputs, target_tensor=None):
        ## encoder_ouputs [B, seq_len, encode_dim]
        ## target_tensor [B, seq_len]
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros(batch_size,1,dtype=torch.long).to(self.device) ## [B, 1]
        decoder_outputs = []
        attentions = []
        
        for i in range(self.sequence_size):
            decoder_output, attn_weights = self.forward_step(decoder_input, encoder_outputs) ## [B,1,encode_dim],[B,1,seq_len]
            decoder_outputs.append(decoder_output) 
            attentions.append(attn_weights)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) ## [B, 1]
            else:
                _ , topi = decoder_output.topk(1)          ## [B ,1 , 1]
                decoder_input = topi.squeeze(-1).detach()  ## [B, 1]

        decoder_outputs = torch.cat(decoder_outputs, dim=1) ## [B, seq_len]
        attentions = torch.cat(attentions, dim=1)          ## [B,seq_len,seq_len]
        return decoder_outputs, attentions

    def forward_step(self,decoder_input,encoder_outputs):
        ## decode input: [B,1]
        ## encoder_outputs: [B,seq_len,encode_dim]
        embedd = self.embedding(decoder_input).squeeze(1) ##[B,encode_dim]
        context, weights = self.attention.forward(embedd,encoder_outputs) ## [B, 1,encode_dim], [B,1,seq_len]
        out = F.relu(context) ## [B,1,encode_dim]
        out = self.decoder(out)
        out = self.dropout(out)
        out = F.relu(out)
        return out, weights


class TransfModel(nn.Module):
    def __init__(self, embeddings, n_vocab, n_emotions, device,sequence_size=199):
        super(TransfModel,self).__init__()
        self.device = device
        self.sequence_size = sequence_size
        
        self.encoder_text = Encoder(n_vocab,300,device,embeddings=embeddings)
        self.encoder_emo = Encoder(n_emotions,300,device)

        self.Linear_text = nn.Linear(300,300) 
        self.Linear_emo = nn.Linear(300,300)

        ## for the Fusion layer // attention-based
        self.attention = AttentionModule(300,device)
        self.Linear_fusion = nn.Linear(300,300) 

        ## Final layers
        self.Linear_text_final = nn.Linear(300,300)  ## Decide dimensions
        self.Linear_emo_final = nn.Linear(300,300)  ## Decide dimenstions

        self.decoder_text = Decoder(300, n_vocab,device)
        self.decoder_emo = Decoder(300, n_emotions,device)
        self.dropout = nn.Dropout(0.2)

    def forward(self,text,target_emotion=None,target_text=None):
        ## text: [B,seq_len, 1]
        ## emotion: [B, 1]
        ## target_emotion: [B,seq_len,1] 
        ## target_text: [B,seq_len,1] TBD

        text = self.encoder_text(text) ## [B,seq_len,300]
        text = self.Linear_text(text)  ## [B,seq_len,300]
        text = F.relu(text)            ## [B,seq_len,300]
        
        emotion = text[:,0,:]          ## [B,300]
        decoder_inp = []
        for i in range(text.size(1)):
            context, _ = self.attention(emotion, text)  ## [B, 1, 300], [B,1,seq_len]
            context = self.Linear_fusion(context)       ## [B, 1, 300]
            context = self.dropout(context)
            context = F.relu(context)                   ## [B, 1, 300]
            decoder_inp.append(context)
            if target_emotion is None:
                emotion = self.Linear_emo(context.squeeze(1))  ## [B, 300]
                emotion = self.dropout(emotion)
                emotion = F.relu(emotion)                      ## [B, 300]

            else:
                ## target_emotion[:,i,:]: [B,1]
                emotion = self.encoder_emo(target_emotion[:,i,:]) ## [B, 300]
                emotion = self.Linear_emo(emotion)                ## [B, 300]
                emotion = self.dropout(emotion)
                emotion = F.relu(emotion)                         ## [B, 300]

        decoder_inp = torch.cat(decoder_inp, dim=1) ## [B,seq_len,300]
        if target_emotion is not None:
            out_emotion, _ = self.decoder_emo(decoder_inp,target_tensor=target_emotion.squeeze(2)) ## [B,seq_len]
        else:
            out_emotion, _ = self.decoder_emo(decoder_inp)## [B,seq_len]
        out_emotion = out_emotion.permute(0,2,1)
        
        return out_emotion

def load_model(path,embeddings,device, n_emotions, n_vocab):
    model = TransfModel(embeddings,n_vocab,n_emotions,device)
    model.load_state_dict(torch.load(path,weights_only=False,map_location=torch.device('cpu')))
    return model
