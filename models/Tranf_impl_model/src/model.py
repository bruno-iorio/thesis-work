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
    def __init__(self,embeddings,encoded_dim, encoded_emo, n_vocab,n_emotion,device):
        super(AttentionModel,self).__init__()
        self.attention = AdditiveAttention(encoded_dim)
        self.multi_head_attention1 = nn.MultiheadAttention(embed_dim=encoded_dim,num_heads=20,dropout = 0.2, batch_first=True)
        self.multi_head_attention2 = nn.MultiheadAttention(embed_dim=encoded_dim,num_heads=20,dropout = 0.2, batch_first=True)
        self.multi_head_attention_masked = nn.MultiheadAttention(embed_dim=encoded_dim,num_heads=20,dropout = 0.2, batch_first=True)
        
        self.encoder_emo = nn.Embedding(n_emotion,encoded_dim)
        self.encoder_text = nn.Embedding.from_pretrained(embeddings, freeze=True)
        #self.encoder_text = nn.Embedding(n_vocab,encoded_dim)
        

        self.Linear_text1 = nn.Linear(encoded_dim,encoded_dim)
        self.Linear_text2 = nn.Linear(encoded_dim, encoded_dim)
        self.Linear_emo1 = nn.Linear(encoded_dim,encoded_dim)
        ## Final layers

        self.Final_text = nn.Linear(encoded_dim,n_vocab)
        self.Final_emo1 = nn.Linear(encoded_dim,50)
        self.Final_emo2 = nn.Linear(50,n_emotion)
		
		## extra
        self.dropout = nn.Dropout(0.2)
        self.device = device
    def forward(self,input_text,input_emotion):
        ## input_text : [B, seq_len]
        ## input_emotion: [B, seq_len] 
       
		## input channels: 
        text = self.encoder_text(input_text).squeeze(-2) ## [B, seq_len, embed_dim]
        text = self.Linear_text1(text)                   ## [B, seq_len, embed_dim]
        text = self.dropout(text)      					 ## [B, seq_len, embed_dim]
        text = F.relu(text)                              ## [B, seq_len, embed_dim]
        
        emo = self.encoder_emo(input_emotion).squeeze(-2) ## [B, seq_len, embed_dim]
        emo = self.Linear_emo1(emo)                       ## [B, seq_len, embed_dim]
        emo = self.dropout(emo)							  ## [B, seq_len, embed_dim]
        emo = F.relu(emo) 								  ## [B, seq_len, embed_dim]
		
		## self attention on text because...
        # text_out, _ = self.multi_head_attention1(text,text,text)   ## [B, seq_len, embed_dim]
        #text_out = self.Linear_text2(text_out) 					 ## [B, seq_len, embed_dim]
        #text_out = self.dropout(text_out) 						     ## [B, seq_len, embed_dim]
        #text_out = F.relu(text_out) 							     ## [B, seq_len, embed_dim]

		## fusion network
        # fus, _ = self.multi_head_attention2(text_out, emo, emo) 								            ## [B, seq_len, embed_dim]
        # mask = torch.triu(torch.full((text.size(1),text.size(1)),float('-inf')),diagonal=1).to(self.device) ## [B, seq_len, embed_dim]
        fus, _ = self.multi_head_attention_masked(text, text, text) #, attn_mask=mask) 		                    ## [B, seq_len, embed_dim]
		
		## Final text layer
        text_out = self.Final_text(fus)             ## [B,seq_len,n_vocab]
        text_out = self.dropout(text_out) 			## [B,seq_len,n_vocab]
        text_out = F.relu(text_out) 			 	## [B,seq_len,n_vocab]
		## Final emotion layers
        emo_out = self.Final_emo1(fus)            ## [B,seq_len,embed_dim]
        emo_out = self.dropout(emo_out)           ## [B,seq_len,embed_dim]
        emo_out = F.relu(emo_out) 				  ## [B,seq_len,embed_dim]
        
        emo_out = self.Final_emo2(emo_out)        ## [B,seq_len,n_emotion]
        emo_out = self.dropout(emo_out)			  ## [B,seq_len,n_emotion]
        emo_out = F.relu(emo_out) 				  ## [B,seq_len,n_emotion]
		
        return emo_out, text_out
        


def load_model(path,embeddings,device, n_emotions, n_vocab):
    model = TransfModel(embeddings,n_vocab,n_emotions,device)
    model.load_state_dict(torch.load(path,weights_only=False,map_location=torch.device('cpu')))

