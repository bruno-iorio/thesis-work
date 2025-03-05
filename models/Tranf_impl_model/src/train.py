from include.include import *
from include.include_train import *

from sklearn.metrics import f1_score

def remove_zeros(trues,preds):
    to_remove = []
    for i in range(len(trues)):
        if trues[i] == 0:
            to_remove.append(i)
    for i in to_remove[::-1]:
        trues.pop(i)
        preds.pop(i)
    return trues, preds

def find_weights_emo(loader,n_emotions):
    freq = {emo:0 for emo in range(n_emotions)}
    for batch in loader:
        target_emotions = batch['target_emotions'].tolist()
        for l in target_emotions:
            for i in l:
                freq[i] += 1
    freq = {i:1/freq[i] for i in freq}
    weights = torch.tensor([freq[i] for i in freq])
    weights = F.normalize(weights,dim=0)
    weights[0] = 0
    return weights

def find_weights_text(loader,n_vocab):
    freq = {word_index : 0 for word_index in range(n_vocab)}
    for batch in loader:
        target_texts = batch['target_texts'].tolist()
        for l in target_texts:
            for i in l:
                freq[i] += 1
    freq = {i:1/freq[i] if freq[i] !=0 else 0 for i in freq}
    weights = torch.tensor([freq[i] for i in freq])
    weights = F.normalize(weights,dim=0)
    weights[0] = 0
    return weights


def activate_gpu(force_cpu=False): # check if gpu available ; code taken from template
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available(): # for both Nvidia and AMD GPUs
            device = 'cuda'
            print('DEVICE = ', torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available(): # for mac ARM chipset
            device = 'mps'
            print('DEVICE = ', "mps" )
        else: # for cpu only
            device = 'cpu'
            print('DEVICE = ', 'CPU', "blue")
    return device

def inference(model, loader, device):
    model.eval()
    model = model.to(device)
    trues = []
    preds = []
    for batch in loader:
        batch = {
            "texts": batch['texts'].to(device),
            "emotions": batch['emotions'].to(device),
            "target_texts": batch['target_texts'].to(device),
            "target_emotions": batch['target_emotions'].to(device),
        }
        inp_text = batch['texts'].unsqueeze(-1)
        inp_emotion = torch.zeros(inp_text.size(0),dtype=torch.long).unsqueeze(-1).to(device)
        for i in range(1,inp_text.size(1)):
            pred_emotion, _ = model.forward(inp_text[:,:i,:],inp_emotion.unsqueeze(-1))
            pred_emotion = torch.argmax(pred_emotion,dim=-1) ## [B, seq_len]
            pred_emotion = pred_emotion[:,-1].unsqueeze(1) ##[B, 1]
            inp_emotion = torch.cat((inp_emotion,pred_emotion),dim=1) ## [B,seq_len+1]
        inp_emotion = inp_emotion[:,1:] ##[ B, max_size - 1]
        for i in range(batch['emotions'].size(0)):
            trues.extend(batch["target_emotions"][i,:-1].tolist())
            preds.extend(inp_emotion[i,:].tolist())
    return trues, preds 

            
def train_batch(model,optimizer,batch,device,weights_emo=None,weights_text=None):
    lossfn_emo = nn.CrossEntropyLoss(weight=weights_emo,ignore_index=0)
    lossfn_text = nn.CrossEntropyLoss(weight=weights_text,ignore_index=0)
    inp_text = batch['texts'].unsqueeze(-1)
    inp_emo = batch['emotions'].unsqueeze(-1)
    target_emotions = batch['target_emotions']
    target_texts = batch['target_texts']
    
    pred_emotions, pred_text = model.forward(inp_text, inp_emo)
    pred_text = pred_text.permute(0,2,1)
    pred_emotions = pred_emotions.permute(0,2,1)

    loss_emo = lossfn_emo(pred_emotions,target_emotions)
    loss_text = lossfn_text(pred_text,target_texts)
    loss = loss_emo*0.9 + loss_text*0.1
    loss_emo.backward()
    optimizer.step()
    return loss_emo.mean().item()

def train(model, loader, test_loader , epochs, device,n_emotion,n_vocab):
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    model = model.to(device)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    losses_to_plot = []
    weights_text = find_weights_text(loader,n_vocab).to(device)
    weights_emo = find_weights_emo(loader,n_emotion).to(device)
    for epoch in range(epochs):
        print(f'epoch: {epoch+1}/{epochs}')
        losses = []
        model.train()
        optimizer.zero_grad()
        for it,batch in enumerate(loader):
            model.zero_grad()
            optimizer.zero_grad()
            batch = {
            "texts": batch['texts'].to(device),
            "emotions": batch['emotions'].to(device),
            "target_texts": batch['target_texts'].to(device),
            "target_emotions": batch['target_emotions'].to(device),
            }
            loss = train_batch(model,optimizer,batch,device,weights_emo=weights_emo,weights_text=weights_text)
            losses.append(loss)
        trues, preds = inference(model, test_loader, device)
        trues, preds = remove_zeros(trues, preds)
        f1 = f1_score(trues,preds,average='macro')
        losses_to_plot.append(sum(losses)/len(losses))
        print(f'loss: {losses_to_plot[-1]}, f1_score: {f1}')
    return losses_to_plot

    ## Maybe add scheduler after 
