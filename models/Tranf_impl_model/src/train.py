from include.include import *
from include.include_train import *

from sklearn.metrics import f1_score

def find_weights(loader,n_emotions):
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
        pred_emotion, _ = model.forward(inp_text)
        pred_emotion = torch.argmax(pred_emotion,dim=-1)
        for i in range(batch['emotions'].size(0)):
            trues.extend(batch["target_emotions"][i,:].tolist())
            preds.extend(pred_emotion[i,:].tolist())
    return trues, preds 

            
def train_batch(model,optimizer,batch,device,weights=None):
    lossfn_emo = nn.CrossEntropyLoss(weight=weights)
    lossfn_text = nn.CrossEntropyLoss()
    inp_text = batch['texts'].unsqueeze(-1)
    target_emotions = batch['target_emotions']
    target_texts = batch['target_texts']
    pred_emotions, pred_text = model.forward(inp_text, target_emotions=target_emotions.unsqueeze(-1))
    pred_emotions = pred_emotions.permute(0,2,1)
    pred_text = pred_text.permute(0,2,1)
    loss_emo = lossfn_emo(pred_emotions,target_emotions)
    loss_text = lossfn_text(pred_text,target_texts)
    loss = loss_emo*0.8 + loss_text*0.2 
    loss.backward()
    optimizer.step()
    return loss_emo.mean().item()

def train(model, loader, test_loader , epochs, device,n_emotion):
    optimizer = optim.Adam(model.parameters(),lr = 0.00001)
    model = model.to(device)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    losses_to_plot = []
    weights = find_weights(loader,n_emotion)
    for epoch in range(epochs):
        print(f'epoch: {epoch}/{epochs}')
        losses = []
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        for it,batch in enumerate(loader):
            print(f'itteration: {it}/{loader.__len__()}')
            batch = {
            "texts": batch['texts'].to(device),
            "emotions": batch['emotions'].to(device),
            "target_texts": batch['target_texts'].to(device),
            "target_emotions": batch['target_emotions'].to(device),
            }
            loss = train_batch(model,optimizer,batch,device,weights=weights)
            losses.append(loss)
        trues, preds = inference(model,test_loader,device)
        f1 = f1_score(trues,preds,average='macro')
        losses_to_plot.append(sum(losses)/len(losses))
        print(f'loss: {losses_to_plot[-1]}, f1_score: {f1}')
    return losses_to_plot

    ## Maybe add scheduler after 
