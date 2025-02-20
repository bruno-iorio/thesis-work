from include.include import *
from include.include_train import *

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
        pred_emotion = model.forward(inp_text)
        pred_emotion = torch.argmax(pred_emotion,dim=1)
        for i in range(batch['emotions'].size(0)):
            trues.extend(batch["target_emotions"][i,:].tolist())
            preds.extend(pred_emotion[i,:].tolist())
    return trues, preds 

            
def train_batch(model,optimizer,batch,device,check=False):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    inp_text = batch['texts'].unsqueeze(-1)
    target_emotions = batch['target_emotions']
    pred_emotions = model.forward(inp_text)#,target_emotion=target_emotions.unsqueeze(-1))
    loss = loss_fn(pred_emotions,target_emotions)
    loss.backward()
    optimizer.step()
    return loss.mean().item()

def train(model, loader, epochs, device):
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    model = model.to(device)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    losses_to_plot = []
    for epoch in range(epochs):
        losses = []
        print(f'epoch {epoch+1}/{epochs}:')
        model.zero_grad()
        optimizer.zero_grad()
        for batch in loader:
            batch = {
            "texts": batch['texts'].to(device),
            "emotions": batch['emotions'].to(device),
            "target_texts": batch['target_texts'].to(device),
            "target_emotions": batch['target_emotions'].to(device),
            }
            loss = train_batch(model,optimizer,batch,device)
            losses.append(loss)
        losses_to_plot.append(sum(losses)/len(losses))
        print(f'loss: {losses_to_plot[-1]}')
    return losses_to_plot

    ## Maybe add scheduler after 
