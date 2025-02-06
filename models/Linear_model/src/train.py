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

def get_frequency_all(batches,n_total):
    freq_dict = {i:0 for i in range(n_total)}
    for batch in batches:
        c = dict(Counter(batch['target_emotions'].view(-1).tolist()))
        for tens,freq in c.items():
            if tens!=-1:
                freq_dict[tens] += freq
    for i in freq_dict:
        if i != 0:
            if freq_dict[i] != 0:
                freq_dict[i] = 1/(freq_dict[i] + 1e-6)
        if freq_dict[i] < 0.05:
            freq_dict[i] /= 10
        if freq_dict[i] > 10:
            freq_dict[i] = 0
    freq_dict[0] = 0.0000001
    return F.normalize(torch.tensor(list(freq_dict.values()),dtype=torch.float32),dim=0)

def inference(model,loader,device,batch_size,weights=None):
    model.eval()
    model = model.to(device)
    preds = []
    trues = []
    words = []
    pred_words = []
    losses = []
    for batch in loader:
        batch = {          
						 'texts' : batch['texts'].to(device),
                      'emotions' : batch['emotions'].to(device),
                  'target_texts' : batch['target_texts'].to(device),
               'target_emotions' : batch['target_emotions'].to(device),
                           'dec' : batch['dec'].to(device)
        }
        mask = (batch['target_texts'] != -1).float()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1) if weights is None else nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=-1)
        inp_emotion = torch.tensor([0]*batch['target_texts'].size()[0]).to(device)
        hidden = torch.zeros((batch['texts'].size()[0],400)).to(device)
        loss = 0
        for t in range(batch['texts'].size()[1]):
            inp_token = batch['texts'][:, t]
            target_token = batch['target_texts'][:, t]
            target_emotion = batch['target_emotions'][:, t]
            pt , pe, hidden = model.forward(inp_token,inp_emotion,hidden)
            pred_emotion = torch.argmax(pe,1)
            pred_token = torch.argmax(pt,1)
            if target_token.size()[0] == batch_size:
                preds.append(pred_emotion)
                trues.append(target_emotion)
                words.append(target_token)
                pred_words.append(pred_token)
            if mask[:,t].sum().item() == 0:  ## if it has to stop, then append first
                continue
            loss += 0.7*loss_fn(pe, target_emotion)/batch_size
            inp_emotion = pred_emotion
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue
        losses.append(loss.mean().item())
    trues = torch.stack(trues)
    preds = torch.stack(preds)
    words = torch.stack(words)
    pred_words = torch.stack(pred_words)
    new_trues = []
    new_preds = []
    new_words = []
    new_pred_words = []
    for i in range(trues.size()[1]):
        new_preds.append(preds[:, i])
        new_trues.append(trues[:, i])
        new_words.append(words[:, i])
        new_pred_words.append(pred_words[:,i])
    pred_words = torch.stack(new_pred_words).view(-1).tolist()
    preds = torch.stack(new_preds).view(-1).tolist()
    trues = torch.stack(new_trues).view(-1).tolist()
    words = torch.stack(new_words).view(-1).tolist()
    return F.f1_score(preds,trues,average='macro'), sum(losses)/len(losses),preds,trues,words,pred_words

## train functions
def train_batch(model, batch, optimizer,lr,device,batch_size ,weights=None): ## train for each batch (set to size 1 in this case)
    loss = 0
    loss_words = nn.CrossEntropyLoss(ignore_index=-1)
    loss_emotions = nn.CrossEntropyLoss(ignore_index=-1) if weights is None else nn.CrossEntropyLoss(weight=weights,ignore_index=-1)
    hidden = torch.zeros((batch['texts'].size()[0],400), requires_grad=True).to(device)
    losses = []
    valids = (batch['target_emotions'] != -1).float()
    for t in range(55):#batch['texts'].size()[1]):
        if valids[:,t].sum().item() == 0:
            continue
        inp_emotion = batch['emotions'][:, t].to(device)
        inp_token = batch['texts'][:, t].to(device)
        target_token = batch['target_texts'][:, t].to(device)
        target_emotion = batch['target_emotions'][:, t].to(device)
        pt, pe, hidden = model.forward(inp_token,inp_emotion,hidden)
        assert(torch.isnan(pt).any() or (torch.isnan(pe).any() or torch.isnan(hidden).any()))
        loss1 = 0.7 * loss_emotions(pe, target_emotion)
        loss2 = 0.3 * loss_words(pt, target_token)
        if torch.isnan(loss1).any() or torch.isinf(loss1).any():
            continue
        if torch.isnan(loss2).any() or torch.isinf(loss2).any():
            continue
        loss += loss1 + loss2
        losses.append(loss1.mean().item())
        loss.backward(retain_graph=True) ## unholy consumption of memory
        
        del inp_emotion, inp_token, target_token
        gc.collect()
        torch.cuda.empty_cache()		
    optimizer.step()
    if len(losses) == 0:
      return 0, 0
    else:
        return sum(losses)/(len(losses)),len(losses) ## average of the loss over the emotions
def train(model, train_loader, epochs, device, n_total ,batch_size ,lr=0.01,compute_preds = False):
    optimizer = optim.SGD(model.parameters(),lr=lr)
    #scheduler = StepLR(optimizer, step_size=300), gamma=0.1) 
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    model = model.to(device)
    loss_to_plot = []
    weights = get_frequency_all(train_loader,n_total).to(device)
    for epoch in range(epochs):
        losses = []
        print(f"Epoch {epoch+1}/{epochs}")
        l = 0
        k = 0
        optimizer.zero_grad()
        model.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        if compute_preds:
            f1, loss, _, _, _, _ = inference(model, train_loader,device,batch_size,weights=weights)
            print(f'f1: {f1}, loss: {loss}')
        for it, batch in enumerate(train_loader): 
            batch = {          
                         'texts' : batch['texts'],
                      'emotions' : batch['emotions'],
                  'target_texts' : batch['target_texts'],
               'target_emotions' : batch['target_emotions'],
                           'dec' : batch['dec']
            }
            loss_, ln = train_batch(model,batch,optimizer,lr,device,batch_size,weights=weights)
            l += ln
            losses.append(loss_)
 	        ## Cleaning memory to avoid leaks:
            optimizer.zero_grad()
            model.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
        loss_to_plot.append(sum(losses)/len(losses))
        print(l)
        print(f"training loss: {loss_to_plot[-1]}")
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return loss_to_plot
