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
        batch = {        'texts' : batch['texts'].to(device),
                      'emotions' : batch['emotions'].to(device),
                  'target_texts' : batch['target_texts'].to(device),
               'target_emotions' : batch['target_emotions'].to(device),
                           'dec' : batch['dec'].to(device)
        }
        valids1 = (batch['target_texts'] != -1).float()
        valids2 = (batch['target_emotions'] != -1).float()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1) if weights is None else nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=-1)
        inp_emotion = torch.tensor([0]*batch['target_texts'].size()[0]).to(device)
        hidden = torch.zeros((batch['texts'].size()[0],400)).to(device)
        loss = 0
        for t in range(batch['texts'].size()[1]):
            inp_token = batch['texts'][:, t]
            target_token = batch['target_texts'][:, t]
            target_emotion = batch['target_emotions'][:, t]
            pt , pe, hidden = model.infer(inp_token,inp_emotion,hidden)
            pred_emotion = torch.argmax(pe,1)
            pred_token = torch.argmax(pt,1)
            if target_token.size()[0] == batch_size:
                preds.append(pred_emotion)
                trues.append(target_emotion)
                words.append(target_token)
                pred_words.append(pred_token)
            if valids1[:,t].sum().item() == 0 or valids2[:,t].sum.item() == 0:  ## if it has to stop, then append first
                continue
            loss += 0.7*loss_fn(pe, target_emotion)/batch_size
            inp_emotion = pred_emotion
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue
        losses.append(loss.mean().item())
    
	## Annoying appending part:
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
def train_batch(model, batch, optimizer,lr,device,batch_size ,weights=None,max_norm = 1.0): ## train for each batch (set to size 1 in this case)
    loss = 0
    loss_words = nn.CrossEntropyLoss(ignore_index=-1)
    loss_emotions = nn.CrossEntropyLoss(ignore_index=-1) if weights is None else nn.CrossEntropyLoss(weight=weights,ignore_index=-1)
    hidden = torch.zeros((batch['texts'].size()[0],400), requires_grad=True).to(device)
    losses = []
    valids1 = (batch['target_emotions'] != -1).float()
    valids2 = (batch['target_texts'] != -1).float()
    for t in range(50):# batch['texts'].size()[1]): ## TODO: Test if it works with 5 instead of 55, to run faster
        if valids1[:,t].sum().item() == 0 or valids2[:,t].sum().item() == 0: ## Am I using the valids/ mask ideally? Or should I change the preprocessing so that it creates a masked object? 
            continue                      ## TODO: Test this alternative with small examples to see if it works
        inp_emotion = batch['emotions'][:, t]
        inp_token = batch['texts'][:, t]
        target_token = batch['target_texts'][:, t]
        target_emotion = batch['target_emotions'][:, t]

        pt, pe, hidden = model.forward(inp_token, inp_emotion, hidden)
        loss1 = 0.7 * loss_emotions(pe, target_emotion)
        loss2 = 0.3 * loss_words(pt, target_token)
        if torch.isnan(loss1).any() or torch.isinf(loss1).any(): ## TODO: Are the nans impossible to evade? Check
            continue
        if torch.isnan(loss2).any() or torch.isinf(loss2).any():
            continue

        loss += loss1 + loss2
        losses.append(loss1.mean().item())
        loss.backward(retain_graph=True) ## unholy consumption of memory

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) ## prevent that the gradient explodes (otherwise will have nan everywhere) ## TODO: Why does it make everything so slow?
    optimizer.step()
    if len(losses) == 0:
      return 0, 0
    else:
        return sum(losses)/(len(losses)),len(losses) ## average of the loss over the emotions
def train(model, train_loader, epochs, device, n_total ,batch_size ,lr=0.01,compute_preds = False):
    max_norm = 1.0  # Define a threshold
    optimizer = optim.SGD(model.parameters(),lr=lr)
    #scheduler = StepLR(optimizer, step_size=300), gamma=0.1) 
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
        if compute_preds:
            f1, loss, _, _, _, _ = inference(model, train_loader,device,batch_size,weights=weights)
            print(f'f1: {f1}, loss: {loss}')
        for it, batch in enumerate(train_loader): 
            batch = {          
                         'texts' : batch['texts'].to(device),
                      'emotions' : batch['emotions'].to(device),
                  'target_texts' : batch['target_texts'].to(device),
               'target_emotions' : batch['target_emotions'].to(device)
                       #    'dec' : batch['dec'].to(device) ## maybe add it later
            }
            loss_, ln = train_batch(model,batch,optimizer,lr,device,batch_size,weights=weights,max_norm=max_norm)
            l += ln
            losses.append(loss_)
 	        ## Cleaning memory to avoid leaks:
            optimizer.zero_grad()
            model.zero_grad()
            #torch.cuda.empty_cache()
        loss_to_plot.append(sum(losses)/len(losses))
        torch.cuda.empty_cache()
        print(l)
        print(f"training loss: {loss_to_plot[-1]}")
    # del optimizer
    torch.cuda.empty_cache()
    return loss_to_plot


def train_one_iteration(model, train,device,n_total,batch_size,lr=0.01):
	## Debugging purposes
    model.train()
    weights = get_frequency_all(train,n_total).to(device)
    optimizer = optim.SGD(model.parameters(),lr=lr)
    max_norm = 1.0 
    for batch in train:
        batch = {          
                         'texts' : batch['texts'].to(device),
                      'emotions' : batch['emotions'].to(device),
                  'target_texts' : batch['target_texts'].to(device),
               'target_emotions' : batch['target_emotions'].to(device)
                       #   'dec' : batch['dec'].to(device) ## maybe add it later
            }
        loss_, ln = train_batch(model,batch,optimizer,lr,device,batch_size,weights=weights,max_norm=max_norm)
        break


def inference_sentence(model,loader,device, batch_size):
    model.eval()
    model = model.to(device)
    for batch in loader:
        batch = { 'texts' : batch['texts'].to(device),
                'emotions': batch['emotions'].to(device),
                'target_texts' : batch['target_texts'].to(device),
                'target_emotions' : batch['target_emotions'].to(device)
        }
        hidden = model.init_hidden(batch['emotions'].size()[0],300,device)
        inp_emotion = batch['emotions'][:,0]
        for t in range(inp_emotion.size()[1]):
            inp_text = batch['texts'][t]
            target_emotion = batch['target_emotions']
            target_texts = batch['target_texts']

            pt, pe, hidden = model.forward(inp_text,inp_emotion,hidden)
            




def train_sentence_batch(model, batch, optimizer,lr,device,batch_size): ## train for each batch (set to size 1 in this case)
    loss_emotions = nn.CrossEntropyLoss()
    loss_texts = nn.MSELoss()
    hidden = model.init_hidden(batch['texts'].size()[0],300,device)
    losses = []
    for t in batch['emotions'].size()[1]:
        inp_text = batch['texts'][t]
        inp_emotion = batch['emotions'][:,t]
        target_emotion = batch['target_emotions'][:,t]
        target_text = batch['target_texts'][t]
        pt, pe, hidden = model.forward(inp_text,inp_emotion,hidden)
        loss1 = loss_emotions(pe,target_emotion)
        loss2 = loss_texts(pt,target_texts)
        loss = loss1*0.8 + 0.2*loss2
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) ## prevent that the gradient explodes (otherwise will have nan everywhere) ## TODO: Why does it make everything so slow?
        optimizer.step()
        losses.append(loss.mean().item())
    if len(losses) == 0:
        return 0
    return sum(losses)/len(losses)
    
def train_sentence(model, train_loader, device, epochs, batch_size,lr=0.01):
    model.train()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=lr)
    max_norm = 1.0
        
    for batch in train_loader:
        batch = { 'texts' : batch['texts'].to(device),
                'emotions': batch['emotions'].to(device),
                'target_texts' : batch['target_texts'].to(device),
                'target_emotions' : batch['target_emotions'].to(device)
        }
        loss = train_sentence_batch(model,batch,optimizer,lr,device,batch_size)


    
        
        


 
