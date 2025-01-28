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

## train functions

def train_batch(model, batch, optimizer,lr,device): ## train for each batch (set to size 1 in this case)
  loss = 0
  loss_fn = nn.CrossEntropyLoss()
  hidden = torch.zeros(80, requires_grad=True).to(device)
  optimizer.zero_grad()
  inp_emotion = batch['emotions'][0]
  losses = []
  for t in range(len(batch['texts'])):
    inp_emotion = batch['emotions'][t]
    inp_token = batch['texts'][t]
    target_token = batch['target_texts'][t]
    target_emotion = batch['target_emotions'][t]

    pt, pe, hidden = model.forward(inp_token,inp_emotion,hidden)
    if target_token.item() not in [0,1]:
      loss1 = loss_fn(pe,target_emotion) 
      loss2 = loss_fn(pt,target_token -2)
      losses.append(loss1.item())
      loss += loss1 + loss2
    else:
      loss1 = loss_fn(pe,target_emotion)
      losses.append(loss1.item())
      loss += loss1
  loss.backward()
  optimizer.step()
  return sum(losses)/len(losses) ## average

def train(model, train_loader, epochs, device):
  lr = 0.01 ## impltement this later (vary the learning rate)
  optimizer = optim.Adam(model.parameters())
  loss_fn = nn.CrossEntropyLoss()
  model.train()
  model = model.to(device)
  loss_to_plot = []
  for epoch in range(epochs):
    losses = []
    print(f"Epoch {epoch+1}/{epochs}")
    for it, batch in tqdm(enumerate(train_loader),total = train_loader.__len__()):
      batch = {'texts': torch.from_numpy(batch['texts']).to(device),
               'emotions': torch.from_numpy(batch['emotions']).to(device),
               'target_texts': torch.from_numpy(batch['target_texts']).view(-1).to(device), ## reshape is necessary to compare predictions
               'target_emotions': torch.from_numpy(batch['target_emotions']).view(-1).to(device)
      }
      if lr % 20 == 0:
        lr /= 10
      losses.append(train_batch(model, batch, optimizer,lr,device))
    loss_to_plot.append(sum(losses)/len(losses))
    print(f"loss: ",loss_to_plot[-1])
  return loss_to_plot
