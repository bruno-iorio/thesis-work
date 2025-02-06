import torch
import gc

device = 'cuda'
print(torch.cuda.memory_summary(device=torch.device('cuda')))
def f():	
	x = torch.randn(1024,1024,device=device)
	print(torch.cuda.memory_summary(device=torch.device('cuda')))
def g():
	f()
	#gc.collect()
	#torch.cuda.empty_cache()
	print(torch.cuda.memory_summary(device=torch.device('cuda')))
g()
#gc.collect()
#torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=torch.device('cuda')))



