## train model for EmoryNLP

from src.model import *
from src.preprocess import * 
from src.train import *
from src.dataset import *

k = 50000
SAVEPATH = "model/nn_hidden_model.pth"

def emory():
	print("Start preprocessing")
	## Data preprocessing
	tok = get_tokenizer()
	encoder_model = get_encoder()
	pretrained_embeddings,embeddings_vectors, stoi, itos = get_embeddings(encoder_model)
		## Parse dataset:
	X_train_emory, Y_train_emory, X_test_emory, Y_test_emory, X_val_emory, Y_val_emory = parse_emory()
	pretrained_embeddings, stoi_emory, itos_emory = get_topk(X_train_emory,k,tok,stoi,pretrained_embeddings,embeddings_vectors)
		## Preprocess train data:
	X_train_emory, Y_train_emory, dec_train_emory = preprocess_data(X_train_emory,Y_train_emory,stoi_emory,tok)
	Y_train_emory, lookup_emory = change_Y(Y_train_emory)
	dec_train_emory, dec_lookup_emory = change_Y(dec_train_emory)
	X_train_emory,X_train_target_emory, Y_train_emory, Y_train_target_emory,dec_train_emory ,dec_train_target_emory = get_target(X_train_emory,Y_train_emory,dec_train_emory)
		## Preprocess test data:
	X_test_emory, Y_test_emory, dec_test_emory = preprocess_data(X_test_emory,Y_test_emory,stoi_emory,tok)
	Y_test_emory, _ = change_Y(Y_test_emory,lookup_emory)
	dec_test_emory, _ = change_Y(dec_test_emory,dec_lookup_emory)
	X_test_emory,X_test_target_emory, Y_test_emory, Y_test_target_emory, dec_test_emory, dec_test_target_emory = get_target(X_test_emory,Y_test_emory,dec_test_emory)

	# Preprocess validation data:
	X_val_emory, Y_val_emory, dec_val_emory = preprocess_data(X_val_emory,Y_val_emory,stoi_emory,tok)
	Y_val_emory, _ = change_Y(Y_val_emory,lookup_emory)
	dec_val_emory, _ = change_Y(dec_val_emory,dec_lookup_emory)
	X_val_emory, X_val_target_emory, Y_val_emory, Y_val_target_emory, dec_val_emory, dec_val_target_emory = get_target(X_val_emory,Y_val_emory,dec_val_emory)

	print("Creating datasets")
	## Create database
	train_data_emory = CustomedDataset(X_train_emory,Y_train_emory,dec_train_emory,X_train_target_emory,Y_train_target_emory,dec_train_target_emory)
	test_data_emory = CustomedDataset(X_test_emory,Y_test_emory,dec_test_emory,X_test_target_emory,Y_test_target_emory,dec_test_target_emory)
	val_data_emory = CustomedDataset(X_val_emory,Y_val_emory,dec_val_emory,X_val_target_emory,Y_val_target_emory,dec_val_target_emory)
	print("Creating dataloaders")
	## Create dataloader

	batch_size = 5 
	train_loader_emory = DataLoader(train_data_emory, batch_size=batch_size,shuffle = True)
	test_loader_emory = DataLoader(test_data_emory, batch_size=batch_size,shuffle = True)
	val_loader_emory = DataLoader(val_data_emory, batch_size=batch_size, shuffle = True)
	x = 0

	print("Creating model")
## Create model and add tweeks
	device = activate_gpu()
	print(device)	
	
	emotion_dim = 200
	n_emotions = len(lookup_emory) 
	n_words = len(stoi_emory)
	model = SimpleModel(pretrained_embeddings,emotion_dim,n_emotions,n_words)
	print("n_words is ", n_words)
	print('n_emotions is', n_emotions)
	print("training")
	## train
	epochs = 10
	n_total = len(lookup_emory)
	print("n_total is", n_total)
	losses = train(model,train_loader_emory, epochs,device,n_total,batch_size)
	
	print(torch.cuda.memory_summary(device=torch.device('cuda')))
	torch.save(model.state_dict(), SAVEPATH)
	return 0

def test():
	tok = get_tokenizer()
	encoder_model = get_encoder()
	pretrained_embeddings,embeddings_vectors, stoi, itos = get_embeddings(encoder_model)

	## Parse dataset:
	X_train_emory, Y_train_emory, X_test_emory, Y_test_emory, X_val_emory, Y_val_emory = parse_emory()
	pretrained_embeddings, stoi_emory, itos_emory = get_topk(X_train_emory,k,tok,stoi,pretrained_embeddings,embeddings_vectors)

## Preprocess train data:
	X_train_emory, Y_train_emory, dec_train_emory = preprocess_data(X_train_emory,Y_train_emory,stoi_emory,tok)
	Y_train_emory, lookup_emory = change_Y(Y_train_emory)
	dec_train_emory, dec_lookup_emory = change_Y(dec_train_emory)
	X_train_emory,X_train_target_emory, Y_train_emory, Y_train_target_emory,dec_train_emory ,dec_train_target_emory = get_target(X_train_emory,Y_train_emory,dec_train_emory)

## Preprocess test data:
	X_test_emory, Y_test_emory, dec_test_emory = preprocess_data(X_test_emory,Y_test_emory,stoi_emory,tok)
	Y_test_emory, _ = change_Y(Y_test_emory,lookup_emory)
	dec_test_emory, _ = change_Y(dec_test_emory,dec_lookup_emory)
	X_test_emory,X_test_target_emory, Y_test_emory, Y_test_target_emory, dec_test_emory, dec_test_target_emory = get_target(X_test_emory,Y_test_emory,dec_test_emory)
	
# Preprocess validation data:
	X_val_emory, Y_val_emory, dec_val_emory = preprocess_data(X_val_emory,Y_val_emory,stoi_emory,tok)
	Y_val_emory, _ = change_Y(Y_val_emory,lookup_emory)
	dec_val_emory, _ = change_Y(dec_val_emory,dec_lookup_emory)
	X_val_emory, X_val_target_emory, Y_val_emory, Y_val_target_emory, dec_val_emory, dec_val_target_emory = get_target(X_val_emory,Y_val_emory,dec_val_emory)
	print(X_train_emory[:5])
	print(X_val_emory[:5])
	print(X_test_emory[:5])
	assert(len(X_train_emory) == len(Y_train_emory))
	print(Y_train_emory[:2])
	print(len(Y_train_emory[0]))

emory()
