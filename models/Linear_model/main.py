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
    X_train_emory, Y_train_emory, X_test_emory, Y_test_emory, X_val_emory, Y_val_emory, lookup = parse_emory()
    pretrained_embeddings, stoi_emory, itos_emory = get_topk(X_train_emory,k,tok,stoi,pretrained_embeddings,embeddings_vectors)
    
    X_train_emory, Y_train_emory = preprocess_data(X_train_emory,Y_train_emory,stoi_emory,tok)
    X_train_emory,X_train_target_emory, Y_train_emory, Y_train_target_emory = get_target(X_train_emory,Y_train_emory)
    X_test_emory, Y_test_emory = preprocess_data(X_test_emory,Y_test_emory,stoi_emory,tok)
    X_test_emory,X_test_target_emory, Y_test_emory, Y_test_target_emory = get_target(X_test_emory,Y_test_emory)
    X_val_emory, Y_val_emory = preprocess_data(X_val_emory,Y_val_emory,stoi_emory,tok)
    X_val_emory, X_val_target_emory, Y_val_emory, Y_val_target_emory = get_target(X_val_emory,Y_val_emory)

    print("Creating datasets")
    ## Create database
    train_data_emory = EmorynlpDataset(X_train_emory,Y_train_emory,X_train_target_emory,Y_train_target_emory)
    test_data_emory = EmorynlpDataset(X_test_emory,Y_test_emory,X_test_target_emory,Y_test_target_emory)
    val_data_emory = EmorynlpDataset(X_val_emory,Y_val_emory,X_val_target_emory,Y_val_target_emory)

    print("Creating dataloaders")
    ## Create dataloader
    
    batch_size = 1
    train_loader_emory = DataLoader(train_data_emory, batch_size=batch_size,shuffle = True)
    test_loader_emory = DataLoader(test_data_emory, batch_size=batch_size,shuffle = True)
    val_loader_emory = DataLoader(val_data_emory, batch_size=batch_size, shuffle = True)

    print("Creating model")
    ## Create model and add tweeks
    device = activate_gpu()
    emotion_dim = 30
    n_emotions = 8 
    n_words = len(stoi_emory)

    model = SimpleModel(pretrained_embeddings,emotion_dim,n_emotions,n_words)
    print("training")
    ## train
    epochs = 10
    losses = train(model,train_data_emory, epochs,device)
    print(losses)
    
    ## saving model:
    torch.save(model.state_dict(), SAVEPATH)
    return 0
emory()






 








