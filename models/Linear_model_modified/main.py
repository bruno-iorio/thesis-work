
##############################################################
##############################################################
###### Main file of model training ###########################
###### Author: Bruno Iorio, École Polytechnique ##############
###### Supervisor: Gaël Guibon ###############################
###### Project: Bachelor Thesis ##############################
###### Date: February, 2025 ##################################
##############################################################
##############################################################

from src.model import *
from src.preprocess import * 
from src.train import *
from src.dataset import *
########################
### Global Variables ###
########################

k = 50000
SAVEPATH = "model/modified_nn_model3.pth"
SAVELOOKUP = "data/lookup.json"
SAVESTOI = "data/stoi.json"
LOADPATH = "model/modified_nn_model2.pth"


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
    X_train_emory, Y_train_emory = preprocess_data(X_train_emory,Y_train_emory,stoi_emory,tok)
    Y_train_emory, lookup_emory = change_Y(Y_train_emory)
    X_train_emory,X_train_previous_emory ,X_train_target_emory, Y_train_emory,Y_train_previous_emory, Y_train_target_emory = get_target(X_train_emory,Y_train_emory)
    ## Preprocess test data:
    X_test_emory, Y_test_emory = preprocess_data(X_test_emory,Y_test_emory,stoi_emory,tok)
    Y_test_emory, _ = change_Y(Y_test_emory,lookup_emory)
    X_test_emory, X_test_previous_emory, X_test_target_emory, Y_test_emory, Y_test_previous_emory, Y_test_target_emory = get_target(X_test_emory,Y_test_emory)

    ## Preprocess validation data:
    X_val_emory, Y_val_emory = preprocess_data(X_val_emory,Y_val_emory,stoi_emory,tok)
    Y_val_emory, _ = change_Y(Y_val_emory,lookup_emory)
    X_val_emory, X_val_previous_emory,X_val_target_emory, Y_val_emory,Y_val_previous_emory, Y_val_target_emory = get_target(X_val_emory,Y_val_emory)

    print("Creating datasets")
    ## Create database
    train_data_emory = CustomedDataset(X_train_emory,X_train_previous_emory,Y_train_emory,Y_train_previous_emory,X_train_target_emory,Y_train_target_emory)
    test_data_emory = CustomedDataset(X_test_emory,X_test_previous_emory,Y_test_emory,Y_test_previous_emory,X_test_target_emory,Y_test_target_emory)
    val_data_emory = CustomedDataset(X_val_emory,X_val_previous_emory,Y_val_emory,Y_val_previous_emory,X_val_target_emory,Y_val_target_emory)
    print("Creating dataloaders")
    ## Create dataloader

    batch_size = 15
    train_loader_emory = DataLoader(train_data_emory, batch_size=batch_size,shuffle = True)
    test_loader_emory = DataLoader(test_data_emory, batch_size=batch_size,shuffle = True)
    val_loader_emory = DataLoader(val_data_emory, batch_size=batch_size, shuffle = True)

    #save_dict_json(SAVELOOKUP,lookup_emory)
    #save_dict_json(SAVESTOI,stoi_emory)

    emotion_dim = 200
    n_emotions = len(lookup_emory) 
    n_words = len(stoi_emory)

    print("Creating model")
    ## Create model and add tweeks
    device = activate_gpu()
    print(device)	


    model = SimpleModel(pretrained_embeddings,emotion_dim,n_emotions,n_words)
    #model = load_model(LOADPATH,pretrained_embeddings,emotion_dim,n_emotions,n_words)
    print("training")
    ## train
    epochs = 10
    n_total = len(lookup_emory)
    
    #train_one_iteration(model,train_loader_emory,device,n_total,batch_size)
    losses = train(model,train_loader_emory, epochs,device,n_total,batch_size)
    torch.save(model.state_dict(), SAVEPATH)

    loss, preds, trues, words, pred_words = inference(model,val_loader_emory,device,batch_size)
    uplook = {e:x for (x,e) in lookup_emory.items()}
    uplook[-1] = 'NE'
    itos_emory[-1] = '<pad>'
    casted_preds , _ = cast_back(words,preds,itos_emory,uplook)
    casted_trues, _ = cast_back(words,trues, itos_emory,uplook)
    
    print(confusion_matrix(casted_preds, casted_trues))
    print(classification_report(casted_trues,casted_preds))
    return 0

def test():
    tok = get_tokenizer()
    encoder_model = get_encoder()
    pretrained_embeddings,embeddings_vectors, stoi, itos = get_embeddings(encoder_model)
    ## Parse dataset:
    X_train_emory, Y_train_emory, X_test_emory, Y_test_emory, X_val_emory, Y_val_emory = parse_emory()
    pretrained_embeddings, stoi_emory, itos_emory = get_topk(X_train_emory,k,tok,stoi,pretrained_embeddings,embeddings_vectors)

    ## Preprocess train data:
    X_train_emory, Y_train_emory = preprocess_data(X_train_emory,Y_train_emory,stoi_emory,tok)
    Y_train_emory, lookup_emory = change_Y(Y_train_emory)
    X_train_emory,X_train_target_emory, X_train_previous_emory, Y_train_emory, Y_train_target_emory ,Y_train_previous_emory = get_target(X_train_emory,Y_train_emory)
    print(X_train_previous_emory[0])
    print(X_train_emory[0])
    print(len(X_train_previous_emory),len(X_train_emory))
emory()



