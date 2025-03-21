
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
SAVEPATH = "model/transf_model1.pth"
SAVELOOKUP = "data/lookup.json"
SAVESTOI = "data/stoi.json"
LOADPATH = "model/transf_model.pth"


def emory():
    print("Start preprocessing")
    ## Data preprocessing
    tok = get_tokenizer()
    encoder_model = get_encoder()
    pretrained_embeddings, embeddings_vectors, stoi_emory, itos_emory = get_embeddings(encoder_model)

    ## Parse dataset:
    X_train_emory, Y_train_emory, X_test_emory, Y_test_emory, X_val_emory, Y_val_emory = parse_emory()
    ## Preprocess train data:
    X_train_emory, Y_train_emory  = preprocess_data(X_train_emory,Y_train_emory,stoi_emory,tok)
    Y_train_emory, lookup_emory = label_to_index(Y_train_emory)
    X_train_emory,X_train_target_emory, Y_train_emory, Y_train_target_emory = get_target(X_train_emory,Y_train_emory)

    ## Preprocess test data:
    X_test_emory, Y_test_emory= preprocess_data(X_test_emory,Y_test_emory,stoi_emory,tok)
    Y_test_emory, _ = label_to_index(Y_test_emory,lookup_emory)
    X_test_emory,X_test_target_emory, Y_test_emory, Y_test_target_emory =get_target(X_test_emory,Y_test_emory)

    ## Preprocess validation data:
    X_val_emory, Y_val_emory = preprocess_data(X_val_emory,Y_val_emory,stoi_emory,tok)
    Y_val_emory, _ = label_to_index(Y_val_emory,lookup_emory)
    X_val_emory, X_val_target_emory, Y_val_emory, Y_val_target_emory = get_target(X_val_emory,Y_val_emory)

    ## Datasets and dataloaders:
    train_dataset_emory = CustomedDataset(X_train_emory,Y_train_emory,X_train_target_emory,Y_train_target_emory) 
    test_dataset_emory = CustomedDataset(X_test_emory,Y_test_emory,X_test_target_emory,Y_test_target_emory) 
    val_dataset_emory = CustomedDataset(X_val_emory,Y_val_emory,X_val_target_emory,Y_val_target_emory) 

    batch_size = 8
    train_loader_emory = DataLoader(train_dataset_emory,batch_size=batch_size,shuffle=True)
    test_loader_emory = DataLoader(test_dataset_emory,batch_size=batch_size,shuffle=True)
    val_loader_emory = DataLoader(val_dataset_emory,batch_size=batch_size,shuffle=True)

    device = activate_gpu()
    n_vocab = len(stoi_emory)
    n_emotion = len(lookup_emory)
    model = AttentionModel(pretrained_embeddings,300,50,n_vocab,n_emotion,device)
    
    epochs = 50
    
    losses = train(model,train_loader_emory,test_loader_emory,epochs,device,n_emotion,n_vocab)
    uplook = {a:b for (b,a) in lookup_emory.items()}
    trues, preds = inference(model,val_loader_emory,device)
    trues, preds = remove_zeros(trues,preds)
    
    trues = [uplook[i] for i in trues]
    preds = [uplook[i] for i in preds]

    print(classification_report(trues, preds))
    print(matthews_corrcoef(trues,preds))
    print(confusion_matrix(trues, preds))
    return 0


###### For DEBUG purposes ######
def test_model():
    print('aa')
    encoder_model = get_encoder()
    pretrained_embeddings, embeddings_vectors, stoi_emory, itos_emory = get_embeddings(encoder_model)
    device = activate_gpu()
    n_vocab = 20
    n_emotion = 20  
    print('b')
    model = AttentionModel(pretrained_embeddings,300,50,n_vocab,n_emotion,device)
    model.eval()
    inp_text = torch.ones(4,199,1,dtype=torch.long)
    inp_emo = torch.ones(4,199,1,dtype=torch.long)
    print('hereh')
    out, other = model.forward(inp_text,inp_emo)
    print(out)
    print(out.size())

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

#test_model()
emory()


