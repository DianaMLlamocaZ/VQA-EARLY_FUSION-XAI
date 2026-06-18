import numpy as np
from src.load_dataset import DatasetVQA
from utils.utils import load_data,split_train_val_data,create_vocab,save_losses,class_weight,set_seed
from src.models import ModeloVQA_BoW
import torch
from torch.utils.data import DataLoader


#Seed --> para comparación de modelos
set_seed()



#Cargar los datos de train y test
train_data=load_data(path_data="./data/train")
test_data=load_data(path_data="./data/test")


#División de la data de train, en train y valid
train,valid=split_train_val_data(data=train_data,train_size=0.8)


#Creación del vocabulario a partir de la data de train (evitar data leakage)
vocab=create_vocab(data_train=train)



#Creación de los datasets
ds_train=DatasetVQA(data=train,vocab_train=vocab,path="./data/train")
ds_valid=DatasetVQA(data=valid,vocab_train=vocab,path="./data/train")
ds_test=DatasetVQA(data=test_data,vocab_train=vocab,path="./data/test")


#Dataloaders
dl_train=DataLoader(ds_train,batch_size=64,shuffle=True,num_workers=0)    #64
dl_valid=DataLoader(ds_valid,batch_size=64,shuffle=False,num_workers=0)
dl_test=DataLoader(ds_test,batch_size=64,shuffle=False,num_workers=0)


#Model: El vocabulario tiene 27 palabras
vqa=ModeloVQA_BoW(vocab_length=27,num_clases=13)


#Hiperparámetros de entrenamiento
epocas=50 
learning_rate=1e-4
optim=torch.optim.Adam(params=vqa.parameters(),lr=learning_rate)

#Scheduler Learning Rate
scheduler_lr=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,mode="min",factor=0.1,patience=3)

#Weights de las clases
weight_classes=torch.tensor(class_weight(),dtype=torch.float32) 
print(f"weight classes: {weight_classes}, type: {weight_classes.dtype}")

#Loss function: Cross Entropy Loss == LogSoftMax de los logits + NLLLoss sobre eso
cel=torch.nn.CrossEntropyLoss(weight=weight_classes)



#Lista del train loss
train_loss_epoca_list=[]
val_loss_epoca_list=[]

#Early Stopping --> Solo si hay data de validación
early_stopping=5
min_loss=float("inf")

#Nombre de los pesos del modelo
name_weights="nombre_modelo"  #acá colocar el nombre del archivo que contendrá los pesos del modelo


#Bucle de entrenamiento
for epoca in range(0,epocas):
    train_loss_epoca=0


    #Entrenamiento
    vqa.train()
    for (index_train,(train_x,train_y)) in enumerate(dl_train):
        
        #Batch de Questions and Images
        qts_batch,imgs_batch=train_x[0].squeeze(1).to(torch.float32),train_x[1]
        

        #Paso los datos por el modelo: inicia el autograd de pytorch para los pesos
        y_pred_train=vqa(imgs_batch,qts_batch)
        

        #En Cross Entropy Loss, el target shape debe ser [batch_size] --> solo 1 dimensión, NO [batch_size,1]
        loss_train=cel(y_pred_train,train_y.squeeze(1)) #dim (solo 1) --> [n_samples_batch]
        

        #Clean de los gradientes para evitar acumulación
        optim.zero_grad()

        #Calculo los gradientes actuales con respecto al loss --> Mini Batch Gradient Descent
        loss_train.backward()

        #Actualizo los pesos con los gradientes obtenidos, una vez que ya han sido calculados
        optim.step()

        
        #Seguimiento/tracking del training loss
        train_loss_epoca+=loss_train.item()

        
    #Average Loss Train
    train_average_epoca=train_loss_epoca/(index_train+1)
    train_loss_epoca_list.append(train_average_epoca)
    
    

    #Validation
    vqa.eval()
    valid_loss_epoch=0
    for (index_valid,(valid_x,valid_y)) in enumerate(dl_valid):
           
        #Datos: questions e images
        qsts_valid,imgs_valid=valid_x[0].squeeze(1).to(torch.float32),valid_x[1]


        #Forward Pass
        with torch.no_grad():
            preds_valid=vqa(imgs_valid,qsts_valid)
            
        
            #Calcular el Loss
            loss_valid=cel(preds_valid,valid_y.squeeze(1))
            

        #Tracking del val loss
        valid_loss_epoch+=loss_valid.item()


    #Average del validation loss por época
    val_average_epoca=valid_loss_epoch/(index_valid+1)
    val_loss_epoca_list.append(val_average_epoca)


    #El scheduler se actualiza en cada ÉPOCA del validation loss
    scheduler_lr.step(val_average_epoca)


    print(f"Época: {epoca+1}. Train Loss: {train_average_epoca}, Valid Loss: {val_average_epoca}, Learning Rate: {optim.param_groups[0]["lr"]}")


    #Early Stopping
    if val_average_epoca<min_loss:

        weights_model_dict=vqa.state_dict()
        torch.save(weights_model_dict,f"./saved_models/{name_weights}.pth") 

        min_loss=val_average_epoca
        early_stopping_actual=0
    

    else:
        early_stopping_actual+=1
        print(f"Modelo no mejorando {early_stopping_actual} veces")

        if early_stopping_actual==early_stopping:
            break


#Guardar las listas de loss para la visualización
save_losses(train_loss_list=train_loss_epoca_list,valid_loss_list=val_loss_epoca_list)
