from utils.utils import load_model
from src.load_dataloaders import load_dls

import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score


def complete_evaluation(model,data_eval):
    model.eval()
    y_preds,y_real=[],[]

    for (x_batch,y_batch) in data_eval:
        x_bow_batch,x_img_batch=x_batch[0].squeeze(1).to(torch.float32),x_batch[1]
        
        with torch.no_grad():
            preds=model(x_img_batch,x_bow_batch)
            class_preds=torch.argmax(preds,dim=-1)
            
            y_preds.append(class_preds),y_real.append(y_batch.squeeze(1))

    y_preds_flattened=np.concatenate(y_preds)
    y_real_flattened=np.concatenate(y_real)
    
    #Métricas
    #Confusion matrix:
    c_m=confusion_matrix(y_true=y_real_flattened,y_pred=y_preds_flattened)
    c_m_d=ConfusionMatrixDisplay(confusion_matrix=c_m)
    c_m_d.plot()
    plt.show()


    #Report
    print(classification_report(y_true=y_real_flattened,y_pred=y_preds_flattened,zero_division=0))
    

    #Accuracy
    print(f"Accuracy: {accuracy_score(y_real_flattened,y_preds_flattened)}")
    



#Cargar los datos
dl_train,dl_valid,_,vocab=load_dls(batch_train=64,batch_valid=64,batch_test=64)

#Modelo
model_weights="model_weights" #nombre del archivo que contiene los pesos del modelo
modelo=load_model(path_weights_model=f"./saved_models/{model_weights}.pth")


complete_evaluation(modelo,dl_valid)
