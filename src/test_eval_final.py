import torch
import numpy as np
import matplotlib.pyplot as plt

from architecture_models.vqa_with_out_class_weights import ModeloVQAWith_outCW_BoW
from architecture_models.vqa_with_class_weights_and_layer_norm import ModeloVQA_CW_LN_BoW
from src.load_dataloaders import load_dls

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score


#Cargar los 3 modelos finales
def final_models_eval():
    
    #Modelo SIN class weight
    modelo_no_cw=ModeloVQAWith_outCW_BoW(vocab_length=27,num_clases=13)
    weights_no_cw=torch.load("./saved_models/1_vqa_without_class_weights_.pth",weights_only=True)
    modelo_no_cw.load_state_dict(weights_no_cw)
    
    
    #Modelo CON class weight
    modelo_si_cw=ModeloVQAWith_outCW_BoW(vocab_length=27,num_clases=13)
    weights_si_cw=torch.load("./saved_models/2_vqa_weights_w_class.pth",weights_only=True)
    modelo_si_cw.load_state_dict(weights_si_cw)


    #Modelo CON class weight y Layer Norm
    modelo_si_cw_ln=ModeloVQA_CW_LN_BoW(vocab_length=27,num_clases=13)
    weights_si_cw_ln=torch.load("./saved_models/3_vqa_weights_w_class_and_layer_norm.pth",weights_only=True)
    modelo_si_cw_ln.load_state_dict(weights_si_cw_ln)


    return modelo_no_cw,modelo_si_cw,modelo_si_cw_ln



#Función de evaluación y gráficas
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



#======#



#Modelos
modelo_no_cw,modelo_si_cw,modelo_si_cw_ln=final_models_eval()

#Data
train_dl,valid_dl,test_dl,vocab=load_dls(batch_train=64,batch_valid=64,batch_test=64)


#Evaluación
print(f"Modelo SIN class weights")
complete_evaluation(model=modelo_no_cw,data_eval=test_dl)

print("====")

print(f"Modelo CON class weights")
complete_evaluation(model=modelo_si_cw,data_eval=test_dl)

print("====")

print(f"Modelo CON class weights y Layer Norm")
complete_evaluation(model=modelo_si_cw_ln,data_eval=test_dl)
