import torch

from xai.grad_cam import GradCAM

from utils.utils import generate_bow_vector,load_image,load_classes,load_model_instancia
from src.load_dataloaders import load_dls

from architecture_models.vqa_with_class_weights_and_layer_norm import ModeloVQA_CW_LN_BoW
from architecture_models.vqa_with_out_class_weights import ModeloVQAWith_outCW_BoW

#====#

#Cargar los datos de train, test, clases y el modelo
train_dl,valid_dl,test_dl,vocab=load_dls(batch_train=64,batch_valid=64,batch_test=64)

modelo_instancia=ModeloVQA_CW_LN_BoW(vocab_length=27,num_clases=13)
model_weights_name="nombre_del_archivo_de_los_pesos_del_modelo" 
modelo=load_model_instancia(instancia_model=modelo_instancia,path_weights=f"./saved_models/{model_weights_name}.pth")

clases=load_classes()



#Función que se encarga de generar el heatmap
def visualization_grad_cam(vocab,model,sentence,path_image):
    
    #Datos
    bow_sentence=generate_bow_vector(vocab=vocab,sentence=sentence).to(torch.float32)
    tensor_img=load_image(path_image=path_image)
    
    #Grad Cam
    gc=GradCAM(img_tensor=tensor_img,question_tensor=bow_sentence,modelo=model,view_page=False)
    
    acts=gc.forward_output()
    gdts,class_pred=gc.backward_pass()
    print(f"Clase predicha: {clases[class_pred]}")

    avrg_gradients=gc.weights_feature_maps()
    _,_=gc.heatmap(avrg_gradients,tensor_img)



#=======#


#Probar de manera local la visualización del heatmap
question="aquí_va_la_pregunta"
imagen_nombre="aquí_va_el_nombre_del_archivo_que_contiene_la_imagen"
visualization_grad_cam(vocab,modelo,question,path_image=f"./data/test/images/{imagen_nombre}")
