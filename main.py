from flask import Flask,request,render_template,Response
from utils.utils import generate_bow_vector,load_classes,load_model_instancia
from src.load_dataloaders import load_dls
from xai.grad_cam import GradCAM

from functions_utils import load_input_image,save_scatter_flask
from architecture_models.vqa_with_class_weights_and_layer_norm import ModeloVQA_CW_LN_BoW
from visualization.visualization_tsne import visual_tsne
from embeddings_metrics.emb_similarity import sil_score,embs_similarity

import cv2
import torch
import numpy as np


#Cargar los datos para utilizar el modelo
train_dl,valid_dl,test_dl,vocab=load_dls(batch_train=64,batch_valid=64,batch_test=64)

modelo_instancia=ModeloVQA_CW_LN_BoW(vocab_length=27,num_clases=13)
model_weights_name="3_vqa_weights_w_class_and_layer_norm" #Por defecto, el modelo que usa Layer Normalization
modelo=load_model_instancia(modelo_instancia,f"./saved_models/{model_weights_name}.pth")

clases=load_classes()


#Aplicación Flask
app=Flask(__name__)



#Endpoint principal
@app.route("/")
def main_page():
    return render_template("main_interface.html")



#Endpoint de predicción
@app.route("/predict",methods=["POST"])
def prediccion():

    #Input del usuario
    pregunta=request.form["question"]


    #Generación del BOW vector a partir del input
    bow_vector=generate_bow_vector(vocab=vocab,sentence=pregunta)


    #Procesamiento de la imagen 
    imagen=request.files["image"]
    img_final=load_input_image(image=imagen)
    
    
    #Predicción del modelo
    modelo.eval()
    with torch.no_grad():
        pred=modelo(image_tensor=img_final.to(torch.float32),question_tensor=bow_vector.to(torch.float32))
        index_class_pred=torch.argmax(pred,dim=-1).item()
        clase=clases[index_class_pred]



    #GradCAM Heatmap Imagen
    grad_cam=GradCAM(img_tensor=img_final.to(torch.float32),question_tensor=bow_vector.to(torch.float32),modelo=modelo,view_page=True)
    
    acts=grad_cam.forward_output()  #activaciones de la capa CNN final durante el forward pass (ejecutado por el hook en la layer)
    gdts,class_pred=grad_cam.backward_pass()    #gradientes de las activaciones (gradientes de los feature maps) y la clase predicha


    avrg_gradients_weights=grad_cam.weights_feature_maps()  #gradient weights (average de cada feature map)
    heatmap,heatmap_bgr=grad_cam.heatmap(avrg_gradients_weights,img_final) #recibe los weights_gradients y la imagen_rgb --> da el heatmap puro
    
    

    #Imagen conversión a 0-255 (RGB)
    img=np.array(img_final.squeeze(0).permute(2,1,0))
    img_rgb=(img*255).astype(np.uint8)
    img_bgr=cv2.cvtColor(src=img_rgb,code=cv2.COLOR_RGB2BGR)

    
    #Convertir los numpy array a bytes: imencode ESPERA imágenes en BGR
    success_buffer,buffer_heatmap=cv2.imencode(".jpg",img=heatmap_bgr.transpose(1,0,2)) #heatmap color rgb transpuesto para que se muestre adecuadamente
    success_image,img_bgr_final=cv2.imencode(".jpg",img=img_bgr)


    global buffer_heatmap_bytes
    buffer_heatmap_bytes=buffer_heatmap.tobytes()

    global final_img
    final_img=img_bgr_final.tobytes()


    return render_template("main_interface.html",
                           answer=clase,
                           loaded=True)
                           


#Endpoint que muestra el heatmap en el html
@app.route("/get_heatmap")
def obtener_heatmap():
    global buffer_heatmap_bytes

    return Response(response=buffer_heatmap_bytes,mimetype="image/jpeg")



#Endpoint que muestra la imagen en el html
@app.route("/get_image")
def obtener_image():
    global final_img

    return Response(response=final_img,mimetype="image/jpeg")



#Visualización t-sne
@app.route("/tsne", methods=["GET","POST"])
def tsne_visualization():

    pregs_text={}

    if request.method=="POST":

        #Input text al formato que aceptan las funciones para la visualización
        for clase in ["color","shape","present"]:
            raw=request.form.get(clase)
            
            if raw:
                pregs_text[clase]=[x.strip() for x in raw.split("|") if x.strip()]

        sentences=[]
        labels=[]
        order=[]

        for clase,lista in pregs_text.items():
            order.append(clase)
            sentences+=lista
            labels+=[clase]*len(lista)
        
        n=len(list(pregs_text.values())[0])  #se asume que está balanceado



        #Visualizar la gráfica
        scatter=visual_tsne(list_sentences=sentences,number_sent_per_class=n,
                            order_class_list=order,view_page=True)
        
        #Mostrar la gráfica en el html
        scatter_flask=save_scatter_flask(figure=scatter)


        #Silhoutte Score
        silhoutte_score=sil_score(list_sentences=sentences,number_sent_per_class=n,order_class_list=order) #INT


        #Intra-Inter Cluster Mean Cosine Distance
        intra_cos_sim,inter_cos_sim=embs_similarity(list_sentences=sentences,number_sent_per_class=n,order_class_list=order) #(3,1) shape cada uno --> Filas: clusters, Columna: Distancia promedio intra/inter cluster

        
        return render_template("emb_visualization.html",scatter=scatter_flask,silhoutte_score=silhoutte_score,
                               intra_cos_sim=intra_cos_sim,inter_cos_sim=inter_cos_sim,order=order)

    else:
        return render_template("emb_visualization.html")

if __name__=="__main__":
    app.run()
