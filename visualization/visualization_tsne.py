import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


from utils.utils import generate_bow_vector,load_model
from src.load_dataloaders import load_dls 

from architecture_models.vqa_with_out_class_weights import ModeloVQAWith_outCW_BoW
from architecture_models.vqa_with_class_weights_and_layer_norm import ModeloVQA_CW_LN_BoW

#Data y Vocab
train_dl,valid_dl,test_dl,vocab=load_dls(batch_train=64,batch_valid=64,batch_test=64)

modelo=ModeloVQA_CW_LN_BoW(vocab_length=27,num_clases=13)
model_weights_name="3_vqa_weights_w_class_and_layer_norm"   #Por defecto el modelo que usa Layer Normalization
weights=torch.load(f"./saved_models/{model_weights_name}.pth",weights_only=True)
modelo.load_state_dict(weights)


#Generación de los array embeddings, listos para t-SNE
def embeddings_array_gen(list_sentences):
    
    #Bow tensor
    list_bow_vectors=[generate_bow_vector(vocab=vocab,sentence=sentence) for sentence in list_sentences]
    tensor_bow_vectors=torch.stack(list_bow_vectors,dim=1).squeeze(0).to(torch.float32)
    
    #Linear embedding layers
    q_l1,q_l2,q_l3,relu=modelo.q_l1,modelo.q_l2,modelo.q_l3,modelo.relu

    modelo.eval()
    with torch.no_grad():
        res_1=relu(q_l1(tensor_bow_vectors.to(torch.float32)))
        res_2=relu(q_l2(res_1))  #relu(q_l2(res_1)), q_l2(res_1)
        res_3=q_l3(res_2)

    #Array de preds
    emb_array=res_3.numpy()
    
    return emb_array



#Algoritmo t-SNE --> t-SNE is only used for visualization and not for prediction.
def t_sne(list_sentences):
    
    #Obtención del embedding array
    emb_array=embeddings_array_gen(list_sentences)
    print(f"emb array: {emb_array.shape}")
    
    #t-SNE
    emb_array_red=TSNE(n_components=2,perplexity=3).fit_transform(X=emb_array)

    print(f"emb_array_red: {emb_array_red.shape}")


    return emb_array,emb_array_red



#Visualización t-SNE
def visual_tsne(list_sentences,number_sent_per_class,order_class_list,view_page:bool):
    
    #Labels
    labels_list_init=[]
    for clase in order_class_list:
        labels_list_init+=[clase]*number_sent_per_class

    print(labels_list_init)


    #Emb arrays
    emb_array_org,emb_array_red=t_sne(list_sentences)


    #Silouhette Score
    si_score=silhouette_score(X=emb_array_org,labels=labels_list_init,metric="euclidean")
    print(f"si_score: {si_score}")



    #Dataframe para el gráfico
    df_tsne=pd.DataFrame({
        "x":emb_array_red[:,0],
        "y":emb_array_red[:,1],
        "Categoría":labels_list_init,
        "Pregunta":list_sentences
    })


    #Gráfico
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    scatter=sns.scatterplot(
        data=df_tsne, 
        x='x', 
        y='y', 
        hue='Categoría',   #Agrupar por categoría
        style='Categoría', #Cambia la forma del punto en base a categorías
        s=200,             #Tamaño de los puntos
        palette="viridis"  #Paleta de colores profesional
    )

    print(f"scatter: {type(scatter)}, what is: {scatter}")

    if not view_page:
        plt.title("Visualización t-SNE de Intenciones de Preguntas", fontsize=15)
        plt.show()


    #Se guarda la imagen para visualizarlo 
    figure=scatter.get_figure()
    
    
    return figure



#Preguntas: WHAT color, shape, present
pregs_text=["what color is the rectangle?","what color is the triangle?","what is the color of the circle?","what is the color of the shape?",
            "what shape is present?","what shape does the image contain?","what shape is in the image?","what is the gray shape?",
            "is a rectangle present?","is there not a circle?","is there a triangle?","is there not a green shape?"]

#scatter=visual_tsne(pregs_text,4,["color","shape","present"],view_page=False) #Uncomment para visualizarlo de manera local

