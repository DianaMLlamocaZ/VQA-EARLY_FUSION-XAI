import numpy as np
import torch

from utils.utils import generate_bow_vector
from src.load_dataloaders import load_dls
from sklearn.metrics import silhouette_score
from scipy.spatial import distance

from architecture_models.vqa_with_out_class_weights import ModeloVQAWith_outCW_BoW
from architecture_models.vqa_with_class_weights_and_layer_norm import ModeloVQA_CW_LN_BoW

#Cargar el vocabulario y el modelo
dl_train,dl_valid,dl_test,vocab=load_dls(batch_train=64,batch_valid=64,batch_test=64)

modelo=ModeloVQA_CW_LN_BoW(vocab_length=27,num_clases=13)
model_weights="3_vqa_weights_w_class_and_layer_norm" #Por defecto el modelo que usa Layer Normalization
weights=torch.load(f"./saved_models/{model_weights}.pth",weights_only=True)
modelo.load_state_dict(weights)




#Generación de los array embeddings
def embeddings_array_gen(list_sentences):
    
    #Bow tensor
    list_bow_vectors=[generate_bow_vector(vocab=vocab,sentence=sentence) for sentence in list_sentences]
    tensor_bow_vectors=torch.stack(list_bow_vectors,dim=1).squeeze(0).to(torch.float32)
    
    #Linear embedding layers
    q_l1,q_l2,q_l3,relu=modelo.q_l1,modelo.q_l2,modelo.q_l3,modelo.relu

    modelo.eval()
    with torch.no_grad():
        res_1=relu(q_l1(tensor_bow_vectors.to(torch.float32)))
        res_2=relu(q_l2(res_1))
        res_3=q_l3(res_2)

    #Array de preds
    emb_array=res_3.numpy()
    
    return emb_array



#Calcular el Silhoutte Score
def sil_score(list_sentences,number_sent_per_class,order_class_list):

    #Labels
    labels_list_init=[]
    for clase in order_class_list:
        labels_list_init+=[clase]*number_sent_per_class


    #Array bow embeddings
    array_bow=embeddings_array_gen(list_sentences)
    sil_score=silhouette_score(X=array_bow,labels=labels_list_init,metric="euclidean")
    print(f"Silhoutte Score: {sil_score}")
    return sil_score



#Calcular el embedding similarity: intra/inter cluster
def embs_similarity(list_sentences,number_sent_per_class,order_class_list):

    #Labels
    labels_list_init=[]
    for clase in order_class_list:
        labels_list_init+=[clase]*number_sent_per_class


    #Array bow embeddings
    array_bow=embeddings_array_gen(list_sentences)


    #Intra cluster Cosine Distance: distancia promedio entre puntos del mismo grupo --> Cohesión dentro del grupo
    intra_mean_cluster=[]
   
    for i in range(0,len(order_class_list)):
        clase_actual=order_class_list[i]
        indices_clases_igls=np.where(np.array(labels_list_init)==clase_actual)[0]
        
        
        emb_vectors=array_bow[indices_clases_igls]
        
        
        cos_dis=distance.cdist(emb_vectors,emb_vectors,metric="cosine") #cosine_similarity(X=emb_vectors)  distance.cdist(centroides_array,centroides_array,metric="cosine")  
        

        np.fill_diagonal(cos_dis,val=np.nan)
        mean_points_cluster=np.nanmean(cos_dis,axis=1)  #mean por cada fila --> cada punto vs sus vecinos
        
        intra_mean_cluster.append(mean_points_cluster) #similitud promedio de CADA punto dentro del cluster con el resto del cluster”
        
        
    intra_mean_cos_dis=np.mean(np.array(intra_mean_cluster),axis=1).reshape(-1,1)

    print(f"Intra Mean Cluster Cosine Distance: {np.array(intra_mean_cluster)},")
    print(f"Intra Mean Cosine Distance por cada cluster (fila): {intra_mean_cos_dis}")


    print("====")
    #Inter cluster similarity --> Euclidean y Cosine distancia para la distancia ENTRE clusters en base a sus centroides
    
    centroides=[]
    for i in range(0,len(order_class_list)):
        clase_actual=order_class_list[i]
        
        indices=np.where(np.array(labels_list_init)==clase_actual)[0]
        centroid_cluster_i=np.mean(array_bow[indices],axis=0)
        
        centroides.append(centroid_cluster_i)


    #Calculo la matriz de distancia entre los centroides: Filas --> clusters, Columnas --> distancia entre cada cluster
    centroides_array=np.array(centroides)
    matriz_distancia_euc_centroides=distance.cdist(centroides_array,centroides_array,metric="euclidean")
    matriz_distancia_cos_centroides=distance.cdist(centroides_array,centroides_array,metric="cosine")   

    np.fill_diagonal(matriz_distancia_euc_centroides,val=np.nan)
    np.fill_diagonal(matriz_distancia_cos_centroides,val=np.nan)
    
    
    mean_distancia_euc_clusters=np.nanmean(matriz_distancia_euc_centroides,axis=1).reshape(-1,1)
    mean_distancia_cos_clusters=np.nanmean(matriz_distancia_cos_centroides,axis=1).reshape(-1,1)
    
    
    #print(f"Matriz - Distancia Euc. entre Centroides: {matriz_distancia_euc_centroides}")
    #print(f"Matriz - Distancia Cos. entre Centroides: {matriz_distancia_cos_centroides}")


    print(f"Mean Distancia Clusters EUC: {mean_distancia_euc_clusters.shape},{mean_distancia_euc_clusters}")
    print(f"Mean Distance Clusters COS: {mean_distancia_cos_clusters.shape},{mean_distancia_cos_clusters}")

    return intra_mean_cos_dis,mean_distancia_cos_clusters 



#Preguntas: WHAT color, shape, present
pregs_text=["what color is the rectangle?","what color is the triangle?","what is the color of the circle?","what is the color of the shape?",
            "what shape is present?","what shape does the image contain?","what shape is in the image?","what is the gray shape?",
            "is a rectangle present?","is there not a circle?","is there a triangle?","is there not a green shape?"]


#sil_score(pregs_text,3,["color","shape","present"])
#_,_=embs_similarity(pregs_text,4,["color","shape","present"]) #Uncomment para ejecutar de manera local



#Return:
#filas: clases (clusters)
#Columna: cosine distance PROMEDIO
