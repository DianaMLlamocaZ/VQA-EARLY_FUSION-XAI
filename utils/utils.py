#Funciones que ayudan al preprocesamiento
import json
import re
import numpy as np
import torch
import cv2

from src.models import ModeloVQA_BoW
from sklearn.utils.class_weight import compute_class_weight

#Cargar datos
def load_data(path_data):
    
    #Obtener los datos: question, answer, imagen_id
    with open(f"{path_data}/questions.json") as file:
        data_json=json.load(fp=file)
            
    data=data_json
    
    return data
    


#Split de la data
def split_train_val_data(data,train_size):
    #Clases
    with open("./data/answers.txt") as file_classes:
        clases=[line.rstrip('\n') for line in file_classes]
        
    clases={clase:idx for (idx,clase) in enumerate(clases)}        
    
        
    #División de datos
    len_data=len(data)
    train_data_size=int(len_data*train_size)

    data_train,data_valid=data[:train_data_size],data[train_data_size:]
    #print(f"data train: {len(data_train)}, data train 0: {data_train[0]}")

    return data_train,data_valid



#Creación del vocabulario
def create_vocab(data_train):
    dict_vocab={}
    id_inc=0

    
    #Bucle de inserción
    for i in range(0,len(data_train)):
        #Preprocesamiento de datos
        oracion=data_train[i][0]
        oracion_prcs=re.sub(r'\?',' ?',oracion)
        
        oracion_final=oracion_prcs.split(" ")
        
        #Inserción de datos en el diccionario
        for word in oracion_final:
            if word not in dict_vocab:
                dict_vocab[word]=id_inc
                id_inc+=1

    
    return dict_vocab



#Bag of Words dada la lista de tokens
def bag_of_words(vocab,list_id_tokens):
    bag_of_words=np.zeros(shape=(1,len(vocab))) #[1,len(vocab)]

    for id in list_id_tokens:
        bag_of_words[0][id]+=1
    
    return bag_of_words



#Generar el BOW de una oración dada por el usuario
def generate_bow_vector(vocab,sentence):
    
    #Preprocesamiento de texto
    sentence_lower=sentence.lower()
    oracion_prcs=re.sub(r'\?',' ?',sentence_lower)
    oracion_final=oracion_prcs.split(" ")

    #ID List tokens
    list_id_tokens=[vocab[word] for word in oracion_final if word in vocab] #Solo agregar las palabras identificadas en el vocabulario

    #BOW vector
    bow_vector=bag_of_words(vocab=vocab,list_id_tokens=list_id_tokens)
    #print(f"bow vector: {bow_vector}")

    return torch.tensor(bow_vector)



#Cargar y procesar la imagen
def load_image(path_image):
    
    #Cargar y resize imagen
    imagen_rgb_array=cv2.imread(filename=path_image,flags=1)[:,:,::-1]
    imagen_rgb_rsz=cv2.resize(src=imagen_rgb_array,dsize=(64,64))

    #Tensor
    imagen_tensor=torch.from_numpy(imagen_rgb_rsz).permute(2,1,0)/255

    return imagen_tensor.unsqueeze(0)



#Guardar datos de las losses en formato json
def save_losses(train_loss_list,valid_loss_list):
    with open("./results/train_loss.json","w") as f1, open("./results/valid_loss.json","w") as f2:
        json.dump(train_loss_list,f1), json.dump(valid_loss_list,f2)



#Cargar modelo
def load_model(path_weights_model):
    instance_model=ModeloVQA_BoW(vocab_length=27,num_clases=13)

    weights=torch.load(path_weights_model,weights_only=True)
    instance_model.load_state_dict(weights)
    #print(f"weights: {weights}")    
    return instance_model



#Cargar las clases
def load_classes():
    with open("./data/answers.txt") as file_classes:
        clases=[line.rstrip('\n') for line in file_classes]

    clases_final={idx:clase for idx,clase in enumerate(clases)}
    
    return clases_final



#Cargar data loaders
def load_data_loaders():

    train=load_data(path_data="./data/train")
    test_data=load_data(path_data="./data/test")
    
    train_data,valid_data=split_train_val_data(data=train,train_size=0.8)


    #Creación del vocabulario a partir de la data de train (evitar data leakage)
    vocab=create_vocab(data_train=train_data)


    return train_data,valid_data,test_data,vocab
   


#Class Weight
def class_weight():
    train_data,valid_data,test_data,vocab=load_data_loaders()
    train_array_clases,valid_array_clases=np.array(train_data)[:,1],np.array(valid_data)[:,1]
    

    unique_clases_train=np.array(list(load_classes().values()))
    print(f"unique clases: {unique_clases_train}")

    weight_classes=compute_class_weight(class_weight="balanced",classes=unique_clases_train,y=train_array_clases)
    
    return weight_classes



#Cargar el modelo dada la instancia
def load_model_instancia(instancia_model,path_weights):
    
    weights_model=torch.load(path_weights,weights_only=True)
    
    instancia_model.load_state_dict(weights_model)
    
    return instancia_model



#Seed
def set_seed(seed=42):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
