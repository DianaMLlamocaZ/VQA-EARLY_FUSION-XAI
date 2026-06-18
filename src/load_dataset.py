from torch.utils.data import Dataset
import torch
import cv2
import re
from utils.utils import bag_of_words

class DatasetVQA(Dataset):
    #Inicialización
    def __init__(self,data,vocab_train,path):
        self.data=data
        self.dict_vocab=vocab_train
        self.path=path

        #Clases
        with open("./data/answers.txt") as file_classes:
            clases=[line.rstrip('\n') for line in file_classes]
        
        self.clases={clase:idx for (idx,clase) in enumerate(clases)}
        

    #Tamaño del dataset
    def __len__(self):
        len_dataset=len(self.data)
        return len_dataset


    #Obtener item
    def __getitem__(self,idx):
        
        #Datos obtenidos
        q,a,id_img=self.data[idx][0],self.data[idx][1],self.data[idx][2]
        #print(f"Question: {q}")
        #Procesamiento de texto
        question_preproc=re.sub(r'\?',' ?',q)

        tokens_id_list=[self.dict_vocab[word] for word in question_preproc.split(" ")]
        answer_class=[self.clases[a]]

        #Bag Of Words (Si el usuario desea entrenar con este enfoqoe)
        array_bow=bag_of_words(vocab=self.dict_vocab,list_id_tokens=tokens_id_list)

        
        #Procesamiento de imágenes
        path_img=f"{self.path}/images/{id_img}.png"
        
        img_array_rgb=cv2.resize(cv2.imread(filename=path_img,flags=1)[:,:,::-1],dsize=(64,64)) #RGB  #cv2.imread(filename=path_img,flags=1)
        img_tensor_rgb=torch.from_numpy(img_array_rgb).permute(2,0,1)/255
        

        return (torch.tensor(array_bow),img_tensor_rgb),torch.tensor(answer_class) #q,a,img_tensor_rgb
