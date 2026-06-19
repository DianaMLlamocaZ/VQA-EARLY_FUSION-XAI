import numpy as np
import cv2
import torch

import io
import base64

#Cargar la imagen input del usuario
def load_input_image(image):
    file_bytes=np.frombuffer(image.read(),np.uint8)
    
    #Numpy Array
    img=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)[:,:,::-1] #w,h,c --> c,w,h
    imagen_rgb_rsz=cv2.resize(src=img,dsize=(64,64))
    
    #Tensor
    imagen_tensor=torch.from_numpy(imagen_rgb_rsz).permute(2,1,0)/255

    return imagen_tensor.unsqueeze(0)



#Guardar la imagen para mostrarlo en Flask
def save_scatter_flask(figure):
    
    #Variable(memoria) para almacenar la gráfica
    scatter_img=io.BytesIO()

    #Almacenar
    figure.savefig(scatter_img,format="jpg",bbox_inches="tight")

    #Se vuelve el cursor al inicio del buffer
    scatter_img.seek(0)
    
    #Convierto a b64 para el html
    scatter_final=base64.b64encode(scatter_img.getvalue()).decode("utf-8")

    return scatter_final
