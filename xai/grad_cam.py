import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


import matplotlib.cm as cm
import matplotlib.colors as colors


class GradCAM:
    def __init__(self,img_tensor,question_tensor,modelo,view_page:bool):               
        self.modelo=modelo
        self.img_tensor=img_tensor
        self.question_tensor=question_tensor
        self.view_page=view_page

        self.last_layer_cnn=modelo.vc2  #Última capa convolucional: vc2

        self.activations=[]             #Output tensor de la última capa convolucional
        self.gradients=[]               #Gradiente respecto a la salida (output) de la última capa convolucional

        
    
    def hook_forward_function(self,module,input,output):   #Hook del forward pass
        self.activations.append(output.detach().numpy().squeeze())  #squeeze porque se sobreentiende que es solo 1 sample
        

    def forward_output(self):
        self.modelo.eval()
        hook_forward=self.last_layer_cnn.register_forward_hook(self.hook_forward_function)  #Se coloca el hook a la capa
        pred=self.modelo(image_tensor=self.img_tensor,question_tensor=self.question_tensor) #Forward Pass (automáticamente el se llama el hook)
        hook_forward.remove()                                                               #Se remueve el hook para evitar llamadas cada vez que hay forward pass

        return self.activations
    


    def hook_backward_function(self,module,grad_input,grad_output):   #Hook del backward pass
        self.gradients.append(grad_output[0].numpy().squeeze())
        

    def backward_pass(self):
        self.modelo.eval()
        hook_backward=self.last_layer_cnn.register_backward_hook(self.hook_backward_function)
        preds=self.modelo(image_tensor=self.img_tensor,question_tensor=self.question_tensor)
        
        
        score=torch.max(preds,dim=-1).values
        
        score.backward()    #Aquí automáticamente se llama al "backward hook" para la última capa convolucional
        hook_backward.remove()  #Se elimina el hook para evitar llamadas continuas

        return self.gradients,torch.argmax(preds,dim=-1).item()
    
    
    def weights_feature_maps(self):
        
        avg_gradients_fm=np.mean(self.gradients[0],axis=(1,2))
        avg_gradients_fm_final=avg_gradients_fm[:,np.newaxis,np.newaxis]
        
        plt.hist(avg_gradients_fm,bins=50,label="Average Gradients for Feature Map")
        plt.legend()
        plt.plot()

        return avg_gradients_fm_final
    

    def heatmap(self,weights_fm,img_src):
        
        weighted_heatmap=np.sum(weights_fm*self.activations[0],axis=0)  #multiplicación de los gradient weights * las activaciones de los feature maps de la última capa
        
        heatmap_final=np.maximum(weighted_heatmap,0)    #permanecen solo las activaciones positivas (contribuyen a la pred. de la clase)
        
        upsampled_heatmap=cv2.resize(src=heatmap_final,dsize=img_src.shape[2:],interpolation=cv2.INTER_LINEAR) #resize del heatmap para que tenga el mismo shape que la imagen original
        
        

        #======#

        #Rescalado del heatmap a valores de 0 a 1  (el "min") es opcional, porque al aplicar RELU se sobreentiende que es 0
        heatmap_rescale=(upsampled_heatmap-upsampled_heatmap.min())/(upsampled_heatmap.max()-upsampled_heatmap.min()+1e-8)
        print(f"heatmap rescale max, min: {heatmap_rescale.max()}, {heatmap_rescale.min()}")

        
        #Scale heatmap
        heatmap_mapped=np.uint8(heatmap_rescale*255)
        
        heatmap_colored_bgr=cv2.applyColorMap(src=heatmap_mapped,colormap=cv2.COLORMAP_JET)
        heatmap_colored_final_rgb=cv2.cvtColor(heatmap_colored_bgr,cv2.COLOR_BGR2RGB) #heatmap volteado

        
        #Imagen ya en RGB
        img_final=np.uint8(np.array(img_src.squeeze(0).permute(1,2,0))*255)  #imagen volteada
        

        #Overlay del heatmap y la imagen
        alpha=0.5
        heatmap_overlayed=cv2.addWeighted(src1=heatmap_colored_final_rgb,alpha=alpha,src2=img_final,beta=1-alpha,gamma=0) #NO cambia CANALES --> RGB


        #Mapa de color
        norm=colors.Normalize(vmin=heatmap_mapped.min(),vmax=heatmap_mapped.max())

        sm=cm.ScalarMappable(norm=norm,cmap='jet')
        sm.set_array([])



        #======#


        #visualise the heatmap RGB
        fig,ax=plt.subplots(1,3,figsize=(12,12)) #1,2

        #Imagen
        ax[0].imshow(img_final.transpose(1,0,2))
        ax[0].set_title("Imagen Original")
        ax[0].axis("off")


        #Heatmap SOLO
        ax[1].imshow(heatmap_colored_final_rgb.transpose(1,0,2))
        ax[1].set_title("Heatmap Puro RGB")
        ax[1].axis("off")
        

        #Overlayed Heatmap
        ax[2].imshow(heatmap_overlayed.transpose(1,0,2))
        ax[2].set_title("Heatmap Overlayed")
        ax[2].axis("off")


        fig.colorbar(sm, ax=ax) #ColorBar


        if not self.view_page:
            plt.show()

        
        return upsampled_heatmap,heatmap_colored_bgr
