import torch.nn

#Modelo VQA w. BoW
#CNN: Visual Feature Embedding
class ModeloVQA_BoW(torch.nn.Module):
    def __init__(self,vocab_length,num_clases):
        super().__init__()

        #CNN: Visual Feature Embedding
        self.vc1=torch.nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3)    
        self.vc2=torch.nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3)
        self.vc3=torch.nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3)

        self.mp=torch.nn.MaxPool2d(kernel_size=2)                               

        self.v_fl=torch.nn.Linear(in_features=16*6*6,out_features=32)       #Se multiplica 2 veces porque representa al ancho y alto
        
        self.layer_norm_1=torch.nn.LayerNorm(32)   #====>LAYER NORM para evitar números GRANDES
        

        #Question Embeddings: Linear Layers
        self.q_l1=torch.nn.Linear(in_features=vocab_length,out_features=64) #32, 64
        self.q_l2=torch.nn.Linear(in_features=64,out_features=128)   #32, 64
        self.q_l3=torch.nn.Linear(in_features=128,out_features=32)

        self.layer_norm_2=torch.nn.LayerNorm(32)    #====>LAYER NORM para evitar números GRANDES
        

        #Concatenate or sumar or multiplicate embeddings y obtener predicciones --> Multiplico en este caso
        self.c1=torch.nn.Linear(in_features=32,out_features=16)
        self.layer_norm_3=torch.nn.LayerNorm(16)
        
        #Final layer joint embeddings
        self.fl_je=torch.nn.Linear(in_features=16,out_features=num_clases)
        

        #Activation functions
        self.relu=torch.nn.ReLU()
    

    def forward(self,image_tensor,question_tensor):
        #Image
        f1_img=self.mp(self.relu(self.vc1(image_tensor)))
        #print(f"f1 img mp: {f1_img.shape}")

        f2_img=self.mp(self.relu(self.vc2(f1_img)))
        #print(f"f2 img mp: {f2_img.shape}")

        f3_img=self.mp(self.relu(self.vc3(f2_img)))
        #print(f"f3 img: {f3_img.shape}")

        vse_flatten=f3_img.view(f3_img.shape[0],-1) #f2_img.shape[0],-1
        #print(f"vse flatten: {vse_flatten.shape}")

        flatten_img=self.v_fl(vse_flatten)              #Linear Layer
        #print(f"flatten img: {flatten_img.shape}")

        layer_norm_img=self.layer_norm_1(flatten_img) #Añadido
        #print(f"layer norm img: {layer_norm_img}")


        #Question
        f1_q=self.relu(self.q_l1(question_tensor))
        #print(f"f1 question: {f1_q.shape}, dtype: {f1_q.dtype}")

        f2_q=self.relu(self.q_l2(f1_q))   #self.relu(self.q_l2(f1_q))
        #print(f"f2 question: {f2_q.shape}")

        f3_q=self.q_l3(f2_q)
        #print(f"f3_q: {f3_q.shape}")
       
        layer_norm_bow=self.layer_norm_2(f3_q)
        #print(f"layer norm bow: {layer_norm_bow}")


        #Final Joint Embedings:
        #Multiplicación de vectores de imagen y texto
        join_embeddings=layer_norm_img*layer_norm_bow #layer_norm_img*layer_norm_bow    #flatten_img*f3_q
        
        je_1=self.relu(self.c1(join_embeddings))
        je_1_norm=self.layer_norm_3(je_1)
        #print(f"joint emb 1: {je_1}") 

        je_2_final=self.fl_je(je_1_norm) #je_1    je_1_norm
        #print(f"final layer joint emb: {je_2_final.shape}")


        return je_2_final
