import torch.nn

#Modelo VQA w. BoW
#CNN: Visual Feature Embedding
class ModeloVQAWith_outCW_BoW(torch.nn.Module):
    def __init__(self,vocab_length,num_clases):
        super().__init__()

        #CNN: Visual Feature Embedding
        self.vc1=torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)    
        self.vc2=torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)

        self.mp=torch.nn.MaxPool2d(kernel_size=2)                              

        self.v_fl=torch.nn.Linear(in_features=32*14*14,out_features=32)       #Se multiplica 2 veces porque representa al ancho y alto
        
        
        
        #Question Embeddings: Linear Layers
        self.q_l1=torch.nn.Linear(in_features=vocab_length,out_features=64) 
        self.q_l2=torch.nn.Linear(in_features=64,out_features=128)   
        self.q_l3=torch.nn.Linear(in_features=128,out_features=32)

        
        
        #Multiplicar embeddings de los features textuales y de imagen
        self.c1=torch.nn.Linear(in_features=32,out_features=16)
        
        
        #Final layer joint embeddings
        self.fl_je=torch.nn.Linear(in_features=16,out_features=num_clases)
        

        #Activation functions
        self.relu=torch.nn.ReLU()
    

    def forward(self,image_tensor,question_tensor):
        #Image
        f1_img=self.mp(self.relu(self.vc1(image_tensor)))
        
        f2_img=self.mp(self.relu(self.vc2(f1_img)))
        
        vse_flatten=f2_img.view(f2_img.shape[0],-1)
        
        flatten_img=self.v_fl(vse_flatten)              
        
        

        #Question
        f1_q=self.relu(self.q_l1(question_tensor))
        
        f2_q=self.relu(self.q_l2(f1_q))   
        
        f3_q=self.q_l3(f2_q)
        
        

        #Final Joint Embedings:
        #--> Multiplicación de vectores de imagen y texto
        join_embeddings=flatten_img*f3_q    
        
        je_1=self.relu(self.c1(join_embeddings))
         
        je_2_final=self.fl_je(je_1)
        

        return je_2_final
