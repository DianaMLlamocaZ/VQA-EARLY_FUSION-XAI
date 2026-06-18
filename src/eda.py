import matplotlib.pyplot as plt
import numpy as np

from utils.utils import load_data,load_data_loaders

def bar_plot_total_train_data():
    data_train=load_data(path_data="./data/train")
    data_array_train=np.array(data_train)   #(preg,resp,img_id)
    

    total_clases_data=data_array_train[:,1]
    unique_classes,count_classes=np.unique(ar=total_clases_data,return_counts=True)
    
    plt.barh(unique_classes,count_classes)
    plt.show()



def bar_plot_split_data():
    train_data,valid_data,test_data,vocab=load_data_loaders()
    train_array_clases,valid_array_clases=np.array(train_data)[:,1],np.array(valid_data)[:,1]
    

    unique_clases_train,count_clases_train=np.unique(ar=train_array_clases,return_counts=True)
    unique_clases_valid,count_clases_valid=np.unique(ar=valid_array_clases,return_counts=True)
       

    fig,ax=plt.subplots(nrows=1,ncols=2)
    
    ax[0].barh(unique_clases_train,count_clases_train,color="blue")
    ax[0].set_title("Train Data")

    ax[1].barh(unique_clases_valid,count_clases_valid,color="green")
    ax[1].set_title("Valid Data")

  
    plt.tight_layout()
    plt.show()

#bar_plot_total_train_data()  #--> MUESTRA IMBALANCE DE CLASES EN LA DATA DE TRAIN
bar_plot_split_data()
