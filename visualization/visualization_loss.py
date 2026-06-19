import matplotlib.pyplot as plt
import json

def loss_visualization():
    with open("./results/train_loss_with_class_w8s_layer_norm.json","r") as f1, open("./results/valid_loss_with_class_w8s_layer_norm.json","r") as f2:
        train_loss,valid_loss=json.load(f1),json.load(f2)

    plt.plot(range(len(train_loss)),train_loss,color="r",label="Train Loss")
    plt.plot(range(len(train_loss)),valid_loss,color="b",label="Valid Loss")

    plt.legend()
    plt.show()


loss_visualization()
