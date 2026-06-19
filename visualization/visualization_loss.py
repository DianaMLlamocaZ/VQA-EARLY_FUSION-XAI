import matplotlib.pyplot as plt
import json

def loss_visualization():
    train_loss_name_json="acá_coloca_el_nombre_del_archivo_json_del_train_loss"
    valid_loss_name_json="acá_coloca_el_nombre_del_archivo_json_del_valid_loss"

    with open(f"./results/{train_loss_name_json}","r") as f1, open(f"./results/{valid_loss_name_json}","r") as f2:
        train_loss,valid_loss=json.load(f1),json.load(f2)

    plt.plot(range(len(train_loss)),train_loss,color="r",label="Train Loss")
    plt.plot(range(len(train_loss)),valid_loss,color="b",label="Valid Loss")

    plt.legend()
    plt.show()


loss_visualization()
