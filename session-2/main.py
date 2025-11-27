import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy
import kagglehub
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(dataloader,my_model,optimizer,criterion,loss_history):
    my_model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = my_model(x)
        #print(f"Output {outputs.shape}")
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)
        total_loss += loss_value
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss


def eval_single_epoch(testloader,my_model,criterion):
    my_model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = my_model(x)

            loss = criterion(outputs, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # outputs may be logits or log-probs; argmax works for both
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

    avg_test_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    print(f"Validation Loss: {avg_test_loss:.4f}  Accuracy: {accuracy:.4f}")
    return avg_test_loss


def train_model(config):
    
     # 1) Descargar dataset de Kaggle
    path = kagglehub.dataset_download("gpreda/chinese-mnist")
    print("Path to dataset files:", path)
    
    # 2) Transformaciones
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    
     # 3) Dataset completo
    my_dataset = MyDataset(path + "/data/data",path + "/chinese_mnist.csv",transform=data_transforms)
    
    
    # 4) TRAIN/VAL/TEST SPLIT
    total_size = len(my_dataset)
    test_size  = int(0.10 * total_size)
    val_size   = int(0.10 * total_size)
    train_size = total_size - test_size - val_size
    
    
    train_dataset, val_dataset, test_dataset = random_split(
        my_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    
    # 5) DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    
    # 6) Crear modelo 15 clases .. 15 tipos de numeros chinos
    my_model = MyModel(15).to(device)
    
     # 7) Optimizador + criterio
    #optimizer = optim.SGD(my_model.parameters(), lr=config["lr"])
    optimizer = optim.Adam(my_model.parameters(), lr=config["lr"])
    # Use CrossEntropyLoss for logits (combines LogSoftmax + NLLLoss)
    criterion = nn.CrossEntropyLoss()
    #para nnloss el modelo debe devoler softmax
    #criterion =nn.NLLLoss()
    
    loss_history = []
    
     # 8) Entrenamiento
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}")
        avg_train_loss = train_single_epoch(train_loader, my_model, optimizer, criterion, loss_history)
        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")
        avg_val_loss = eval_single_epoch(val_loader, my_model, criterion)
        print(f"Epoch {epoch+1} val loss: {avg_val_loss:.4f}")
    #    print("hola")
    # 9) Evaluación final en test
    #print("Evaluación final en test:")
    #eval_single_epoch(test_loader, my_model, criterion)

    return my_model


if __name__ == "__main__":

    config = {
        "hyperparam_1": 1,
        "hyperparam_2": 2,
        "epochs": 20,
        "lr": 0.01,
        #"num_classes": 15,
        #"n_features": 20,
        #"n_hidden": 64,
        #"n_outputs": 10,
        "batch_size": 128
    }
    train_model(config)
