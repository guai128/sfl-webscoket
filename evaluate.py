import os
from glob import glob

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from Client import ClientModel
from data.DataLoader import SkinData, load_data
from Server import ServerNormal, SeverNonNormal


def evaluate(client_model, server_non_normal, server_normal, test_loader, device='cpu'):
    client_model.eval()
    server_non_normal.eval()
    server_normal.eval()

    tot_test = 0
    correct_test = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        client_fx = client_model.forward(data)
        layer_count = client_model.layerCount()
        if layer_count < 4:
            server_fx = server_non_normal(client_fx, layer_count - 2)
            server_fx = server_normal(server_fx)
        else:
            server_fx = server_normal(client_fx)

        pred = server_fx.argmax(dim=1, keepdim=True)
        correct_test += pred.eq(target.view_as(pred)).sum().item()
        tot_test += len(data)

    return correct_test / tot_test


def main():
    client_model = ClientModel(4)
    server_non_normal = SeverNonNormal()
    server_normal = ServerNormal()

    # load model parameters
    client_model.load_state_dict(torch.load("model/client_model.pth"))
    server_normal.load_state_dict(torch.load("model/server_normal.pth"))

    # load test data
    test_dataset = load_data("data/split-data/test.csv")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    accuracy = evaluate(client_model, server_non_normal, server_normal, test_loader, 'cpu')
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
