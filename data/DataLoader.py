import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

        # load all pictures into memory
        self.images = [Image.open(path).resize((64, 64)) for path in self.df['path']]
        if self.transform:
            self.images = [self.transform(image) for image in self.images]

        self.targets = torch.from_numpy(np.array(self.df['target']))
        self.len = len(self.images)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def dataset_iid(data_size, num_users):
    num_items = int(data_size / num_users / 2)
    user_items = [num_items] * num_users

    # randomly assign the remaining items to users
    # get the distribution of the remaining items with their sum is 1
    probs = np.random.dirichlet(np.ones(num_users), size=1)[0]

    for i in range(data_size - num_items * num_users):
        choice = np.random.choice(num_users, p=probs)
        user_items[choice] += 1

    dict_users, all_idxs = {}, [i for i in range(data_size)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, user_items[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def load_data(filepath: str):
    df = pd.read_csv(filepath)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.Pad(3),
                                           transforms.RandomRotation(10),
                                           transforms.CenterCrop(64),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std)
                                           ])

    return SkinData(df, transform=train_transforms)
