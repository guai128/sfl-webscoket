# Description: This file is used to split the date into
# training data for every client and testing data for server.
import os
from glob import glob

import pandas as pd
from data.DataLoader import dataset_iid

# config
client_num = 2  # the number of clients
test_size = 0.2  # the proportion of the dataset to include in the test split
random_state = 42  # random seed

# load data
if __name__ == '__main__':
    df = pd.read_csv('./HAM10000_metadata.csv')
    lesion_type = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    imageid_path = {os.path.splitext(os.path.basename(x))[0]: os.path.join('data', x)
                    for x in glob(os.path.join('*', '*.jpg'))}

    df['path'] = df['image_id'].map(imageid_path.get)
    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['target'] = pd.Categorical(df['cell_type']).codes
    # get the column of the target and the path
    df = df[['path', 'target']]

    print('finish loading data')

    # split the data
    train = df.sample(frac=1 - test_size, random_state=random_state)
    test = df.drop(train.index)

    # then split the train data into client_num parts
    client_data_idxs = dataset_iid(train.shape[0], client_num)

    # save the data
    for i in range(client_num):
        client_data = train.iloc[list(client_data_idxs[i])]
        client_data.to_csv(f'split-data/client{i}.csv', index=False)

    test.to_csv('split-data/test.csv', index=False)
