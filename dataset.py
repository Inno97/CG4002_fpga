# import the data from csv
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

class Dataset():
    def __init__(self):
        self.df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/dataset/dataset_56_inputs_5_channels.csv')
        self.train_loader, self.test_loader = load_datasets(self.df)
        print("loaded dataset")

    # get the next input from the either DataLoader (datasets are random)    
    def get_next_train_data(self):
        test_input, test_output = next(iter(self.train_loader))
        #test_output = test_output[0]

        #print("input: {}, output: {}".format(test_input[0], test_output))
        return test_input, test_output

    def get_next_train_data(self):
        test_input, test_output = next(iter(self.test_loader))
        #test_output = test_output[0]

        #print("input: {}, output: {}".format(test_input[0], test_output))
        return test_input, test_output

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

# helper functions
def parse_input(df, starting_index):
    data = np.array(((df.iloc[starting_index, 0] / 180,df.iloc[starting_index + 1, 0] / 180,df.iloc[starting_index + 2, 0] / 180,df.iloc[starting_index + 3, 0] / 180,df.iloc[starting_index + 4, 0] / 180,df.iloc[starting_index + 5, 0] / 180,df.iloc[starting_index + 6, 0] / 180,df.iloc[starting_index + 7, 0] / 180, 
          df.iloc[starting_index + 8, 0] / 180, df.iloc[starting_index + 9, 0] / 180, df.iloc[starting_index + 10, 0] / 180, df.iloc[starting_index + 11, 0] / 180, df.iloc[starting_index + 12, 0] / 180, df.iloc[starting_index + 13, 0] / 180, df.iloc[starting_index + 14, 0] / 180, df.iloc[starting_index + 15, 0] / 180,
          df.iloc[starting_index + 16, 0] / 180, df.iloc[starting_index + 17, 0] / 180, df.iloc[starting_index + 18, 0] / 180, df.iloc[starting_index + 19, 0] / 180, df.iloc[starting_index + 20, 0] / 180, df.iloc[starting_index + 21, 0] / 180, df.iloc[starting_index + 22, 0] / 180, df.iloc[starting_index + 23, 0] / 180,
          df.iloc[starting_index + 24, 0] / 180,df.iloc[starting_index + 25, 0] / 180,df.iloc[starting_index + 26, 0] / 180,df.iloc[starting_index + 27, 0] / 180,df.iloc[starting_index + 28, 0] / 180,df.iloc[starting_index + 29, 0] / 180,df.iloc[starting_index + 30, 0] / 180,df.iloc[starting_index + 31, 0] / 180, 
          df.iloc[starting_index + 32, 0] / 180, df.iloc[starting_index + 33, 0] / 180, df.iloc[starting_index + 34, 0] / 180, df.iloc[starting_index + 35, 0] / 180, df.iloc[starting_index + 36, 0] / 180, df.iloc[starting_index + 37, 0] / 180, df.iloc[starting_index + 38, 0] / 180, df.iloc[starting_index + 39, 0] / 180,
          df.iloc[starting_index + 40, 0] / 180, df.iloc[starting_index + 41, 0] / 180, df.iloc[starting_index + 42, 0] / 180, df.iloc[starting_index + 43, 0] / 180, df.iloc[starting_index + 44, 0] / 180, df.iloc[starting_index + 45, 0] / 180, df.iloc[starting_index + 46, 0] / 180, df.iloc[starting_index + 47, 0] / 180,  
          df.iloc[starting_index + 48, 0] / 180, df.iloc[starting_index + 49, 0] / 180, df.iloc[starting_index + 50, 0] / 180, df.iloc[starting_index + 51, 0] / 180, df.iloc[starting_index + 52, 0] / 180, df.iloc[starting_index + 53, 0] / 180, df.iloc[starting_index + 54, 0] / 180, df.iloc[starting_index + 55, 0] / 180),
         (df.iloc[starting_index, 1] / 180,df.iloc[starting_index + 1, 1] / 180,df.iloc[starting_index + 2, 1] / 180,df.iloc[starting_index + 3, 1] / 180,df.iloc[starting_index + 4, 1] / 180,df.iloc[starting_index + 5, 1] / 180,df.iloc[starting_index + 6, 1] / 180,df.iloc[starting_index + 7, 1] / 180, 
          df.iloc[starting_index + 8, 1] / 180, df.iloc[starting_index + 9, 1] / 180, df.iloc[starting_index + 10, 1] / 180, df.iloc[starting_index + 11, 1] / 180, df.iloc[starting_index + 12, 1] / 180, df.iloc[starting_index + 13, 1] / 180, df.iloc[starting_index + 14, 1] / 180, df.iloc[starting_index + 15, 1] / 180,
          df.iloc[starting_index + 16, 1] / 180, df.iloc[starting_index + 17, 1] / 180, df.iloc[starting_index + 18, 1] / 180, df.iloc[starting_index + 19, 1] / 180, df.iloc[starting_index + 20, 1] / 180, df.iloc[starting_index + 21, 1] / 180, df.iloc[starting_index + 22, 1] / 180, df.iloc[starting_index + 23, 1] / 180,
          df.iloc[starting_index + 24, 1] / 180,df.iloc[starting_index + 25, 1] / 180,df.iloc[starting_index + 26, 1] / 180,df.iloc[starting_index + 27, 1] / 180,df.iloc[starting_index + 28, 1] / 180,df.iloc[starting_index + 29, 1] / 180,df.iloc[starting_index + 30, 1] / 180,df.iloc[starting_index + 31, 1] / 180, 
          df.iloc[starting_index + 32, 1] / 180, df.iloc[starting_index + 33, 1] / 180, df.iloc[starting_index + 34, 1] / 180, df.iloc[starting_index + 35, 1] / 180, df.iloc[starting_index + 36, 1] / 180, df.iloc[starting_index + 37, 1] / 180, df.iloc[starting_index + 38, 1] / 180, df.iloc[starting_index + 39, 1] / 180,
          df.iloc[starting_index + 40, 1] / 180, df.iloc[starting_index + 41, 1] / 180, df.iloc[starting_index + 42, 1] / 180, df.iloc[starting_index + 43, 1] / 180, df.iloc[starting_index + 44, 1] / 180, df.iloc[starting_index + 45, 1] / 180, df.iloc[starting_index + 46, 1] / 180, df.iloc[starting_index + 47, 1] / 180,  
          df.iloc[starting_index + 48, 1] / 180, df.iloc[starting_index + 49, 1] / 180, df.iloc[starting_index + 50, 1] / 180, df.iloc[starting_index + 51, 1] / 180, df.iloc[starting_index + 52, 1] / 180, df.iloc[starting_index + 53, 1] / 180, df.iloc[starting_index + 54, 1] / 180, df.iloc[starting_index + 55, 1] / 180),
         (df.iloc[starting_index, 2] / 2,df.iloc[starting_index + 1, 2] / 2,df.iloc[starting_index + 2, 2] / 2,df.iloc[starting_index + 3, 2] / 2,df.iloc[starting_index + 4, 2] / 2,df.iloc[starting_index + 5, 2] / 2,df.iloc[starting_index + 6, 2] / 2,df.iloc[starting_index + 7, 2] / 2, 
          df.iloc[starting_index + 8, 2] / 2, df.iloc[starting_index + 9, 2] / 2, df.iloc[starting_index + 10, 2] / 2, df.iloc[starting_index + 11, 2] / 2, df.iloc[starting_index + 12, 2] / 2, df.iloc[starting_index + 13, 2] / 2, df.iloc[starting_index + 14, 2] / 2, df.iloc[starting_index + 15, 2] / 2,
          df.iloc[starting_index + 16, 2] / 2, df.iloc[starting_index + 17, 2] / 2, df.iloc[starting_index + 18, 2] / 2, df.iloc[starting_index + 19, 2] / 2, df.iloc[starting_index + 20, 2] / 2, df.iloc[starting_index + 21, 2] / 2, df.iloc[starting_index + 22, 2] / 2, df.iloc[starting_index + 23, 2] / 2,
          df.iloc[starting_index + 24, 2] / 2,df.iloc[starting_index + 25, 2] / 2,df.iloc[starting_index + 26, 2] / 2,df.iloc[starting_index + 27, 2] / 2,df.iloc[starting_index + 28, 2] / 2,df.iloc[starting_index + 29, 2] / 2,df.iloc[starting_index + 30, 2] / 2,df.iloc[starting_index + 31, 2] / 2, 
          df.iloc[starting_index + 32, 2] / 2, df.iloc[starting_index + 33, 2] / 2, df.iloc[starting_index + 34, 2] / 2, df.iloc[starting_index + 35, 2] / 2, df.iloc[starting_index + 36, 2] / 2, df.iloc[starting_index + 37, 2] / 2, df.iloc[starting_index + 38, 2] / 2, df.iloc[starting_index + 39, 2] / 2,
          df.iloc[starting_index + 40, 2] / 2, df.iloc[starting_index + 41, 2] / 2, df.iloc[starting_index + 42, 2] / 2, df.iloc[starting_index + 43, 2] / 2, df.iloc[starting_index + 44, 2] / 2, df.iloc[starting_index + 45, 2] / 2, df.iloc[starting_index + 46, 2] / 2, df.iloc[starting_index + 47, 2] / 2,  
          df.iloc[starting_index + 48, 2] / 2, df.iloc[starting_index + 49, 2] / 2, df.iloc[starting_index + 50, 2] / 2, df.iloc[starting_index + 51, 2] / 2, df.iloc[starting_index + 52, 2] / 2, df.iloc[starting_index + 53, 2] / 2, df.iloc[starting_index + 54, 2] / 2, df.iloc[starting_index + 55, 2] / 2),
         (df.iloc[starting_index, 3] / 2,df.iloc[starting_index + 1, 3] / 2,df.iloc[starting_index + 2, 3] / 2,df.iloc[starting_index + 3, 3] / 2,df.iloc[starting_index + 4, 3] / 2,df.iloc[starting_index + 5, 3] / 2,df.iloc[starting_index + 6, 3] / 2,df.iloc[starting_index + 7, 3] / 2, 
          df.iloc[starting_index + 8, 3] / 2, df.iloc[starting_index + 9, 3] / 2, df.iloc[starting_index + 10, 3] / 2, df.iloc[starting_index + 11, 3] / 2, df.iloc[starting_index + 12, 3] / 2, df.iloc[starting_index + 13, 3] / 2, df.iloc[starting_index + 14, 3] / 2, df.iloc[starting_index + 15, 3] / 2,
          df.iloc[starting_index + 16, 3] / 2, df.iloc[starting_index + 17, 3] / 2, df.iloc[starting_index + 18, 3] / 2, df.iloc[starting_index + 19, 3] / 2, df.iloc[starting_index + 20, 3] / 2, df.iloc[starting_index + 21, 3] / 2, df.iloc[starting_index + 22, 3] / 2, df.iloc[starting_index + 23, 3] / 2,
          df.iloc[starting_index + 24, 3] / 2,df.iloc[starting_index + 25, 3] / 2,df.iloc[starting_index + 26, 3] / 2,df.iloc[starting_index + 27, 3] / 2,df.iloc[starting_index + 28, 3] / 2,df.iloc[starting_index + 29, 3] / 2,df.iloc[starting_index + 30, 3] / 2,df.iloc[starting_index + 31, 3] / 2, 
          df.iloc[starting_index + 32, 3] / 2, df.iloc[starting_index + 33, 3] / 2, df.iloc[starting_index + 34, 3] / 2, df.iloc[starting_index + 35, 3] / 2, df.iloc[starting_index + 36, 3] / 2, df.iloc[starting_index + 37, 3] / 2, df.iloc[starting_index + 38, 3] / 2, df.iloc[starting_index + 39, 3] / 2,
          df.iloc[starting_index + 40, 3] / 2, df.iloc[starting_index + 41, 3] / 2, df.iloc[starting_index + 42, 3] / 2, df.iloc[starting_index + 43, 3] / 2, df.iloc[starting_index + 44, 3] / 2, df.iloc[starting_index + 45, 3] / 2, df.iloc[starting_index + 46, 3] / 2, df.iloc[starting_index + 47, 3] / 2,  
          df.iloc[starting_index + 48, 3] / 2, df.iloc[starting_index + 49, 3] / 2, df.iloc[starting_index + 50, 3] / 2, df.iloc[starting_index + 51, 3] / 2, df.iloc[starting_index + 52, 3] / 2, df.iloc[starting_index + 53, 3] / 2, df.iloc[starting_index + 54, 3] / 2, df.iloc[starting_index + 55, 3] / 2),
         (df.iloc[starting_index, 4] / 2,df.iloc[starting_index + 1, 4] / 2,df.iloc[starting_index + 2, 4] / 2,df.iloc[starting_index + 3, 4] / 2,df.iloc[starting_index + 4, 4] / 2,df.iloc[starting_index + 5, 4] / 2,df.iloc[starting_index + 6, 4] / 2,df.iloc[starting_index + 7, 4] / 2, 
          df.iloc[starting_index + 8, 4] / 2, df.iloc[starting_index + 9, 4] / 2, df.iloc[starting_index + 10, 4] / 2, df.iloc[starting_index + 11, 4] / 2, df.iloc[starting_index + 12, 4] / 2, df.iloc[starting_index + 13, 4] / 2, df.iloc[starting_index + 14, 4] / 2, df.iloc[starting_index + 15, 4] / 2,
          df.iloc[starting_index + 16, 4] / 2, df.iloc[starting_index + 17, 4] / 2, df.iloc[starting_index + 18, 4] / 2, df.iloc[starting_index + 19, 4] / 2, df.iloc[starting_index + 20, 4] / 2, df.iloc[starting_index + 21, 4] / 2, df.iloc[starting_index + 22, 4] / 2, df.iloc[starting_index + 23, 4] / 2,
          df.iloc[starting_index + 24, 4] / 2,df.iloc[starting_index + 25, 4] / 2,df.iloc[starting_index + 26, 4] / 2,df.iloc[starting_index + 27, 4] / 2,df.iloc[starting_index + 28, 4] / 2,df.iloc[starting_index + 29, 4] / 2,df.iloc[starting_index + 30, 4] / 2,df.iloc[starting_index + 31, 4] / 2, 
          df.iloc[starting_index + 32, 4] / 2, df.iloc[starting_index + 33, 4] / 2, df.iloc[starting_index + 34, 4] / 2, df.iloc[starting_index + 35, 4] / 2, df.iloc[starting_index + 36, 4] / 2, df.iloc[starting_index + 37, 4] / 2, df.iloc[starting_index + 38, 4] / 2, df.iloc[starting_index + 39, 4] / 2,
          df.iloc[starting_index + 40, 4] / 2, df.iloc[starting_index + 41, 4] / 2, df.iloc[starting_index + 42, 4] / 2, df.iloc[starting_index + 43, 4] / 2, df.iloc[starting_index + 44, 4] / 2, df.iloc[starting_index + 45, 4] / 2, df.iloc[starting_index + 46, 4] / 2, df.iloc[starting_index + 47, 4] / 2,  
          df.iloc[starting_index + 48, 4] / 2, df.iloc[starting_index + 49, 4] / 2, df.iloc[starting_index + 50, 4] / 2, df.iloc[starting_index + 51, 4] / 2, df.iloc[starting_index + 52, 4] / 2, df.iloc[starting_index + 53, 4] / 2, df.iloc[starting_index + 54, 4] / 2, df.iloc[starting_index + 55, 4] / 2)))

    return data

def load_datasets(df):
    print("loading dataset")
    # 20% set aside for testing
    print("dataset size is", df.shape)
    num_items = (df.shape[0]) // 56
    batch_size = 2

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for i in range(num_items):

        starting_index = i * 56

        # 16 inputs
        data = parse_input(df, starting_index)

        # do encoding, go by index as shown below
        if 'elbow_lock' in df.iloc[starting_index, 5]:
            value = (0)
        elif 'hair' in df.iloc[starting_index, 5]:
            value = (1)
        elif 'pushback' in df.iloc[starting_index, 5]:
            value = (2)
        elif 'rocket' in df.iloc[starting_index, 5]:
            value = (3)
        elif 'scarecrow' in df.iloc[starting_index, 5]:
            value = (4)
        elif 'shouldershrug' in df.iloc[starting_index, 5]:
            value = (5)
        elif 'windows' in df.iloc[starting_index, 5]:
            value = (6)
        elif 'zigzag' in df.iloc[starting_index, 5]:
            value = (7)
        elif 'logout' in df.iloc[starting_index, 5]:
            value = (8)
        else:
            continue

        if i % 5 != 4: # training
            x_train_list.append(data)
            y_train_list.append(value) 
        else: # testing
            x_test_list.append(data)
            y_test_list.append(value)

    # remove extra inputs that cannot fit in batch_size
    while len(x_train_list) % batch_size != 0:
        x_train_list.pop()
        y_train_list.pop()
        
    while len(x_test_list) % batch_size != 0:
        x_test_list.pop()
        y_test_list.pop()
    
    print("parsed data, loading into DataLoaders for training and testing")
    
    # transform to PyTorch DataLoader
    tensor_x_train = torch.Tensor(x_train_list) # transform to torch tensor
    tensor_y_train = torch.Tensor(y_train_list)
        
    print("loaded training data into DataLoader:", tensor_x_train.size())

    dataset_train = TensorDataset(tensor_x_train,tensor_y_train)
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 1)

    tensor_x_test = torch.Tensor(x_test_list) # transform to torch tensor
    tensor_y_test = torch.Tensor(y_test_list)
        
    print("loaded training data into DataLoader:", tensor_x_test.size())

    dataset_test = TensorDataset(tensor_x_test,tensor_y_test)
    test_loader = DataLoader(dataset_test, batch_size = batch_size, shuffle = True, num_workers = 1)
    
    return train_loader, test_loader
