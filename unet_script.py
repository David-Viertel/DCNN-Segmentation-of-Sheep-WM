# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scipy.io
import numpy as np

import os

from tqdm import tqdm
import matplotlib.pyplot as plt

import copy

###################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),            
            nn.ReLU(inplace=True),           
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)



class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p



class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, in_channels, n_class):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1)

        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)

        out = self.sigmoid(out)
        
        return out







#####################

class MatDataset_train(Dataset):
    def __init__(self, data_file, label_file):
        

        data = scipy.io.loadmat(data_file)['FO04_denoised']
        labels = scipy.io.loadmat(label_file)['FO04_label']


        self.data = torch.Tensor(data).permute(2, 1, 0)
        self.labels = torch.Tensor(labels).permute(2, 1, 0)
        

    def __getitem__(self, index):

        
        data = self.data[index].unsqueeze(0)
        label = self.labels[index].unsqueeze(0)
        return data, label

    def __len__(self):
        return self.data.shape[0]



class MatDataset_infer(Dataset):
    def __init__(self, image_file, label_file):
        image = scipy.io.loadmat(image_file)['FO05_denoised_inf']
        labels = scipy.io.loadmat(label_file)['FO05_label_inf']

        self.image = torch.Tensor(image).permute(2, 1, 0)
        self.labels = torch.Tensor(labels).permute(2, 1, 0)

    def __getitem__(self, index):
        image = self.image[index].unsqueeze(0)
        label = self.labels[index].unsqueeze(0)
        return image, label

    def __len__(self):
        return self.image.shape[0]




############### DICE COEFF


###########TRAINING



################################################
#INFERENCE


def infer(model, dataloader, device, output_path):
    #Load the image

    model.eval()
    
    output_list = []

    #Perform inference
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data[0].to(device)

            output = model(inputs)


            output = output.squeeze(1)
            output = (output > 0.5).float()
            
            output_list.append(output.cpu())


    output_total = torch.cat(output_list, dim=0)

    #Save the output
    output_total_numpy = output_total.cpu().numpy()
    output_permuted = np.transpose(output_total_numpy, (2,1,0))

    scipy.io.savemat(output_path, {'output': output_permuted})

#######################################################
#Dice Metric

def dice_metric(y_true, y_pred):

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred))



######################################################


def main():
    device = torch.device("cuda:0")
    model = UNet(in_channels=1, n_class=1)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    image_train_path = os.path.expandvars('$TMPDIR/FO04_denoised.mat')
    label_path = os.path.expandvars('$TMPDIR/FO04_label.mat')
    
    image_infer_path = os.path.expandvars('$TMPDIR/FO05_denoised_inf.mat')
    infer_label_path = os.path.expandvars('$TMPDIR/FO05_label_inf.mat')


    dataloader_train = DataLoader(MatDataset_train(image_train_path, label_path), batch_size=4, shuffle = True)
    dataloader_infer = DataLoader(MatDataset_infer(image_infer_path, infer_label_path), batch_size=4)

############################################################
#TRAINING + VAL with GRAPHING

#NOTE, CODE BELOW IS PLAGIARISED!!!!


    epoch_number = 300


    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    best_model_weights = None
    best_loss = 100000 #a high number
    patience = 10
   
    for epoch in tqdm(range(epoch_number)):  # loop over the dataset multiple times
        
        model.train()
        train_running_loss = 0
        train_running_dc = 0

        for i, data in enumerate(tqdm(dataloader_train, position=0, leave=True)):
        
            inputs, label = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            outputs_binary = (outputs > 0.5).float()

            dc = dice_metric(outputs_binary, label)
            loss = criterion(outputs, label)

            train_running_loss += loss.item()
            train_running_dc += dc.item()

            loss.backward()
            optimizer.step()


        train_loss = train_running_loss / (i+1)
        train_dc = train_running_dc / (i+1)
               
        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        ###VAL
        
        model.eval()
        val_running_loss = 0
        val_running_dc = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader_infer, position=0, leave=True)):
                        
                inputs, label = data[0].to(device), data[1].to(device)

                outputs = model(inputs)

                outputs_binary = (outputs > 0.5).float()

                dc = dice_metric(outputs_binary, label)
                loss = criterion(outputs, label)

                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (i+1)
            val_dc  = val_running_dc / (i+1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)


        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print(" ")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        print("-" * 30)

        #code for early stopping
        if epoch > 2:
            if val_loss < best_loss:
                print(patience)
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict()) 
                patience = 45
            else:
                patience = patience - 1
                if patience == 0:
                    break


        print("max epoch is: ", val_dcs.index(max(val_dcs)), ", at: ", max(val_dcs))
        print("max epoch is: ", val_losses.index(min(val_losses)), ", at: ", min(val_losses))
        print("\n")

    

  
############################################################
   
    model.load_state_dict(best_model_weights)

    save_path = os.path.expandvars('$TMPDIR/unet_model_dn_f4.pth')
    torch.save(model.state_dict(), save_path)


    output_path = os.path.expandvars('$TMPDIR/output_unet_dn_f4.mat')


    dataloader_infer = DataLoader(MatDataset_infer(image_infer_path, infer_label_path), batch_size=1)
    infer(model, dataloader_infer, device, output_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

if __name__ == '__main__':
    main()
