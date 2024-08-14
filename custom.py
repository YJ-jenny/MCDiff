from torch.utils.data import Dataset
import torch
import numpy as np
from glob import glob
import nibabel as nib
import os
from torchvision import transforms
import nibabel as nib
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,root,train=False):
        self.files = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
        ])

        self.train = train
        if train == True:
            path = os.path.join(root,'train/datax_1/*')

            file_list = glob(path)
            if len(file_list) == 0:
                print("path wrong!")

            self.files += file_list
    
        
        else:
            path = os.path.join(root,'val/datax_1/*')

            file_list = glob(path)
            if len(file_list) == 0:
                print("path wrong!")

            self.files += file_list
                      
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):

        TR = []
        TE = []
        TI = []        
        
        t1_tr = float(450.0)
        t1_te = float(9.8)
        t1_ti = float(0.0)
        TR.append(t1_tr)
        TE.append(t1_te)
        TI.append(t1_ti)
            
        t2_tr = float(3760.0)
        t2_te = float(100.0)
        t2_ti = float(0.0)
        TR.append(t2_tr)
        TE.append(t2_te)
        TI.append(t2_ti)
        
        TR = np.array(TR)
        TE = np.array(TE)
        TI = np.array(TI)
        
        TR = torch.from_numpy(TR)
        TE = torch.from_numpy(TE)
        TI = torch.from_numpy(TI)
        TR = TR.float()
        TE = TE.float()
        TI = TI.float()

        file_x2 = self.files[i].replace('datax_1','datax_2')
        file_y = self.files[i].replace('datax_1','datay')
        
        data_x1 = nib.load(self.files[i]).get_fdata()
        data_x2 = nib.load(file_x2).get_fdata()
        data_y = nib.load(file_y).get_fdata()

        data_x1 = np.array(data_x1)
        data_x2 = np.array(data_x2)
        data_y = np.array(data_y)
        
        data_x1[data_x1<0]=0  
        data_x2[data_x2<0]=0  
        data_y[data_y<0]=0

        data_x1 = data_x1 *2 - 1 
        data_x2 = data_x2 *2 - 1 
        data_y = data_y *2 - 1 

        data_x1 = self.transform(data_x1)
        data_x2 = self.transform(data_x2)
        data_y = self.transform(data_y)

        dataset = torch.cat((data_y,data_x1,data_x2),dim=0)    
        dataset = dataset.float()
        return dataset, TR, TE, TI
    
    




