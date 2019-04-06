import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class DataLoader(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
            csv_file (string): path to total.csv
            root_dir (string): path to directory with images
                (empty if total.csv contains full path)
            transform (callable, optional): optional transform to be applied to sample
                (convert image to torch.Tensor by default)
            target_transform (callable, optional): optional transform to be applied to target
            
        """
        self.meta = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        img_name = self.root_dir + self.meta.urls[idx]
        image = io.imread(img_name)
        pic = Image.open(img_name)
        coords_str = self.meta.face_coords[idx]
        #coords has string type since saved as csv
        coords = [float(x) for x in coords_str[1:-1].split()]
        target = self.meta.age_cluster[idx]
        #sample = {'pic': pic, 'image': image, 'coords': coords}
        sample = image
      
        if self.transform:
            sample = self.transform(sample)
        else:
            to_tens = transforms.ToTensor()
            sample = to_tens(sample)
            
        if self.target_transform:
            target = self.target_transform(target)
            
        return (sample, target)