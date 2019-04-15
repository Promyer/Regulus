import pandas as pd
from torch.utils.data import Dataset
import albumentations as alb
from albumentations import pytorch
import cv2


class FaceDataSet(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
            csv_file (string): path to total.csv
            root_dir (string): path to directory with images
                (empty str if total.csv contains full path)
            transform (callable, optional): optional transform to be applied to sample
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
        image_ = image = cv2.imread(img_name)
        image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.meta.age_cluster[idx]
        
        if (self.transform is None):
            self.transform = alb.Compose([
            alb.Resize(224, 224),
            alb.HorizontalFlip(),
            alb.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),
            alb.Blur(blur_limit=3, p=0.2),
            alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            alb.GaussNoise(p=0.1),
            alb.pytorch.ToTensor()
            ])
        augmented = self.transform(image=image_)
        image_ = augmented['image']
        
        if self.target_transform:
            target = self.target_transform(target)
            
        return (image_, target)
