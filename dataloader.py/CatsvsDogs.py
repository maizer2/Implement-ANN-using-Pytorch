import torch

from typing import Any, Callable, List, Optional, Union, Tuple
from PIL import Image
from glob import glob

class CatsvsDogs(torch.utils.data.Dataset):
    def __init__(self, root: str, phase: str = 'train', transform: Optional[Callable] = None):
        # 데이터셋의 전처리를 해주는 부분
        '''
         Cats-vs-Dogs dataset download from https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

         root: str = "../Cats-vs-Dogs"
         phase: str = 'train' or 'test'
         transform: torchvision.transforms.Compose()
        '''

        def get_img_path(root: str):
            # "/data/DataSet/Cats-vs-Dogs/"
            if root[-1] == "/":
                root = root
            else:
                root = root + "/"

            dog_img_path = glob(root + "PetImages/Dog/*.jpg")
            cat_img_path = glob(root + "PetImages/Cat/*.jpg")

            assert phase == "test" or phase == "train", f"{phase} is not match. phase using train or test"

            if phase == "train":
                return dog_img_path[:int(len(dog_img_path) * 0.8)] + cat_img_path[:int(len(cat_img_path) * 0.8)]
            else:
                return dog_img_path[int(len(dog_img_path) * 0.8) :] + cat_img_path[:int(len(cat_img_path) * 0.8) :]

        self.transform = transform

        self.img_path = get_img_path(root)
        

    def __len__(self):
        # 데이터셋 길이, 총 샘플의 수를 리턴
        return len(self.img_path)

    def __getitem__(self, idx):
        # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
        img_path = self.img_path[idx]
        
        img = Image.open(img_path).convert("RGB")

        img_transformed = self.transform(img)

        label = img_path.split('/')[-2]

        return img_transformed, label
