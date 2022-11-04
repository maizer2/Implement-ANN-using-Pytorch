import torch
import json

from typing import Dict, Optional, Callable, List, Tuple
from glob import glob
from PIL import Image

# -------------------------------------------------- #

def val_root(root: str, train: bool) -> str:
    if root[-1] != "/":
        root += "/"

    if train:
        root += "Training/"
    else:
        root += "Validation/"

    return root 

# -------------------------------------------------- #

def read_json(json_path: str) -> Dict:
    path = open(json_path, 'r')
    return json.load(path)
    
# -------------------------------------------------- #

class FP_and_WI(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: str, 
        train: bool = True,
        transform: Optional[Callable] = None
        ):

        # ../FP_and_WI/
        self.root = val_root(root, train)

        self.wearing_info = read_json(self.root + "labels/wearing_info.json")

        self.transform = transform

    def __len__(self):
        return len(self.wearing_info)

    def __getitem__(self, idx):
        
        #############################################
        # Get cloth information that fits the index #
        #############################################

        wearing_info = self.wearing_info[idx]
        
        ##########################################
        # Split a information using wearing_info #
        # 1. Get model_path_info                 #
        # 2. Split Keys and Values each variable #
        ##########################################

        model_path_info = wearing_info['wearing'].split('.')[0]
        item_name_info = list(wearing_info.keys())[1:]
        item_path_info = list(wearing_info.values())[1:]
        
        ########################################################
        #                      L A B E L                       #
        # Get model_info = [model_parse_info, model_pose_info] #
        # 1. Dict of model_parse_info get using model_pat_info #  
        # 2. Dict of model_pose_info get using model_path_info #
        # 3. Combine parse and pose to List                    #
        ########################################################

        model_parse_info = read_json(self.root + f"labels/Model-Parse_f/{model_path_info}.json")
        
        model_pose_info = read_json(self.root + f"labels/Model-Pose_f/{model_path_info}.json")

        model_info = [model_parse_info, model_pose_info]

        #############################################################
        # Get item_info = [item_parse_info, item_pose_info]         #
        # Using item_path_info but that is more than one value      #
        # 1. Loop using item_path_info each value                   #
        # 2. Some value is None, and then append None value to list #
        # 3. Read each json file and append value to list           #
        # 4. Combine parse and pose to List                         #
        #############################################################

        item_F_parse_info, item_B_parse_info = [], []
        item_F_pose_info, item_B_pose_info = [], []

        for item in item_path_info:
            if item != None:
                F_parse_json = read_json(self.root + f"labels/Item-Parse_f/{item}_F.json")
                item_F_parse_info.append(F_parse_json)

                B_parse_json = read_json(self.root + f"labels/Item-Parse_f/{item}_B.json")
                item_B_parse_info.append(B_parse_json)

                F_pose_json = read_json(self.root + f"labels/Item-Pose_f/{item}_F.json")
                item_F_pose_info.append(F_pose_json)

                B_pose_json = read_json(self.root + f"labels/Item-Pose_f/{item}_B.json")
                item_B_pose_info.append(B_pose_json)
            else:
                item_F_parse_info.append(None)
                item_B_parse_info.append(None)

                item_F_pose_info.append(None)
                item_B_pose_info.append(None)

        item_parse_info = [item_F_parse_info, item_B_parse_info]
        item_pose_info = [item_F_pose_info, item_B_pose_info]

        item_info = [item_name_info, [item_parse_info, item_pose_info]]
        
        ########################################
        # Get labels = [model_info, item_info] #
        ########################################

        label = [model_info, item_info]

        ######################################################
        #                     I M A G E                      #
        # Get Model and Item image                           #
        # 1. Get Model image                                 #
        # 2. Get Item image                                  #
        # 3. img = [transform_model_img, transform_item_img] #
        ######################################################

        ####################
        # Get model image  #
        # 1. Image.open    #
        # *. Image splin   #
        # 2. Transform img #
        ####################

        model_img = Image.open(self.root + f"images/Model-Image_deid/{model_path_info}.jpg").convert("RGB")
        
        ###########################################################################################################
        # Some of the images in the data set have a defect that is (1280, 720) size rather than (720, 1280) size. #
        # Image spin Using PIL.Image.transpose model                                                              #
        ###########################################################################################################

        if model_img.size == (1280, 720):
            model_img = model_img.transpose(Image.Transpose.ROTATE_270)

        if self.transform is not None:
            transform_model_img = self.transform(model_img)
        
        ###################################################
        # Get item image                                  #
        # Item image is more then one item                #
        # Item image has F(ront) and B(ehind)             #
        # 1. Loop using item_path_info each value         #
        # 2. Image.open                                   #
        # 3. Transform img                                #
        # 4. Combine each item to list transform_item_img #
        ###################################################

        transform_item_F_img, transform_item_B_img = [], []

        for item in item_path_info:
            if item is not None:
                item_F_img = Image.open(self.root + f"images/Item-Image/{item}_F.jpg")
                item_B_img = Image.open(self.root + f"images/Item-Image/{item}_B.jpg")
                transform_item_F_img.append(self.transform(item_F_img))
                transform_item_B_img.append(self.transform(item_B_img))
            else:
                transform_item_F_img.append(None)
                transform_item_B_img.append(None)

        transform_item_img = [transform_item_F_img, transform_item_B_img]

        ########################################################
        # Get img = [transform_model_img, transform_item_img] #
        ########################################################

        img = [transform_model_img, transform_item_img]

        return img, label