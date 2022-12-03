import torch
import torchvision.transforms as transforms
import json

from typing import Dict, Optional, Callable, List, Tuple, Union
from PIL import Image, ImageDraw

# -------------------------------------------------- #

def val_item(item: str) -> str:

    if not (item == "uppor" or item == "pants"):
        raise ValueError("The input value must be uppor or pants")
    else:
        return item

# -------------------------------------------------- #

def val_item_code(item_code: Optional[Union[int, List[int]]] = None) -> List[int]:

    if isinstance(item_code, int):
        item_code = list(item_code)
    
    return item_code

# -------------------------------------------------- #

def val_root(root: str, train: bool) -> str:

    if root[-1] != "/":
        root += "/"

    if train:
        root += "Training/"
    else:
        root += "Validation/"

    return root 

# --------------------------------------------------- #

def reshape_3_to_1(list: List) -> List:

    reshape_list = []
    for i in range(len(list)):
        for j in range(len(list[i])):
            for k in range(len(list[i][j])):

                reshape_list.append(list[i][j][k])
    
    return reshape_list

# -------------------------------------------------- #

def get_json(json_path: str) -> Dict:

    path = open(json_path, 'r')
    return json.load(path)

# --------------------------------------------------- #

def get_item_code(item: Optional[str] = None) -> List:
    # model seg
    # 0: hair  1: face  2: neck  3: hat  4: outer_rsleeve  5: outer_lsleeve  6: outer_torso  7: inner_rsleeve
    # 8: inner_lsleeve  9: inner_torso  10: pants_hip  11: pants_rsleeve  12: pants_lsleeve  13: skirt
    # 14: right_arm  15: left_arm  16: right_shoe  17: left_shoe  18: right_leg  19: left_leg

    if item == "uppor":
        return [4, 5, 6, 7, 8, 9]

    elif item == "pants":
        return [10, 11, 12, 13]

# -------------------------------------------------- #

def get_part_point(json_file: Dict, item: str) -> List:

    item_codes = val_item_code(get_item_code(item))

    part_seg_list = []
    for item_code in item_codes:
        for idx in range(1, len(json_file) - 1):
            if json_file[f"region{idx}"]["category_id"] == item_code:
                part_seg_list.append(json_file[f"region{idx}"]["segmentation"])
                break
            else:
                continue

    return part_seg_list

# -------------------------------------------------- #

def get_side_point(point_list: List) -> List:
    
    add_value = 10

    low_row = 0
    low_col = 0
    high_row = 0
    high_col = 0

    for point in point_list:
        if low_row == 0 and low_col == 0 and high_row == 0 and high_col == 0:
            # print(point)
            low_row, low_col = point
            high_row, high_col = point

        else:
            if point[0] > high_row:
                # print(f"high_row : {high_row} -> {point[0]}")
                high_row = point[0]

            elif point[0] < low_row:
                # print(f"low_row : {high_row} -> {point[0]}")
                low_row = point[0]
            
            if point[1] > high_col:
                # print(f"high_col : {high_col} -> {point[1]}")
                high_col = point[1]
    
            elif point[1] < low_col:
                # print(f"low_col : {low_col} -> {point[1]}")
                low_col = point[1]

    low_row -= add_value
    low_col -= add_value
    high_row += add_value
    high_col += add_value

    box_point = [(int(low_row), int(low_col)), (int(high_row), int(high_col))]

    return box_point

# ------------------------------------------------------------------------------------------------------------

def get_img_crop(image: Image, box_point: List):
    image = image.copy()

    box = (box_point[0][0], box_point[0][1], box_point[1][0], box_point[1][1])
    cropped_img = image.crop(box=box)

    return cropped_img

# -------------------------------------------------- #

def draw_point(draw: ImageDraw.Draw, json_file: Optional[Dict]):

    for idx in range(0, len(json_file["landmarks"]), 3):
        if json_file['landmarks'][idx] != 0:
            draw.point((json_file['landmarks'][idx], json_file['landmarks'][idx + 1]), 255)
        
# -------------------------------------------------- #

def draw_polygon(draw: ImageDraw.Draw, json_file: Dict, item: Optional[str] = None):

    item_codes = val_item_code(get_item_code(item))

    if item_codes is not None:
        for item_code in item_codes:
            for idx in range(1, len(json_file) - 1):
                if json_file[f"region{idx}"]["category_id"] == item_code:
                    draw.polygon(reshape_3_to_1(json_file[f"region{idx}"]["segmentation"]), json_file[f"region{idx}"]["category_id"] + 10)
                    break
                else:
                    continue

    elif item_codes is None:
        for idx in range(1, len(json_file) - 1):
            draw.polygon(reshape_3_to_1(json_file[f"region{idx}"]["segmentation"]), json_file[f"region{idx}"]["category_id"] + 10)

# -------------------------------------------------- #

def get_model_img(root: str, file_path: str):
    ####################
    # Get model image  #
    # 1. Image.open    #
    # *. Image splin   #
    # 2. Transform img #
    ####################

    model_img = Image.open(root + f"images/Model-Image_deid/{file_path}.jpg").convert("RGB")
    
    ###########################################################################################################
    # Some of the images in the data set have a defect that is (1280, 720) size rather than (720, 1280) size. #
    # Image spin Using PIL.Image.transpose model                                                              #
    ###########################################################################################################

    if model_img.size == (1280, 720):
        model_img = model_img.transpose(Image.Transpose.ROTATE_270)

    return model_img

# -------------------------------------------------- #

def get_model_parse_img(root: str, file_path: str):

    ################################
    # Get model_parse_img          #
    # 1. Image.new                 #
    # 2. get parse info            #
    # 3. draw polygon on new Image #
    ################################

    model_parse_img = Image.new("L", (720, 1280), 0)
    draw = ImageDraw.Draw(model_parse_img)

    model_parse_info = get_json(root + f"labels/Model-Parse_f/{file_path}.json")
    draw_polygon(draw, model_parse_info)
    
    return model_parse_img

# -------------------------------------------------- #

def get_model_pose_img(root: str, file_path: str):

    ##############################
    # Get model_pose_img         #
    # 1. Image.new               #
    # 2. get pose info           #
    # 3. draw point on new Image #
    ##############################

    model_pose_img = Image.new("L", (720, 1280), 0)
    draw = ImageDraw.Draw(model_pose_img)

    model_pose_info = get_json(root + f"labels/Model-Pose_f/{file_path}.json")
    draw_point(draw, model_pose_info)

    return model_pose_img

# -------------------------------------------------- #

def get_model_part_img(root: str, file_path: str, item: str, model_img: Image):

    ###################################
    # Get model_part_img              #
    # 상의 기준                        #
    # 1. item에 맞는 segmentation 얻기 #
    # 2. 상의 boxpoint찾기             #
    # 3. boxpoint에 맞게 이미지 자르기 #
    # 4. 흰 배경 만들기                #   
    # 5. 흰 배경에 자른 이미지 붙여넣기 #

    ####################################
    model_parse_info = get_json(root + f"labels/Model-Parse_f/{file_path}.json")

    part_point_list = reshape_3_to_1(get_part_point(model_parse_info, item))
    
    part_side_point_list = get_side_point(part_point_list)

    model_part_crop_img = get_img_crop(model_img.copy(), part_side_point_list)

    model_part_img = Image.new("RGB", (720, 1280), "white")

    model_part_img.paste(model_part_crop_img, part_side_point_list[0])

    return model_part_img

# -------------------------------------------------- #

class FP_and_WI(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: str, 
        item: str = "uppor",
        train: bool = True,
        transform: Optional[Callable] = None
        ):

        # ../FP_and_WI/
        self.root = val_root(root, train)

        self.wearing_info = get_json(self.root + "labels/wearing_info.json")

        self.item = val_item(item)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
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
        
        ##########################
        #        M O D E L       #
        # 1. Get model_img       #  
        # 2. Get model_parse_img #
        # 3. Get model_pose_img  #
        # 3. Get model_part_img  #
        ##########################

        model_img = get_model_img(self.root, model_path_info)
        model_parse_img = get_model_parse_img(self.root, model_path_info)
        model_pose_img = get_model_pose_img(self.root, model_path_info)
        model_part_img = get_model_part_img(self.root, model_path_info, self.item, model_img)

        return {
            "Model" : self.transform(model_img),
            "Parse" : self.transform(model_parse_img),
            "Pose" : self.transform(model_pose_img),
            "Part" : self.transform(model_part_img)
        }