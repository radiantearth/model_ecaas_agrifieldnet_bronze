import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import glob
import rasterio
import pickle

from tqdm import tqdm
from matplotlib import pyplot as plt
from pylab import rcParams

from PIL import Image

input_dir      = "/opt/radiant/docker-solution/data/input" # "infer/input"
workspace_dir  = "/opt/radiant/docker-solution/workspace"
output_dir     = workspace_dir + "/images/train"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

output_dir     = workspace_dir + "/images/test"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


json_files = glob.glob(f"{input_dir}/**/*.json", recursive=True)
json_files = [f for f in json_files if "source" in f]

test_folder_ids = [f.split('_')[-1].split('.')[0] for f in json_files]
test_folder_ids.sort()

print(len(test_folder_ids), test_folder_ids[:3], test_folder_ids[-3:])


image_dir_pattern = input_dir + "/" + "source_fffff"
folder_dir_pattern = input_dir + "/" + "data_fffff"

image_channels = ["B04", "B02", "B03"] # blue (B2), green (B3), red (B4),
def get_image_from_folder_id(folder_id, max_value=150):
    image_dir_file = image_dir_pattern.replace("fffff", folder_id)
    
    
    img = []
    for c in image_channels:
        image_filename = f"{image_dir_file}/{c}.tif"
        with rasterio.open(image_filename) as src:
            field_data = src.read()[0]
            img.append(field_data)
            
    im = np.stack(img, axis=0)
    im = np.transpose(im, axes=(1, 2, 0)) # image shape as (H, W, D)

    #im = (((im-im.min())/(im.max()-im.min())*255)).astype(np.uint8)
    
    im = ((im/max_value)*255).astype(np.uint8)
    return im

def get_field_ids(folder_id):
    folder_dir_file = folder_dir_pattern.replace("fffff", folder_id)
    field_id_file = f"{folder_dir_file}/field_ids.tif"
    
    with rasterio.open(field_id_file) as src:
        field_data = src.read()[0]
        
    return field_data

def get_field_image(img, field_data, field_id, min_xy=32, max_xy=224): # 64/2, 256-min_xy # 224
    mask_index = np.argwhere(field_data == field_id)
    center = np.round(mask_index.mean(axis=0)).astype(int)
    center = np.clip(center, min_xy, max_xy)
    
    img_field = img[center[0]-min_xy:center[0]+min_xy, 
               center[1]-min_xy:center[1]+min_xy, 
               :]
    return img_field

def get_neighbour_images(img, field_data, field_id, min_xy=32, max_xy=224): # 64/2, 256-min_xy # 224
    
    mask_index = np.argwhere(field_data == field_id)
    center = np.round(mask_index.mean(axis=0)).astype(int)
    center = np.clip(center, min_xy, max_xy)
    
    north = center[0] - min_xy*2
    if north < min_xy:
        north = min_xy
    south = center[0] + min_xy*2
    if south > max_xy:
        south = max_xy
        
    west = center[1] - min_xy*2
    if west < min_xy:
        west = min_xy
    east = center[1] + min_xy*2
    if east > max_xy:
        east = max_xy
    
    
    img_field = img[center[0]-min_xy:center[0]+min_xy, 
                   center[1]-min_xy:center[1]+min_xy, 
                   :].copy()
    img_field_north = img[north-min_xy:north+min_xy, 
                   center[1]-min_xy:center[1]+min_xy, 
                   :].copy()
    img_field_south = img[south-min_xy:south+min_xy, 
                   center[1]-min_xy:center[1]+min_xy, 
                   :].copy()
    img_field_west = img[center[0]-min_xy:center[0]+min_xy, 
                   west-min_xy:west+min_xy, 
                   :].copy()
    img_field_east = img[center[0]-min_xy:center[0]+min_xy, 
                   east-min_xy:east+min_xy, 
                   :].copy()
    outputs = [img_field, img_field_north, img_field_south, img_field_west, img_field_east]
            
    return outputs

def process_folder(folder_id, output_dir="images", 
                   is_scale=True, verbose=True):    
    if verbose:
        print(f"Processing folder {folder_id} ...")
        
    img = get_image_from_folder_id(folder_id)
    field_data = get_field_ids(folder_id)

    field_ids = list(set(field_data.flatten()))
    field_ids = [f for f in field_ids if f > 0]
    field_ids.sort()
    

    for field_id in field_ids:
        
        arr_img_field = get_neighbour_images(img, field_data, field_id)

        for img_field, orientation in zip(arr_img_field, ["main", "north", "south", "west", "east"]):
            field_filename = f"{output_dir}/{folder_id}_{field_id}_{orientation}.png"
            im = img_field.copy()
            if is_scale:
                im = (((im-im.min())/(im.max()-im.min())*255)).astype(np.uint8)
            
            im = Image.fromarray(im)
            im.save(field_filename)

    

folder_id = "001c1"
process_folder(folder_id, output_dir=output_dir)

print("-"*40)
print(f"Processing images to {output_dir} ...")

folder_ids = test_folder_ids[:]
for folder_id in tqdm(folder_ids):
    process_folder(folder_id, output_dir=output_dir)
print("-"*40)