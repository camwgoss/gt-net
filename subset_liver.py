import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil


repo_dir = os.path.dirname(__file__)
rel_data_dir = os.path.join(repo_dir,'data','liver_tumor_segmentation','subset')
os.makedirs(rel_data_dir, exist_ok=True)

rel_image_dir = os.path.join(rel_data_dir,'image')
os.makedirs(rel_image_dir, exist_ok=True)
rel_liver_im_dir = os.path.join(rel_image_dir,'0')
os.makedirs(rel_liver_im_dir, exist_ok=True)
rel_tumor_im_dir = os.path.join(rel_image_dir,'1')
os.makedirs(rel_tumor_im_dir, exist_ok=True)

rel_mask_dir = os.path.join(rel_data_dir,'mask')
os.makedirs(rel_mask_dir, exist_ok=True)
rel_liver_ma_dir = os.path.join(rel_mask_dir,'0')
os.makedirs(rel_liver_ma_dir, exist_ok=True)
rel_tumor_ma_dir = os.path.join(rel_mask_dir,'1')
os.makedirs(rel_tumor_ma_dir, exist_ok=True)


rel_liver = []
rel_tumor = []

image_dir = os.path.join(repo_dir,'data','liver_tumor_segmentation','raw','mask')
image_files = os.listdir(image_dir)

for image_file in image_files:
    image_path = os.path.join(image_dir,image_file)
    image = Image.open(image_dir+'\\'+image_file)
    image = np.array(image)
    distinct_val = len(np.unique(image))
    if distinct_val==2:
        rel_liver.append(image_file)
        save_path = os.path.join(rel_liver_ma_dir,image_file)
        shutil.copy2(image_path,save_path)
    elif distinct_val==3:
        rel_tumor.append(image_file)
        save_path = os.path.join(rel_tumor_ma_dir,image_file)
        shutil.copy2(image_path,save_path)

for image_file in rel_liver:
    image_path = os.path.join(repo_dir,'data','liver_tumor_segmentation','raw','image',image_file)
    save_path = os.path.join(rel_liver_im_dir,image_file)
    shutil.copy2(image_path,save_path)

for image_file in rel_tumor:
    # image_path = os.path.join(repo_dir,'data','liver_tumor_segmentation','image',image_file)
    save_path = os.path.join(rel_tumor_im_dir,image_file)
    shutil.copy2(image_path,save_path)
    
    