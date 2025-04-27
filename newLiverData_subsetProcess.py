import matplotlib.pyplot as plt
import nibabel as nib
import os
import utils.image_processing as ip
import utils.split_data as split_data
import numpy as np
import matplotlib
import re
import gc
import pickle

# %matplotlib inline
# matplotlib.use('Qt5Agg')

def load_processed_data():
    '''
    Returns:
        images_train, masks_train, images_eval, masks_eval, images_test, masks_test:
        List of Numpy arrays containing image and mask data, (row, column, [channel]).
    '''

    repo_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        repo_dir, 'data', 'processed_data.npz')
        # repo_dir, 'data', 'liver_tumor_segmentation', 'trial','processed_data.npz')
    processed_data = np.load(data_path)
    return processed_data

def _save_processed_data(images_train, masks_train,
                         images_val, masks_val,
                         images_test, masks_test):

    processed_data = {'images_train': images_train, 'labels_train': masks_train,
                      'images_val': images_val, 'labels_val': masks_val,
                      'images_test': images_test, 'labels_test': masks_test}

    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', 'processed_data.npz')
        # repo_dir, 'data', 'liver_tumor_segmentation','trial', 'processed_data.npz')

    np.savez(save_path, **processed_data)
    
def get_mask_slices(img_file,root,ext):
    masks_raw=[]
    mask_files = []
    img = nib.load(img_file)
    try:
        data = img.get_fdata()
        i = 0
        while i<data.shape[2]:
            masks_raw.append(data[:,:,i])
            mask_fil = root+'_'+str(i)+ext
            mask_files.append(mask_fil)
            i+=1
    except:
        print(root)
    return masks_raw#,mask_files

def get_image_slices(img_file,root,ext):
    images_raw = []
    image_files = []
    img = nib.load(img_file)
    try:
        data = img.get_fdata()
        i = 0
        while i <data.shape[2]:
            images_raw.append(data[:,:,i])
            image_fil = root+'_'+str(i)+ext
            image_files.append(image_fil)
            i+=1
    except:
        print(root)
        image_files.append(img_file)
    return images_raw#,image_files


def get_subset(images,masks,portion=0.1):
    num_samples = int(len(images)*portion)
    samples = np.random.choice(len(images),num_samples,replace=False)
    for sample in samples:
        images_subset = [images[ss] for ss in samples]
        masks_subset = [masks[ss] for ss in samples]
        
    return images_subset,masks_subset

#%%

## code below was run on various subsets of the available images/masks in order to find the relevant slices
## and subset to a manageable set of data.

repo_dir = "C:\\Users\\gcpyt\\Documents\\gt-net-main"
raw_path = os.path.join(repo_dir,'data','liver_tumor_segmentation','raw')

#get images/masks
images_raw = []
masks_raw = []

pat_nums = np.arange(120,131,1)
for pat_num in pat_nums:
    mask_file = os.path.join(raw_path,'segmentation-'+str(pat_num)+'.nii')
    img_file = os.path.join(raw_path,'volume-'+str(pat_num)+'.nii')
    img = nib.load(img_file)
    img_data = img.get_fdata()
    del img
    gc.collect()
    msk = nib.load(mask_file)
    msk_data = msk.get_fdata()
    del msk
    gc.collect()
    i=0
    while i < msk_data.shape[2]:
        if msk_data[:,:,i].any()>0:
            masks_raw.append(msk_data[:,:,i])
            images_raw.append(img_data[:,:,i])
            i+=1
        else:
            i+=1
    del img_data, msk_data
    gc.collect()
     
# Pickling the data
with open('rel_120-130.pkl', 'wb') as f:
    pickle.dump(images_raw, f)
    pickle.dump(masks_raw,f)

#%%

## code below was run on various subsets of the available images/masks in order to find the non-relevant slices
## and subset to a manageable set of data.

repo_dir = "C:\\Users\\gcpyt\\Documents\\gt-net-main"
raw_path = os.path.join(repo_dir,'data','liver_tumor_segmentation','raw')

#get images/masks
images_raw = []
masks_raw = []

pat_nums = np.arange(120,131,1)
for pat_num in pat_nums:
    mask_file = os.path.join(raw_path,'segmentation-'+str(pat_num)+'.nii')
    img_file = os.path.join(raw_path,'volume-'+str(pat_num)+'.nii')
    img = nib.load(img_file)
    img_data = img.get_fdata()
    del img
    gc.collect()
    msk = nib.load(mask_file)
    msk_data = msk.get_fdata()
    del msk
    gc.collect()
    i=0
    while i < msk_data.shape[2]:
        if msk_data[:,:,i].any()>0:
            i+=1
        else:
            masks_raw.append(msk_data[:,:,i])
            images_raw.append(img_data[:,:,i])
            i+=1
    del img_data, msk_data
    gc.collect()
     
# Pickling the data
with open('nonrel_120-130.pkl', 'wb') as f:
    pickle.dump(images_raw, f)
    pickle.dump(masks_raw,f)
#%%

import pickle
import gc

masks_raw = []
images_raw = []

# Unpickling the dictionary
with open('rel_0-19.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()
    

with open('rel_20-29.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('rel_30-49.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('rel_50-79.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('rel_80-89.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('rel_90-99.pkl', 'rb') as f: #come back
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('rel_100-109.pkl', 'rb') as f: #come back
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('rel_110-119.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('rel_120-130.pkl', 'rb') as f: #1840 #come back
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()
del f,images_sub,masks_sub
gc.collect()

with open('rel_10percent.pkl', 'wb') as f:
    pickle.dump(images_raw, f)
    pickle.dump(masks_raw,f)

#%%

import pickle
import gc

masks_raw = []
images_raw = []

# Unpickling the dictionary
with open('nonrel_0-9.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()
    
with open('nonrel_10-19.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_20-29.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_30-39.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()
    
with open('nonrel_40-49.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_50-59.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()
    
with open('nonrel_60-69.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()
    
with open('nonrel_70-79.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_80-89.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_90-99.pkl', 'rb') as f: #come back
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_100-109.pkl', 'rb') as f: #come back
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_110-119.pkl', 'rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()

with open('nonrel_120-130.pkl', 'rb') as f: #1840 #come back
    images = pickle.load(f)
    masks = pickle.load(f)
    images_sub, masks_sub = get_subset(images,masks,portion=0.1)   
    masks_raw.extend(masks_sub)
    images_raw.extend(images_sub)
    del masks,images
    gc.collect()
del f,images_sub,masks_sub
gc.collect()

with open('nonrel_10percent.pkl', 'wb') as f:
    pickle.dump(images_raw, f)
    pickle.dump(masks_raw,f)

#%%

import pickle
import gc

images_raw = []
masks_raw = []

with open('rel_10percent.pkl','rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_raw.extend(images)
    masks_raw.extend(masks)
    del masks,images
    gc.collect()
    
with open('nonrel_10percent.pkl','rb') as f:
    images = pickle.load(f)
    masks = pickle.load(f)
    images_raw.extend(images)
    masks_raw.extend(masks)
    del masks,images
    gc.collect()

with open('full_10percent.pkl','wb') as f:
    pickle.dump(images_raw,f)
    pickle.dump(masks_raw,f)
    
    #%%
    
repo_dir = "C:\\Users\\gcpyt\\Documents\\gt-net-main"
raw_path = os.path.join(repo_dir,'data','liver_tumor_segmentation','raw')

count0 = 0
count2 = 0
count1 = 0

#get images/masks
images_raw = []
masks_raw = []

pat_nums = np.arange(0,131,1)
for pat_num in pat_nums:
    mask_file = os.path.join(raw_path,'segmentation-'+str(pat_num)+'.nii')
    # img_file = os.path.join(raw_path,'volume-'+str(pat_num)+'.nii')
    # img = nib.load(img_file)
    # img_data = img.get_fdata()
    # del img
    # gc.collect()
    msk = nib.load(mask_file)
    msk_data = msk.get_fdata()
    del msk
    gc.collect()
    i=0
    while i < msk_data.shape[2]:
        if len(np.unique(msk_data[:,:,i]))==3:
            count2+=1
        elif len(np.unique(msk_data[:,:,i]))==2:
            count1+=1
        else:
            count0+=1
        i+=1
        #%%
        if msk_data[:,:,i].any()>0:
            if msk_data[:,:,i].any()==2:
                count2+=1
            else:
                count1+=1
            i+=1
        else:
            count0+=1
            i+=1
    # del msk_data
    # gc.collect()

