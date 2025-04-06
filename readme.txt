Repo for semantic segmentation transfer learning.

Edison Bailey | ebailey60@gatech.edu
Nicole Cheetham | nicole.cheetham33@gmail.com
Cameron Goss | camwgoss@gmail.com
Grace Python | gcpython@gmail.com

1) Create a virtual environment. Anaconda example given here.
> conda create --name gt-net python
> conda activate gt-net

2) Install required Python libraries.
> pip install -r requirements.txt

3) Download and preprocess data. First navigate to /gt-net
> python preprocess_brain.py

4) Download and preprocess liver data. Assuming have previously downloaded and preprocessed brain data.
Given issues downloading the liver data directly from Kaggle within the code. We each need to download by hand. Navigate to https://www.kaggle.com/datasets/ag3ntsp1d3rx/litsdataset2/data?select=images and hit download icon in the images section (using download in the banner will fail). Once completed switch to 'masks' in the Data Explorer at right and repeat the process to download the masks.

Within the 'data' folder established from step 3, create a 'liver_tumor_segmentation' folder and then a 'raw' folder within the liver folder and save liver data in 'image' and 'mask' folders respectively.

5) (Optional as needed.) Update requirements.txt during development.
> pip freeze > requirements.txt
