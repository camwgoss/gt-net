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

4) (Optional as needed.) Update requirements.txt during development.
> pip freeze > requirements.txt
