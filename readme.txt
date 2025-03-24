Repo for semantic segmentation transfer learning.

Edison Bailey | ebailey60@gatech.edu
Nicole Cheetham | ncheetham3@gatech.edu
Cameron Goss | camwgoss@gmail.com
Grace Python | gcpython@gmail.com

1) Create a virtual environment. Anaconda example given here.
> conda create --name gt-net python
> conda activate gt-net

2) Install required Python libraries.
> pip install -r requirements.txt

3) Download data. First navigate to /gt-net .
> python download_data.py

4) (Optional as needed.) Update requirements.txt during development.
> pip freeze > requirements.txt
