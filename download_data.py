import kagglehub
import os

repo_dir = os.path.dirname(__file__)
data_dir = os.path.join(repo_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# data downloads to KAGGLEHUB_CACHE by default; change to ./data
os.environ['KAGGLEHUB_CACHE'] = data_dir

# brain tumor segmentation dataset from
# https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset
path = kagglehub.dataset_download(
    "atikaakter11/brain-tumor-segmentation-dataset")
print('Path to brain tumor segmentation data:', path)

# TODO this is not working due to the file awaiting compression on Kaggle's end.
# liver tumor segmentation dataset from
# https://www.kaggle.com/datasets/ag3ntsp1d3rx/litsdataset2
# path = kagglehub.dataset_download("ag3ntsp1d3rx/litsdataset2")
# print("Path to liver tumor segmentation data:", path)

# TODO download COCO dataset
