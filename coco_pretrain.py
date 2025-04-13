import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from pycocotools.coco import COCO
import os
import requests
from PIL import Image
import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


class COCOSegmentationDataset(Dataset):
    def __init__(self, coco_api, category_ids, transform=None, target_transform=None):
        """
        Args:
            coco_api: Initialized COCO API object
            category_ids: List of category IDs to use for segmentation
            transform: Optional transform for input images
            target_transform: Optional transform for masks
        """
        self.coco = coco_api
        self.transform = transform
        self.target_transform = target_transform
        self.category_ids = category_ids

        # Get image IDs that have annotations for our categories
        self.img_ids = []
        for cat_id in category_ids:
            cat_img_ids = self.coco.getImgIds(catIds=[cat_id])
            self.img_ids.extend(cat_img_ids)
        self.img_ids = list(set(self.img_ids))  # Remove duplicates

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']

        # Download image
        response = requests.get(img_url)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')

        # Create binary mask for the selected categories
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Initialize empty mask with image dimensions
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Fill mask with annotations
        for ann in anns:
            # For polygon annotations
            if type(ann['segmentation']) == list:
                # Convert polygons to binary mask
                pixel_mask = self.coco.annToMask(ann)
                mask = np.maximum(mask, pixel_mask)

        # Convert to PIL Image for transforms
        mask_img = Image.fromarray(mask)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask_img = self.target_transform(mask_img)

        return img, mask_img.float()  # Return image and mask

    def __len__(self):
        return len(self.img_ids)

def train_model(model, dataloader, criterion, optimizer, device, num_epoch=10):
    '''Train UNet model on COCO categories'''
    model.train()
    epoch_losses = []
    for epoch in range(num_epoch):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epoch}') # Print progress bar

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            # Calc loss
            loss = criterion(outputs, masks)

            # Backprop & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epoch}; Loss: {epoch_loss:.4f}')

    return epoch_losses

def pretrain_coco(coco_api_url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                    annotation_file='instances_train2017.json',
                    input_channels=1,
                    batch_size=8,
                    learning_rate=1e-4,
                    num_epochs=10,
                    save_path='unet_coco_pretrained.pth',
                    category_ids=None):
    '''
        Pretrain UNet on COCO dataset using segmentation_models_pytorch

        Args:
            coco_api_url: URL to COCO annotations
            annotation_file: Annotation file name within the zip
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            save_path: Path to save the pretrained model
    '''
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make sure annotations exist
    annotation_path = check_annotations(annotation_file)

    # Initialize COCO API
    coco = COCO(annotation_path)

    # Define transformations
    if input_channels == 1:
        # Grayscale
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226])
        ])
    else:
        # RGB
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # Create dataset
    dataset = COCOSegmentationDataset(coco, category_ids, transform, target_transform)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model from segmentation_models_pytorch
    if input_channels == 1:
        # For grayscale we need to modify the first layer of the encoder
        model = smp.Unet(
            encoder_name="resnet34",  # Choose encoder, resnet34 is a good default
            encoder_weights="imagenet",  # Use pre-trained weights
            in_channels=3,  # Initially 3 channels (will modify)
            classes=1,  # Binary segmentation
        )

        # Replace first conv layer to accept grayscale images
        # Get first conv layer weights
        first_conv_weights = model.encoder.conv1.weight.data

        # Create new layer with 1 input channel but keeping the same output channels
        new_conv = nn.Conv2d(
            1, model.encoder.conv1.out_channels,
            kernel_size=model.encoder.conv1.kernel_size,
            stride=model.encoder.conv1.stride,
            padding=model.encoder.conv1.padding,
            bias=False if model.encoder.conv1.bias is None else True
        )

        # Average the weights across the RGB channels
        new_conv.weight.data = torch.mean(first_conv_weights, dim=1, keepdim=True)

        # Replace the first conv layer
        model.encoder.conv1 = new_conv
    else:
        # For RGB, we can just use the model
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(f"Starting training for {num_epochs} epochs...")
    losses = train_model(model, dataloader, criterion, optimizer, device, num_epochs)

    # Create plots directory if it doesn't exist
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_path = os.path.join(model_dir, save_path)
    # Save model
    torch.save(model.state_dict(), save_path)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Create plots directory if it doesn't exist
    plot_dir = './plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_filename = 'coco_training_loss.png'

    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.close()  # Close the plot to free up memory

    return model

def check_annotations(annotation_file='instances_train2017.json'):
    '''Make sure COCO annotation file exists, download if needed'''
    annotation_path = os.path.join('annotations', annotation_file)

    if not os.path.exists(annotation_path):
        print(f"Annotation file not found. Downloading...")

        # Create annotations directory if it doesn't exist
        if not os.path.exists('annotations'):
            os.makedirs('annotations')

        # Download annotations zip
        coco_api_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        print(f"Downloading from {coco_api_url}...")

        response = requests.get(coco_api_url, stream=True)
        file_size = int(response.headers.get('content-length', 0))

        # Show progress bar while downloading
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading annotations")

        with open('annotations.zip', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        pbar.close()

        # Extract zip file
        import zipfile
        with zipfile.ZipFile('annotations.zip', 'r') as zip_ref:
            zip_ref.extractall('.')

        # Remove zip file
        os.remove('annotations.zip')
        print(f"Annotations successfully downloaded and extracted.")

    return annotation_path

def list_coco_categories():
    """Print available COCO categories to help with selection"""
    try:
        # Make sure annotations exist
        annotation_path = check_annotations()

        # Initialize COCO API
        coco = COCO(annotation_path)

        # Get all categories
        cats = coco.loadCats(coco.getCatIds())

        # Print categories
        print("Available COCO categories:")
        print("ID\tName\tSupercategory")
        print("-" * 30)

        for cat in cats:
            print(f"{cat['id']}\t{cat['name']}\t{cat['supercategory']}")

        return cats

    except Exception as e:
        print(f"Error listing categories: {e}")
        return None

if __name__ == "__main__":
    # Print available categories (limit training categories in coco to similar images to tumors)
    #list_coco_categories() # Uncomment if want to view and change selected categories
    categories = [1, 18, 25, 34, 51, 53, 57, 64, 75, 81]

    # pretrain UNet on COCO dataset
    pretrained_model = pretrain_coco(
        input_channels = 1,
        batch_size = 8,
        learning_rate = 0.0001,
        num_epochs = 10,
        save_path = 'unet_coco_pretrained.pth',
        category_ids=categories
    )