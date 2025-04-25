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
import json

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
            if type(ann['segmentation']) == list:
                pixel_mask = self.coco.annToMask(ann)
                mask = np.maximum(mask, pixel_mask)

        # Convert to PIL Image for transforms
        mask_img = Image.fromarray(mask)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask_img = self.target_transform(mask_img)

        return img, mask_img.float()
    def __len__(self):
        return len(self.img_ids)

def train_model(model, dataloader, criterion, optimizer, device, start_epoch=0, num_epochs=10,
                checkpoint_dir='./checkpoints', model_prefix='unet_coco', save_every=1):
    '''Train UNet model on COCO categories'''
    model.train()
    epoch_losses = []

    # Create checkpoints directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{start_epoch + num_epochs}')

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
        print(f'Epoch {epoch + 1}/{start_epoch + num_epochs}; Loss: {epoch_loss:.4f}')

        # Save checkpoint after each epoch or at specified interval
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_prefix}_epoch{epoch + 1}.pth')

            # Save model state and optimizer state for proper resuming
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'all_losses': epoch_losses
            }

            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

            # Also save a metadata file to track progress
            metadata_path = os.path.join(checkpoint_dir, f'{model_prefix}_metadata.json')
            metadata = {
                'latest_epoch': epoch + 1,
                'total_epochs': start_epoch + num_epochs,
                'losses': epoch_losses
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

    return epoch_losses

def pretrain_coco(coco_api_url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                  annotation_file='instances_train2017.json',
                  input_channels=1,
                  batch_size=8,
                  learning_rate=1e-4,
                  num_epochs=1,
                  checkpoint_dir='./checkpoints',
                  model_prefix='unet_coco',
                  category_ids=None,
                  resume_from=None):
    '''
        Pretrain UNet on COCO dataset using segmentation_models_pytorch with checkpointing

        Args:
            coco_api_url: URL to COCO annotations
            annotation_file: Annotation file name within the zip
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs per run
            checkpoint_dir: Directory to save checkpoints
            model_prefix: Prefix for saved model files
            category_ids: Category IDs to use for segmentation
            resume_from: Path to checkpoint to resume from (if None, start fresh)
    '''
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

    # Initialize model or load from checkpoint
    start_epoch = 0
    all_losses = []

    # Check if we're resuming from checkpoint
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Loading model from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)

            # Initialize a fresh model first
            if input_channels == 1:
                # For grayscale
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights="imagenet",
                    in_channels=3,  # Initially 3 channels (will modify)
                    classes=1,
                )

                # Replace first conv layer for grayscale
                first_conv_weights = model.encoder.conv1.weight.data
                new_conv = nn.Conv2d(
                    1, model.encoder.conv1.out_channels,
                    kernel_size=model.encoder.conv1.kernel_size,
                    stride=model.encoder.conv1.stride,
                    padding=model.encoder.conv1.padding,
                    bias=False if model.encoder.conv1.bias is None else True
                )
                new_conv.weight.data = torch.mean(first_conv_weights, dim=1, keepdim=True)
                model.encoder.conv1 = new_conv
            else:
                # For RGB
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                )

            # Load saved weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            # Initialize optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint['epoch']

            # Load previous losses if available
            if 'all_losses' in checkpoint:
                all_losses = checkpoint['all_losses']

            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Checkpoint file not found: {resume_from}")
            print("Starting from scratch...")
            resume_from = None

    # If not resuming, create a new model
    if not resume_from:
        if input_channels == 1:
            # For grayscale we need to modify the first layer of the encoder
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,  # Initially 3 channels
                classes=1,
            )

            # Replace first conv layer to accept grayscale images
            first_conv_weights = model.encoder.conv1.weight.data
            new_conv = nn.Conv2d(
                1, model.encoder.conv1.out_channels,
                kernel_size=model.encoder.conv1.kernel_size,
                stride=model.encoder.conv1.stride,
                padding=model.encoder.conv1.padding,
                bias=False if model.encoder.conv1.bias is None else True
            )
            new_conv.weight.data = torch.mean(first_conv_weights, dim=1, keepdim=True)
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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Train model
    print(f"Starting training from epoch {start_epoch + 1} for {num_epochs} epochs...")
    epoch_losses = train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        model_prefix=model_prefix
    )

    # Combine previous losses with new ones for plotting
    all_losses.extend(epoch_losses)

    # Plot all training losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_losses) + 1), all_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Create plots directory if it doesn't exist
    plot_dir = './plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Save loss plot
    current_epoch = start_epoch + num_epochs
    plot_filename = f'coco_training_loss_epoch{current_epoch}.png'
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.close()

    # Final save location
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    final_save_path = os.path.join(model_dir, f'{model_prefix}_epoch{current_epoch}.pth')

    # Save final model state for this run
    torch.save(model.state_dict(), final_save_path)

    print(f"Training completed to epoch {current_epoch}.")
    print(f"Model saved to {final_save_path}")
    print(
        f"To continue training, run with: resume_from='{os.path.join(checkpoint_dir, f'{model_prefix}_epoch{current_epoch}.pth')}'")

    return model

def check_annotations(annotation_file='instances_train2017.json'):
    # For google colab
    #base_dir = '/dl_group_assignment'
    #annotation_path = os.path.join(base_dir, 'annotations', annotation_file)
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

def find_latest_checkpoint(checkpoint_dir='./checkpoints', model_prefix='unet_coco'):
    """Find the latest checkpoint based on epoch number in the filename"""
    if not os.path.exists(checkpoint_dir):
        return None

    # Check for metadata file first
    metadata_path = os.path.join(checkpoint_dir, f'{model_prefix}_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            latest_epoch = metadata.get('latest_epoch', 0)
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_prefix}_epoch{latest_epoch}.pth')
            if os.path.exists(checkpoint_path):
                return checkpoint_path
        except:
            pass

    # Fall back to checking filenames
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith(model_prefix) and f.endswith('.pth')]

    if not checkpoints:
        return None

    # Extract epoch numbers from filenames
    epoch_numbers = []
    for checkpoint in checkpoints:
        try:
            epoch = int(checkpoint.split('epoch')[1].split('.')[0])
            epoch_numbers.append((epoch, checkpoint))
        except:
            continue

    if not epoch_numbers:
        return None

    # Get the checkpoint with the highest epoch number
    latest_epoch, latest_checkpoint = max(epoch_numbers, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest_checkpoint)

if __name__ == "__main__":
    # Print available categories (limit training categories in coco to similar images to tumors)
    #list_coco_categories() # Uncomment if want to view and change selected categories
    categories = [1, 18, 25, 34, 51, 53, 57, 64, 75, 81]

    # Define checkpoint directory and model prefix
    checkpoint_dir = './checkpoints'
    model_prefix = 'unet_coco'

    # Find latest checkpoint if it exists
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, model_prefix)

    if latest_checkpoint:
        print(f"Found latest checkpoint: {latest_checkpoint}")
        resume_training = input(f"Resume training from checkpoint? (y/n): ").lower().strip() == 'y'

        if resume_training:
            # Resume training from the latest checkpoint
            pretrained_model = pretrain_coco(
                input_channels=1,
                batch_size=8,
                learning_rate=0.0001,
                num_epochs=1,  # Train for 1 epoch at a time
                checkpoint_dir=checkpoint_dir,
                model_prefix=model_prefix,
                category_ids=categories,
                resume_from=latest_checkpoint
            )
        else:
            # Start fresh training
            pretrained_model = pretrain_coco(
                input_channels=1,
                batch_size=8,
                learning_rate=0.0001,
                num_epochs=1,
                checkpoint_dir=checkpoint_dir,
                model_prefix=model_prefix,
                category_ids=categories
            )
    else:
        print("No existing checkpoints found. Starting fresh training.")
        # Start fresh training
        pretrained_model = pretrain_coco(
            input_channels=1,
            batch_size=8,
            learning_rate=0.0001,
            num_epochs=1,
            checkpoint_dir=checkpoint_dir,
            model_prefix=model_prefix,
            category_ids=categories
        )