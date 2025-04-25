import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

from memory_efficient_loading import ChunkedLiverTumorDataset, create_dataloaders, examine_npz

class LiverTumorDataset(Dataset):
    def __init__(self, images, masks):
        """
        Args:
            images: Numpy array of processed images
            masks: Numpy array of processed masks
            transform: Optional transform to apply to images and masks
        """
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and mask
        image = self.images[idx]
        mask = self.masks[idx]

        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0) # size: (1, H, W)

        if len(mask.shape) == 3:
            mask = mask.squeeze(0)

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Normalize image values to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask

def load_processed_data(data_path):
    processed_data = np.load(data_path)
    print(f"Available keys in dataset: {list(processed_data.keys())}")

    # Extract data
    images_train = processed_data['images_train']
    labels_train = processed_data['labels_train']
    images_val = processed_data['images_val']
    labels_val = processed_data['labels_val']

    print(f"Training images shape: {images_train.shape}")
    print(f"Training labels shape: {labels_train.shape}")
    print(f"Validation images shape: {images_val.shape}")
    print(f"Validation labels shape: {labels_val.shape}")

    return images_train, labels_train, images_val, labels_val

def load_pretrained_model(pretrained_path, input_channels=1):
    # Initialize model with same architecture
    if input_channels == 1:
        # For grayscale
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1, # only need 1 class for binary segmentation (scale from 0-1, no tumor to tumor)
        )

        # Replace first conv layer for grayscale (changes in_channels from 3 to 1)
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
            classes=2,
        )

    # Load weights
    try:
        # First try to load full checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model state from checkpoint dictionary")
        else:
            # Just model weights
            model.load_state_dict(checkpoint)
            print("Loaded model weights directly")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing with random weights")
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
    for images, masks in progress_bar:
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        epoch_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix(loss=epoch_loss / batch_count)

        # Free up memory after each batch, otherwise might have memory issues
        del images, masks, outputs, loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_train_loss = epoch_loss / batch_count

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_train_loss

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            batch_count += 1

            # Free memory
            del images, masks, outputs

    avg_val_loss = val_loss / batch_count

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_val_loss

def train_model(model, data_path, criterion, optimizer, device,
                batch_size, num_epochs=5, downsample_factor=2, save_dir='./models', model_name='teacher_model'):
    """Train the teacher model on the liver dataset"""
    # Make sure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create dataloaders with memory efficiency
    train_loader, val_loader = create_dataloaders(
        data_path, batch_size=batch_size,
        downsample_factor=downsample_factor)

    # Create val_loader if None
    if val_loader is None:
        val_dataset = ChunkedLiverTumorDataset(
            data_path, subset='val', downsample_factor=downsample_factor)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2)

    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        avg_train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, num_epochs)

        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss = validate_model(
            model, val_loader, criterion, device)

        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Save model with both state dict and model architecture
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }

            torch.save(checkpoint, os.path.join(save_dir, f'{model_name}_best.pth'))
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')

        # Save model at regular intervals
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }

            torch.save(checkpoint, os.path.join(save_dir, f'{model_name}_epoch{epoch + 1}.pth'))
            print(f'Checkpoint saved at epoch {epoch + 1}')

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss_plot.png'))
    plt.close()

    # Save final model
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }

    torch.save(checkpoint, os.path.join(save_dir, f'{model_name}_final.pth'))
    print(f'Final model saved after {num_epochs} epochs')

    return model, train_losses, val_losses


def visualize_predictions(model, data_path, device, num_samples=8,
                          downsample_factor=2, save_path='./image/liver_predictions.png'):
    model.eval()

    # Create dataset just for visualization
    val_dataset = ChunkedLiverTumorDataset(
        data_path, subset='val', downsample_factor=downsample_factor)

    # Get sample indices
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    # Get samples
    samples = []
    with torch.no_grad():
        for idx in indices:
            image, mask = val_dataset[idx]
            image = image.to(device)

            # Predict
            output = model(image.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()

            # Add to samples
            samples.append((
                image.squeeze().cpu().numpy(),
                mask.squeeze().cpu().numpy(),
                prob
            ))

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, (image, mask, prob) in enumerate(samples):
        # Show original image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        # Show ground truth mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Show probability map
        im = axes[i, 2].imshow(prob, cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction (Probability)')
        axes[i, 2].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    ### TRAINING PARAMETERS ###
    batch_size = 1
    learning_rate = 0.00001
    num_epochs = 5
    downsample_factor = 4

    # set google drive parameter:
    use_gdrive = False
    save_dir = './models'
    image_dir = './image'

    if use_gdrive:
        try:
            from google.colab import drive
            # Mount Google Drive if not already mounted
            if not os.path.exists('/content/drive'):
                print("Mounting Google Drive...")
                drive.mount('/content/drive')

            # Set base directory in Google Drive
            print("Mapped to dl_group_assignment")
            gdrive_base = '/content/drive/MyDrive/dl_group_assignment'

            # Update save directory to Google Drive
            save_dir = os.path.join(gdrive_base, save_dir.lstrip('./'))
            image_dir = os.path.join(gdrive_base, image_dir.lstrip('./'))

            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            print(f"Models will be saved to: {save_dir}")

            # Look for processed_data.npz in Google Drive
            data_path = os.path.join(gdrive_base, 'data', 'liver_tumor_segmentation', 'processed_data_relBin.npz')

            # Look for the latest pretrained model in Google Drive
            model_dir = os.path.join(gdrive_base, 'models')
            if os.path.exists(model_dir):
                pth_files = [f for f in os.listdir(model_dir) if f.startswith('unet_coco') and f.endswith('.pth')]
                if pth_files:
                    # Sort by epoch number
                    pth_files.sort(key=lambda x: int(
                        ''.join(filter(str.isdigit, x.split('epoch')[-1]))) if 'epoch' in x else 0, reverse=True)
                    pretrained_model_path = os.path.join(model_dir, pth_files[0])
                else:
                    # Check checkpoints directory
                    checkpoint_dir = os.path.join(gdrive_base, 'checkpoints')
                    if os.path.exists(checkpoint_dir):
                        pth_files = [f for f in os.listdir(checkpoint_dir) if
                                     f.startswith('unet_coco') and f.endswith('.pth')]
                        if pth_files:
                            pth_files.sort(key=lambda x: int(
                                ''.join(filter(str.isdigit, x.split('epoch')[-1]))) if 'epoch' in x else 0,
                                           reverse=True)
                            pretrained_model_path = os.path.join(checkpoint_dir, pth_files[0])
        except ImportError:
            print("Google Colab integration failed.")
            use_gdrive = False

    if not use_gdrive:
        # Find the processed_data.npz file
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'liver_tumor_segmentation', 'processed_data_relBin.npz')

        # Look for the latest pretrained model
        model_dir = './models'
        if os.path.exists(model_dir):
            pth_files = [f for f in os.listdir(model_dir) if f.startswith('unet_coco') and f.endswith('.pth')]
            if pth_files:
                # Sort by epoch number (assuming filename format includes epoch)
                pth_files.sort(
                    key=lambda x: int(''.join(filter(str.isdigit, x.split('epoch')[-1]))) if 'epoch' in x else 0,
                    reverse=True)
                pretrained_model_path = os.path.join(model_dir, pth_files[0])
            else:
                # Check checkpoints directory
                checkpoint_dir = './checkpoints'
                if os.path.exists(checkpoint_dir):
                    pth_files = [f for f in os.listdir(checkpoint_dir) if
                                 f.startswith('unet_coco') and f.endswith('.pth')]
                    if pth_files:
                        pth_files.sort(key=lambda x: int(
                            ''.join(filter(str.isdigit, x.split('epoch')[-1]))) if 'epoch' in x else 0, reverse=True)
                        pretrained_model_path = os.path.join(checkpoint_dir, pth_files[0])

        # Print paths
    #print(f"Data path: {data_path}")
    print(f"Pretrained model path: {pretrained_model_path}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    examine_npz(data_path)

    '''# Load data
    #print(f"Loading data from: {data_path}")
    images_train, labels_train, images_val, labels_val = load_processed_data(data_path)

    # Create datasets
    train_dataset = LiverTumorDataset(images_train, labels_train)
    val_dataset = LiverTumorDataset(images_val, labels_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)'''

    # Load pretrained model
    model = load_pretrained_model(pretrained_model_path, input_channels=1)
    model.to(device)

    # Define loss function and optimizer
    criterion = DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train model
    model, train_losses, val_losses = train_model(
        model, data_path, criterion, optimizer, device,
        batch_size=batch_size, num_epochs=num_epochs, downsample_factor=downsample_factor, save_dir=save_dir
    )

    # Visualize predictions
    visualize_predictions(model, data_path, device, downsample_factor=downsample_factor,
                          save_path=os.path.join(image_dir, 'liver_predictions.png'))

    print("Training completed!")