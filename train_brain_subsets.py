import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import random
import json
import utils.image_processing as ip

class BrainTumorDataset(Dataset):
    def __init__(self, images, masks):
        '''
        Args:
            images: numpy array of images
            masks: numpy array of masks (labels)
        '''
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # Add channel dimension if needed (for grayscale images)
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        # Add channel dimension for image if needed
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        # Normalize image to [0, 1] range
        image = image / 255.0

        # Convert multi-class mask to binary mask (0: no tumor, 1: any tumor)
        # Classes 1, 2, 3 are all different types of tumors, combine them into class 1
        binary_mask = (mask > 0).long()

        # Convert to one-hot encoding
        mask = F.one_hot(binary_mask, num_classes=2).permute(2, 0, 1).float()

        return image, mask

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(mode='binary')
    def forward(self, student_outputs, teacher_outputs, targets):
        # Standard model loss (Dice loss)
        supervised_loss = self.dice_loss(student_outputs, targets)

        # Distillation Loss
        # For the teacher model need to convert to match student's multiclass format
        if teacher_outputs.shape[1] == 1:
            # change both student an teacher outputs to probabilities for comparison
            teacher_probs = torch.sigmoid(teacher_outputs)
            student_probs = torch.sigmoid(student_outputs)

            # Create a 4-channel output where channel 0 is 1-teacher_output (background)
            # Set channels 1, 2, 3 to teacher_output (i.e. tumor) since all three are tumor classes
            background = 1.0 - teacher_probs
            teacher_pred = torch.cat([background, teacher_probs], dim=1)

            # Use Dice loss for distillation
            distillation_loss = self.dice_loss(student_probs, teacher_pred)

            # Debug output for dist loss = 0
            '''if torch.rand(1).item() < 0.01:
                #student_probs = torch.sigmoid(student_outputs)
                print(
                    f"Teacher output statistics - min: {teacher_outputs.min().item():.4f}, max: {teacher_outputs.max().item():.4f}")
                print(
                    f"Student prob statistics - min: {student_outputs.min().item():.4f}, max: {student_outputs.max().item():.4f}")
                print(f"Supervised loss value: {supervised_loss.item():.4f}")
                print(f"Distillation loss value: {distillation_loss.item():.4f}")'''

        else:
            # If teacher already has multiclass output
            #print("In else clause for Distillation loss")
            distillation_loss = self.dice_loss(student_outputs, teacher_outputs.detach())

        # Combine losses
        total_loss = self.alpha * supervised_loss + (1 - self.alpha) * distillation_loss

        return total_loss, supervised_loss, distillation_loss

def create_data_subset(images, masks, percentage, data_augmentation):
    """
    Create a subset of the data based on percentage - for model comparison

    Args:
        images: List of numpy arrays (row, column)
        masks: List of numpy arrays (row, column)
        percentage: Percentage of the full dataset to use
        data_augmentation: Whether to use elastic deformation to augment the data

    Returns:
        subset_images: Numpy array of subset images (and augmented versions if enabled)
        subset_masks: Numpy array of subset masks (and augmented versions if enabled)
    """
    print(f"Images: {len(images)}, Percentage: {percentage}")
    num_samples = len(images)
    subset_size = int(num_samples * percentage / 100)

    # Get indices for each class
    no_tumor_indices = []
    tumor_indices = []

    for i in range(num_samples):
        # Check if this is a tumor or non-tumor image
        if np.any(masks[i] > 0):
            tumor_indices.append(i)
        else:
            no_tumor_indices.append(i)

    # Calculate the original distribution
    total_samples = len(no_tumor_indices) + len(tumor_indices)
    no_tumor_ratio = len(no_tumor_indices) / total_samples
    tumor_ratio = len(tumor_indices) / total_samples

    print(f"Original dataset distribution:")
    print(f"No tumor images: {len(no_tumor_indices)} ({no_tumor_ratio:.2%})")
    print(f"Tumor images: {len(tumor_indices)} ({tumor_ratio:.2%})")

    # Calculate how many samples to take from each class
    no_tumor_samples = int(subset_size * no_tumor_ratio)
    tumor_samples = subset_size - no_tumor_samples

    # Make sure we have at least one sample from each class
    no_tumor_samples = max(1, min(no_tumor_samples, len(no_tumor_indices)))
    tumor_samples = max(1, min(tumor_samples, len(tumor_indices)))

    random.seed(42)

    # Randomly select indices from each class
    selected_no_tumor = random.sample(no_tumor_indices, no_tumor_samples)
    selected_tumor = random.sample(tumor_indices, tumor_samples)

    # Combine indices
    selected_indices = selected_no_tumor + selected_tumor
    random.shuffle(selected_indices)

    # Print class distribution in the subset
    subset_images = images[selected_indices]
    subset_masks = masks[selected_indices]

    no_tumor_count = sum(1 for mask in subset_masks if not np.any(mask > 0))
    tumor_count = len(subset_masks) - no_tumor_count

    print(f"Subset ({percentage}%) distribution:")
    print(f"No tumor images: {no_tumor_count} ({no_tumor_count / len(subset_masks):.2%})")
    print(f"Tumor images: {tumor_count} ({tumor_count / len(subset_masks):.2%})")

    # Data augmentation
    if data_augmentation:
        print(f"Generating augmented data...")

        # Convert the numpy arrays to lists for processing
        subset_images_list = [img for img in subset_images]
        subset_masks_list = [mask for mask in subset_masks]

        # Apply elastic deformation
        augmented_images, augmented_masks = ip.elastically_deform_images(
            subset_images_list, subset_masks_list)

        # Combine original and augmented data
        combined_images = np.concatenate([subset_images, augmented_images], axis=0)
        combined_masks = np.concatenate([subset_masks, augmented_masks], axis=0)

        print(f"Final dataset size after augmentation: {len(combined_images)} images")
        return combined_images, combined_masks

    return subset_images, subset_masks

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_predictions(model, dataloader, save_dir, model_name, data_percentage, device):
    """Save example predictions from the model - 2 examples from each class"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()

    # Need to find examples from each class
    no_tumor_examples = []
    tumor_examples = []

    ### Number of examples we want for each class ###
    examples_needed = {'no_tumor': 2, 'tumor': 6}

    with torch.no_grad():
        for images, masks in dataloader:
            if examples_needed['no_tumor'] <= 0 and examples_needed['tumor'] <= 0:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Get predictions
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5

            # Find examples for each class
            for i in range(images.size(0)):
                # For binary masks, check if there are any tumor pixels
                has_tumor = torch.sum(masks[i, 1]) > 0

                # If we need more examples of this class
                if has_tumor and examples_needed['tumor'] > 0:
                    tumor_examples.append((
                        images[i].cpu(),
                        masks[i].cpu(),
                        outputs[i].cpu(),
                        predictions[i].cpu()
                    ))
                    examples_needed['tumor'] -= 1
                elif not has_tumor and examples_needed['no_tumor'] > 0:
                    no_tumor_examples.append((
                        images[i].cpu(),
                        masks[i].cpu(),
                        outputs[i].cpu(),
                        predictions[i].cpu()
                    ))
                    examples_needed['no_tumor'] -= 1

    # Create a figure with all examples
    total_examples = len(no_tumor_examples) + len(tumor_examples)

    if total_examples == 0:
        print(f"Could not find any examples for visualization for {model_name}")
        return

    plt.figure(figsize=(12, 5 * total_examples))
    plot_idx = 1

    # Plot no-tumor examples
    for i, (image, mask, output, prediction) in enumerate(no_tumor_examples):
        # Original image
        plt.subplot(total_examples, 3, plot_idx)
        plt.imshow(image[0].numpy(), cmap='gray')
        plt.title(f'No-Tumor - Image')
        plt.axis('off')
        plot_idx += 1

        # Ground truth mask (binary)
        plt.subplot(total_examples, 3, plot_idx)
        plt.imshow(mask[1].numpy(), cmap='gray')  # Class 1 is tumor
        plt.title(f'No-Tumor - Ground Truth')
        plt.axis('off')
        plot_idx += 1

        # Predicted mask (binary)
        plt.subplot(total_examples, 3, plot_idx)
        plt.imshow(prediction[1].numpy(), cmap='gray')
        plt.title(f'No-Tumor - Prediction')
        plt.axis('off')
        plot_idx += 1

    # Plot tumor examples
    for i, (image, mask, output, prediction) in enumerate(tumor_examples):
        # Original image
        plt.subplot(total_examples, 3, plot_idx)
        plt.imshow(image[0].numpy(), cmap='gray')
        plt.title(f'Tumor - Image')
        plt.axis('off')
        plot_idx += 1

        # Ground truth mask (binary)
        plt.subplot(total_examples, 3, plot_idx)
        plt.imshow(mask[1].numpy(), cmap='gray')
        plt.title(f'Tumor - Ground Truth')
        plt.axis('off')
        plot_idx += 1

        # Predicted mask (binary)
        plt.subplot(total_examples, 3, plot_idx)
        plt.imshow(prediction[1].numpy(), cmap='gray')
        plt.title(f'Tumor - Prediction')
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_{data_percentage}percent.png")
    plt.close()

def train_models(standard_model, teacher_model, student_model, train_loader, val_loader,
                 test_loader, standard_criterion, distillation_criterion,
                 standard_optimizer, student_optimizer, device, num_epochs,
                 save_dir, data_percentage):

    # Create directories for saving models and visualizations
    model_save_dir = os.path.join(save_dir, 'models')
    vis_save_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    # Lists to store metrics
    standard_train_losses = []
    student_train_losses = []
    standard_val_losses = []
    student_val_losses = []

    for epoch in range(num_epochs):
        # Set models to training mode
        standard_model.train()
        student_model.train()
        teacher_model.eval()

        # Initialize loss summation
        standard_train_loss = 0.0
        student_train_loss = 0.0
        student_supervised_loss = 0.0
        student_distillation_loss = 0.0

        # Progress bar!
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for images, masks in progress_bar:
            try:
                images, masks = images.to(device), masks.to(device)

                # Forward pass for standard model
                standard_outputs = standard_model(images)
                standard_loss = standard_criterion(standard_outputs, masks)

                # Backpropagation for standard model
                standard_optimizer.zero_grad()
                standard_loss.backward()
                standard_optimizer.step()

                # Forward pass for teacher model (in eval mode, no gradient)
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)

                # Forward pass for student model
                student_outputs = student_model(images)

                # Calculate knowledge distillation loss
                total_loss, sup_loss, dist_loss = distillation_criterion(
                    student_outputs, teacher_outputs, masks)

                # Backpropagation for student model
                student_optimizer.zero_grad()
                total_loss.backward()
                student_optimizer.step()

                # Update losses
                standard_train_loss += standard_loss.item()
                student_train_loss += total_loss.item()
                student_supervised_loss += sup_loss.item()
                student_distillation_loss += dist_loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'std_loss': standard_loss.item(),
                    'student_loss': total_loss.item()
                })
            except Exception as e: # for errors coming up
                print(f"Error processing batch: {e}")
                continue

        # Calculate average losses
        avg_standard_train_loss = standard_train_loss / len(train_loader)
        avg_student_train_loss = student_train_loss / len(train_loader)
        avg_student_supervised_loss = student_supervised_loss / len(train_loader)
        avg_student_distillation_loss = student_distillation_loss / len(train_loader)

        # Evaluate on validation set
        standard_val_loss = evaluate_model(standard_model, val_loader, standard_criterion, device)
        student_val_loss = evaluate_model(student_model, val_loader, standard_criterion, device)

        # Save losses
        standard_train_losses.append(avg_standard_train_loss)
        student_train_losses.append(avg_student_train_loss)
        standard_val_losses.append(standard_val_loss)
        student_val_losses.append(student_val_loss)

        print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
        print(f"Standard Model - Train Loss: {avg_standard_train_loss:.4f}, Val Loss: {standard_val_loss:.4f}")
        print(f"Student Model - Train Loss: {avg_student_train_loss:.4f}, Val Loss: {student_val_loss:.4f}")
        print(f"Student Supervised Loss: {avg_student_supervised_loss:.4f},"
              f"Distillation Loss: {avg_student_distillation_loss:.4f}")

    # Final evaluation on test set
    standard_test_loss = evaluate_model(standard_model, test_loader, standard_criterion, device)
    student_test_loss = evaluate_model(student_model, test_loader, standard_criterion, device)

    print("\nFinal Evaluation:")
    print(f"Data Percentage: {data_percentage}%")
    print(f"Standard Model - Test Loss: {standard_test_loss:.4f}")
    print(f"Student Model - Test Loss: {student_test_loss:.4f}")

    # Save models
    torch.save(standard_model.state_dict(), f"{model_save_dir}/standard_model_{data_percentage}percent.pth")
    torch.save(student_model.state_dict(), f"{model_save_dir}/student_model_{data_percentage}percent.pth")

    # Save visualizations
    save_predictions(standard_model, test_loader, vis_save_dir, "standard_model", data_percentage, device)
    save_predictions(student_model, test_loader, vis_save_dir, "student_model", data_percentage, device)

    # Save metrics
    metrics = {
        'data_percentage': data_percentage,
        'standard_train_losses': standard_train_losses,
        'student_train_losses': student_train_losses,
        'standard_val_losses': standard_val_losses,
        'student_val_losses': student_val_losses,
        'standard_test_loss': standard_test_loss,
        'student_test_loss': student_test_loss
    }

    with open(f"{save_dir}/metrics_{data_percentage}percent.json", 'w') as f:
        json.dump(metrics, f)

    return standard_model, student_model

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("Loading preprocessed brain tumor data...")
    data = np.load('data/brain_tumor_segmentation/processed_data.npz')
    images_train = data['images_train']
    labels_train = data['labels_train']
    images_val = data['images_val']
    labels_val = data['labels_val']
    images_test = data['images_test']
    labels_test = data['labels_test']

    print(f"Dataset sizes - Train: {len(images_train)}, Val: {len(images_val)}, Test: {len(images_test)}")

    # Load pretrained UNet model from coco_pretrain.py
    checkpoint_dir = './checkpoints'
    model_prefix = 'unet_coco'

    # Find the latest checkpoint - see coco_pretrain.py
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith(model_prefix) and f.endswith('.pth')]

    if not checkpoints:
        raise FileNotFoundError("No UNet checkpoint found. Please run coco_pretrain.py first.")

    # Get the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Using pretrained model from: {checkpoint_path}")

    # Load teacher model
    teacher_model_path = os.path.join('./models', 'teacher_model_best.pth')
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Teacher model not found at {teacher_model_path}")
    print(f"Using teacher model from: {teacher_model_path}")

    ####################################################################
    #### PARAMETERS ####
    # Percentages of data to use
    data_percentages = [5, 10, 20, 30, 40, 50, 100]
    num_epochs = 5
    batch_size = 1
    learning_rate = 1e-5
    alpha_param = 0.7 # For distillation loss with student-teacher model
    data_augmentation = True # To add augmented images to dataset
    ####################################################################

    # Create datasets
    full_train_dataset = BrainTumorDataset(images_train, labels_train)
    val_dataset = BrainTumorDataset(images_val, labels_val)
    test_dataset = BrainTumorDataset(images_test, labels_test)

    # Create validation and test dataloaders (these remain constant)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Base directory for saving results
    if data_augmentation:
        save_dir = './dual_training_results_with_augmentation'
    else:
        save_dir = './dual_training_results'
    os.makedirs(save_dir, exist_ok=True)

    # Load the teacher model first (fixed throughout training)
    teacher_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )

    # Load teacher model weights
    teacher_checkpoint = torch.load(teacher_model_path, map_location=device)
    if 'model_state_dict' in teacher_checkpoint:
        teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    else:
        teacher_model.load_state_dict(teacher_checkpoint)

    teacher_model.to(device)
    teacher_model.eval()

    # Initialize models for the first data percentage
    standard_model = None
    student_model = None

    # Loop through each data percentage
    for i, percentage in enumerate(data_percentages):
        print(f"\n{'=' * 40}")
        print(f"Training with {percentage}% of the data")
        print(f"{'=' * 40}")

        # Create subset of training data
        subset_images, subset_labels = create_data_subset(images_train, labels_train, percentage, data_augmentation)
        train_dataset = BrainTumorDataset(subset_images, subset_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print(f"Training subset size: {len(train_dataset)} images")

        if i == 0:
            # For the first percentage, initialize models with COCO pretrained weights
            standard_model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,  # We'll load from checkpoint
                in_channels=1,
                classes=2,
            )

            # Load weights from COCO pretraining
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint

            # Filter out decoder and segmentation head params
            model_dict = standard_model.state_dict()
            encoder_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict and 'decoder' not in k and 'segmentation_head' not in k}

            # Update model dict with encoder weights
            model_dict.update(encoder_dict)
            standard_model.load_state_dict(model_dict)

            # Initialize student model with same COCO pretrained weights
            student_model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=2,
            )
            student_model.load_state_dict(standard_model.state_dict())

        else:
            # For subsequent percentages, load models from previous training
            prev_percentage = data_percentages[i - 1]

            # Load standard model from previous training
            standard_model_path = f"{save_dir}/models/standard_model_{prev_percentage}percent.pth"
            standard_model.load_state_dict(torch.load(standard_model_path, map_location=device))

            # Load student model from previous training
            student_model_path = f"{save_dir}/models/student_model_{prev_percentage}percent.pth"
            student_model.load_state_dict(torch.load(student_model_path, map_location=device))

        # Move models to device
        standard_model.to(device)
        student_model.to(device)

        # Define loss functions
        standard_criterion = DiceLoss(mode='binary')
        distillation_criterion = KnowledgeDistillationLoss(alpha=alpha_param)

        # Define optimizers
        standard_optimizer = optim.AdamW(standard_model.parameters(), lr=learning_rate)
        student_optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate)

        # Train both models
        standard_model, student_model = train_models(
            standard_model, teacher_model, student_model,
            train_loader, val_loader, test_loader,
            standard_criterion, distillation_criterion,
            standard_optimizer, student_optimizer,
            device, num_epochs, save_dir, percentage
        )

    print("\nExperiment completed")

    plot_final_results(save_dir, data_percentages)

def plot_final_results(save_dir, data_percentages):
    standard_test_losses = []
    student_test_losses = []

    # Collect test losses for each data percentage
    for percentage in data_percentages:
        with open(f"{save_dir}/metrics_{percentage}percent.json", 'r') as f:
            metrics = json.load(f)

        standard_test_losses.append(metrics['standard_test_loss'])
        student_test_losses.append(metrics['student_test_loss'])

    # Plot test losses
    plt.figure(figsize=(10, 6))
    plt.plot(data_percentages, standard_test_losses, 'b-o', label='Standard UNet')
    plt.plot(data_percentages, student_test_losses, 'r-o', label='Student-Teacher UNet')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Test Loss (Dice Loss)')
    plt.title('Model Performance vs. Training Data Size')
    plt.legend()
    plt.savefig(f"{save_dir}/performance_comparison.png")
    plt.close()

    # Export results to CSV
    with open(f"{save_dir}/final_results.csv", 'w') as f:
        f.write("Data Percentage,Standard Model Test Loss,Student-Teacher Model Test Loss\n")
        for i, percentage in enumerate(data_percentages):
            f.write(f"{percentage},{standard_test_losses[i]},{student_test_losses[i]}\n")

if __name__ == "__main__":
    run_experiment()