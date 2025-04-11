import preprocess_brain
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import segmentation_models_pytorch

import utils.image_processing as image_processing


class Trainer:
    '''
    Train U-Net for semantic segmentation. Input data assumed to be grayscale.
    If multi-channel is desired, modify _load_data to not unsqueeze(-).
    Arguments:
        dataset: Dateset to run on, {'brain', 'liver', 'coco'}.
        device: Compute device, {'cuda', 'cpu'}.
    '''

    def __init__(self, dataset: str = 'brain', device: str = 'cpu'):
        self.device = device
        print('Using device:', device)

        model = segmentation_models_pytorch.Unet(in_channels=1, classes=4)
        self.model = model.to(device)

        self.criterion = segmentation_models_pytorch.losses.DiceLoss(
            mode='multiclass')

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # TODO this is hard coded to load brain tumor data
        self._load_data(dataset)

    def train(self):
        batch_size = 1  # batch size used in original U-Net paper
        data_train = DataLoader(
            self.data_train, batch_size=batch_size, shuffle=True)
        epochs = 10
        for epoch in range(epochs):
            for images, labels in data_train:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                labels_predicted = self.model(images)
                loss = self.criterion(labels_predicted, labels)
                loss.backward()
                self.optimizer.step()

            loss_eval, fig = self.evaluate()
            print('Epoch', epoch, '|', 'Validation Loss', loss_eval)

        torch.save(self.model.state_dict(), os.path.join('.', 'model.pth'))

    def evaluate(self):
        '''
        Evaluate model performance. This function returns the loss defined by
        self.criterion as well as a figure of images, labels, and predictions.
        If model_path is not provided, self.model will be used.
        Returns:
            loss: Loss defined by self.criterion.
            fig: Plot showing images, labels, and predictions
        '''

        with torch.no_grad():
            data_eval = DataLoader(self.data_eval, batch_size=10, shuffle=True)
            accumulated_loss = 0
            batch = 0
            for images, labels in data_eval:

                images = images.to(self.device)
                labels = labels.to(self.device)
                labels_predicted = self.model(images)

                batch_loss = self.criterion(labels_predicted, labels)
                # scale loss by number of samples in batch
                accumulated_loss += batch_loss.item() * len(images)

                if batch == 0:  # only plot first batch of data
                    fig = image_processing.plot_images_labels(
                        images=images.squeeze(1).to('cpu'),
                        labels=labels.to('cpu'),
                        labels_predicted=torch.argmax(
                            labels_predicted, dim=1).to('cpu')
                    )
                batch += 1

            loss = accumulated_loss / len(self.data_eval)  # average loss

        return loss, fig

    def _load_data(self, dataset: str = 'brain'):
        if dataset == 'brain':
            data = preprocess_brain.load_processed_data()
        else:
            raise Exception('Error: the ' + dataset +
                            ' dataset has not been implemented.')

        # unsqueeze(-1) because data is grayscale, so need to explicitely
        # define 1 input channel; this channel is squeezed out during image
        # pre-processing during the color -> grayscale conversion
        images_train = torch.tensor(
            data['images_train'], dtype=torch.float32).unsqueeze(1)
        images_eval = torch.tensor(
            data['images_eval'], dtype=torch.float32).unsqueeze(1)
        images_test = torch.tensor(
            data['images_test'], dtype=torch.float32).unsqueeze(1)

        labels_train = torch.tensor(data['labels_train'], dtype=torch.long)
        labels_eval = torch.tensor(data['labels_eval'], dtype=torch.long)
        labels_test = torch.tensor(data['labels_test'], dtype=torch.long)

        self.data_train = TensorDataset(images_train, labels_train)
        self.data_eval = TensorDataset(images_eval, labels_eval)
        self.data_test = TensorDataset(images_test, labels_test)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(device=device)

    # train from scratch
    trainer.train()

    # load existing model parameters
    # model_dir = os.path.dirname(__file__)
    # model_path = os.path.join(model_dir, 'model.pth')
    # trainer.model.load_state_dict(torch.load(model_path))
    # trainer.evaluate()
