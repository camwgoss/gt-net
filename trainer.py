import preprocess_brain
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.unet import UNet
import torch.nn as nn
import os

import utils.image_processing as image_processing


class Trainer:
    '''
    Train U-Net for semantic segmentation. Input data assumed to be grayscale.
    If multi-channel is desired, modify _load_data to not unsqueeze(-).
    '''

    def __init__(self):
        self.loss = nn.CrossEntropyLoss()
        self.model = UNet(in_channels=1, out_channels=4)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters())

        self._load_data()  # TODO this is hard coded to load brain tumor data

    def train(self):
        # TODO batch size is hard coded
        data_train = DataLoader(self.data_train, batch_size=1)
        epochs = 5
        for epoch in range(epochs):
            for images, labels in data_train:
                labels_predicted = self.model(images)
                self.optimizer.zero_grad()
                loss = self.criterion(labels_predicted, labels)
                loss.backward()
                self.optimizer.step()

            loss_eval, fig = self.evaluate()
            print('Epoch', epoch, '|', 'Evaluation Loss', loss_eval)

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

        # evaluation; this is hard coded to evaluate all samples
        data_eval = DataLoader(self.data_eval,
                               batch_size=len(self.data_eval))
        for images, labels in data_eval:
            labels_predicted = self.model(images)
            loss = self.criterion(labels_predicted, labels)

        fig = image_processing.plot_images_labels(
            images=images.squeeze(1),
            labels=labels,
            labels_predicted=torch.argmax(labels_predicted, dim=1)
        )

        return loss.item(), fig

    def _load_data(self):
        # TODO the data loader would really be chosen based on a trainer input, e.g., coco, liver, brain
        data = preprocess_brain.load_processed_data()

        # unsqueeze(-1) because data is grayscale, so need to explicitely
        # define 1 input channel; this channel is squeezed out during image
        # pre-processing during the color -> grayscale conversion
        images_train = torch.Tensor(data['images_train']).unsqueeze(1)
        images_eval = torch.Tensor(data['images_eval']).unsqueeze(1)
        images_test = torch.Tensor(data['images_test']).unsqueeze(1)

        labels_train = torch.Tensor(data['labels_train'])
        labels_eval = torch.Tensor(data['labels_eval'])
        labels_test = torch.Tensor(data['labels_test'])

        # cross entropy loss expects dtype long
        labels_train = labels_train.to(torch.long)
        labels_eval = labels_eval.to(torch.long)
        labels_test = labels_test.to(torch.long)

        self.data_train = TensorDataset(images_train, labels_train)
        self.data_eval = TensorDataset(images_eval, labels_eval)
        self.data_test = TensorDataset(images_test, labels_test)


if __name__ == '__main__':
    trainer = Trainer()

    # train from scratch
    # trainer.train()

    # load existing model parameters
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, 'model.pth')
    trainer.model.load_state_dict(torch.load(model_path))
    trainer.evaluate()
