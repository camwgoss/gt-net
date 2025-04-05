import preprocess_brain
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.unet import UNet
import torch.nn as nn
import os


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

        self._load_data()

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

            # evaluation; this is hard coded to evaluate all samples
            data_eval = DataLoader(self.data_eval, batch_size=len(
                self.data_eval))  # get all evaluation data
            for images, labels in data_eval:
                labels_predicted = self.model(images)
                loss = self.criterion(labels_predicted, labels)
                print('Epoch', epoch, '|', 'Evaluation Loss', loss)

        torch.save(self.model.state_dict(), os.path.join('.', 'model.pth'))

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
    trainer.train()
