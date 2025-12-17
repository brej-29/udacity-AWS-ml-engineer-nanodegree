#TODO: Import your dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import logging
import sys
import time

# Essential imports for debugging and profiling
import smdebug.pytorch as smd
from PIL import ImageFile

# Set up logging and handle corrupted images
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
ImageFile.LOAD_TRUNCATED_IMAGES = True

#TODO: Import dependencies for Debugging andd Profiling


def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Testing started!")
    model.eval()

    # Set the SMDebug hook to evaluation mode
    if hook:
        hook.set_mode(smd.modes.EVAL)

    test_loss = 0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

    average_accuracy = running_corrects / len(test_loader.dataset)
    average_loss = test_loss / len(test_loader.dataset)

    logger.info(f'Test set: Average loss: {average_loss:.4f}, Average accuracy: {100*average_accuracy:.2f}%')


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # Set the SMDebug hook to training mode
    if hook:
        hook.set_mode(smd.modes.TRAIN)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        logger.info(f"Epoch {epoch}: Training Loss: {epoch_loss:.4f}, Accuracy: {100 * epoch_acc:.2f}%")

        # Validation after each epoch
        logger.info("Running Validation...")
        test(model, validation_loader, criterion, device, hook)

    return model


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    # Freeze convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer for 133 classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 133)  # 133 Dog Breeds
    )
    return model


def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    valid_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'valid'), transform=test_transform)
    test_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)

    # Create data loaders (using num_workers=1 for consistency/safety, but could be higher on GPU instances)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, valid_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    # Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device: {device}")

    # Initialize Model
    model = net()
    model = model.to(device)

    '''
    TODO: Create your loss and optimizer
    '''
    # Create Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    # Initialize SMDebug Hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # Create Data Loaders
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    # Train the model
    logger.info("Starting Training...")
    start_time = time.time()
    model = train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device, hook)
    logger.info("Training time: {} seconds.".format(round(time.time() - start_time, 2)))

    '''
    TODO: Test the model to see its accuracy
    '''
    # Final Test
    logger.info("Starting Testing...")
    start_time = time.time()
    test(model, test_loader, criterion, device, hook)
    logger.info("Testing time: {} seconds.".format(round(time.time() - start_time, 2)))

    '''
    TODO: Save the trained model
    '''
    # Save the model
    logger.info("Saving Model...")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters (these will be passed from the notebook)
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    '''
    TODO: Specify any training args that you might need
    '''
    # SageMaker container environment variables
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    main(args)
