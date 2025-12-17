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
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    # IMPORTANT: This print statement matches the Regex in the notebook for HPO
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)")

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        print(f"Epoch {epoch}: Train accuracy: {100. * correct / total:.2f}%")
        
        # Run validation at the end of every epoch
        print("Running Validation...")
        test(model, validation_loader, criterion, device)
    
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    # Freeze convolutional layers to use them as a fixed feature extractor
    for param in model.parameters():
        param.requires_grad = False   

    # Replace the last fully connected layer for our 133 classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 133) # 133 Dog Breeds
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

    # Create data loaders
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
    print(f"Running on Device: {device}")
    
    # Initialize Model
    model = net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    TODO: Test the model to see its accuracy
    TODO: Save the trained model
    '''
    # Create Data Loaders
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    # Train the model
    print("Starting Training...")
    model = train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)
    
    # Final Test
    print("Starting Testing...")
    test(model, test_loader, criterion, device)

    # Save the model
    print("Saving Model...")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    # SageMaker container environment variables
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    
    main(args)
