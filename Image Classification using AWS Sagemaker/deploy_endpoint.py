import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
logger = logging.getLogger(__name__)


def net():
    """
    Initializes the model architecture.
    MUST MATCH train_model.py EXACTLY.
    """
    model = models.resnet50(pretrained=False)  # Weights are loaded from state_dict, so pretrained=False here is fine

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    # Fixed to match train_model.py:
    # Linear(2048, 256) -> ReLU -> Dropout -> Linear(256, 133)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 133)
    )
    return model


def model_fn(model_dir):
    """Loads the model and state dictionary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on {device}")

    model = net()

    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    with open(model_path, 'rb') as f:
        # Map to CPU if no GPU is present (crucial for local testing/CPU instances)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(f))
        else:
            model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    model.to(device).eval()
    logger.info("Model loaded successfully")
    return model


def input_fn(request_body, content_type):
    """Deserializes the input data."""
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body)).convert('RGB')
    raise Exception(f'Requested unsupported ContentType: {content_type}')


def predict_fn(input_object, model):
    """Runs prediction on the input object."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(input_object)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # Add batch dimension and run inference
        prediction = model(input_tensor.unsqueeze(0).to(device))

    # Apply softmax to get probabilities (since we removed LogSoftmax to match training)
    probabilities = F.softmax(prediction, dim=1)
    return probabilities.cpu().numpy()