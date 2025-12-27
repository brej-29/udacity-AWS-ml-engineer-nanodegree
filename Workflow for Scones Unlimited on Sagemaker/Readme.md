# Workflow for Scones Unlimited on SageMaker

## 📋 Project Overview

This project builds and deploys a complete machine learning workflow for **Scones Unlimited**, a scone-delivery logistics company, using **Amazon SageMaker**. The objective is to develop an image classification model that can automatically classify delivery items, integrated with a serverless event-driven architecture using **AWS Lambda** and **AWS Step Functions**.

**Project Type:** Udacity AWS Machine Learning Engineer Nanodegree - Project 2

---

## 🎯 Business Context

Scones Unlimited is a logistics company that delivers scones and other baked goods. This ML workflow helps automate the image classification process for delivery items to improve operational efficiency and reduce manual verification time. The system uses computer vision to identify and classify items in transit.

---

## 🏗️ Project Architecture

### Workflow Components

```
┌─────────────────────────────────────────────────────────┐
│                  Data Preparation                       │
│         (S3 Upload & Image Preprocessing)               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│            SageMaker Model Training                      │
│      (Built-in Image Classification Algorithm)          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│         SageMaker Model Deployment                       │
│            (Create Inference Endpoint)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│         AWS Step Functions State Machine                 │
│    (Orchestrate Lambda Functions & Inference)           │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    Lambda 1  Lambda 2  Lambda 3
    (Serialize) (Classify) (Filter)
```

### Key AWS Services Used

| Service | Purpose |
|---------|---------|
| **Amazon SageMaker** | Model training, deployment, and inference endpoint management |
| **AWS Lambda** | Serverless functions for data processing and orchestration |
| **AWS Step Functions** | Orchestration of Lambda functions and SageMaker endpoints |
| **Amazon S3** | Data storage for training images and model artifacts |
| **AWS IAM** | Identity and access management for service permissions |

---

## 📁 Project Structure

```
Workflow for Scones Unlimited on SageMaker/
├── README.md                                    # Project documentation
├── Project2_Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker.ipynb
│   └── Complete Jupyter notebook with all steps (Data prep → Deployment → Testing)
├── Project2_Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker.html
│   └── HTML export of the notebook for easy viewing
├── Lambda.py                                    # Compiled Lambda function code
├── step-function.json                           # AWS Step Functions state machine definition
└── Screenshot-of-Working-Step-Function.PNG      # Visual proof of working workflow
```

### File Descriptions

#### 1. **Main Jupyter Notebook**
```
Project2_Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker.ipynb
```
- **Size:** Complete end-to-end ML workflow
- **Content:**
  - Data staging and exploration
  - SageMaker image classification model training
  - Model hyperparameter configuration
  - Endpoint deployment and testing
  - Lambda function development
  - Step Functions workflow creation
  - Performance monitoring and evaluation

#### 2. **Lambda.py**
```
Lambda.py
```
- Compiled Python scripts for three AWS Lambda functions
- **Function 1: serializeImageData**
  - Takes S3 image path as input
  - Reads and serializes image to JSON format
  - Output: Serialized image object
  
- **Function 2: Image-Classification (Inference)**
  - Accepts serialized image JSON
  - Sends to SageMaker endpoint for inference
  - Collects model predictions with confidence scores
  - Output: Classification results with probabilities
  
- **Function 3: Filter Low Confidence Inferences**
  - Receives inference results from Function 2
  - Filters based on confidence threshold
  - Identifies high-confidence predictions only
  - Output: Filtered predictions or rejection

#### 3. **Step Functions Workflow (step-function.json)**
- State machine definition in JSON format
- Defines execution flow of the three Lambda functions
- Includes error handling and conditional branching
- Can be imported directly into AWS Step Functions console

#### 4. **HTML Export**
- Readable version of the notebook without Jupyter environment
- Useful for documentation and presentations
- Includes all code, outputs, and explanations

---

## 🚀 Workflow Execution Steps

### Step 1: Data Staging
- Upload training and validation images to S3
- Organize images in appropriate directory structure
- Create `.lst` files (list files) mapping images to class labels

**Input:** Raw image files (.jpg, .png)  
**Output:** Organized S3 datasets with metadata

### Step 2: Model Training & Deployment
- Configure SageMaker image classification algorithm
- Set hyperparameters:
  - `num_layers`: 18 (ResNet-18 architecture)
  - `epochs`: Training iterations
  - `batch_size`: Training batch size
  - `learning_rate`: Model optimization parameter
- Train model on GPU-enabled instance (ml.p3.*)
- Deploy trained model to SageMaker endpoint
- Endpoint enables real-time inference

**Hyperparameter Configuration Example:**
```python
hyperparams = {
    'num_layers': 18,
    'use_pretrained_model': 1,
    'image_shape': '3,224,224',
    'num_classes': 2,
    'epochs': 10,
    'learning_rate': 0.01,
    'mini_batch_size': 32
}
```

### Step 3: Lambda Functions & Step Functions Workflow

#### Execution Flow:
1. **serializeImageData Lambda**
   - Input: S3 image URL
   - Process: Load image and convert to base64 JSON
   - Output: Serialized image object

2. **Image-Classification Lambda**
   - Input: Serialized image object
   - Process: Invoke SageMaker endpoint
   - Output: Model predictions with confidence scores

3. **Filter Low Confidence Inferences Lambda**
   - Input: Classification results
   - Process: Filter by confidence threshold (e.g., > 80%)
   - Output: High-confidence predictions or rejection

### Step 4: Testing & Evaluation
- Test individual Lambda functions
- Execute complete Step Functions workflow
- Monitor execution logs and metrics
- Evaluate model performance on test images
- Verify threshold filtering logic

### Step 5: Monitoring & Cleanup
- Monitor endpoint performance metrics
- Track inference latency and accuracy
- Delete endpoints when not needed (cost optimization)
- Archive training jobs and model artifacts

---

## 📊 Model Architecture

### Algorithm: SageMaker Built-in Image Classification

- **Base Network:** ResNet (Residual Neural Network)
- **Architecture:** ResNet-18 (18 convolutional layers)
- **Input:** RGB images (3 channels, 224×224 pixels)
- **Output:** Class probabilities for each image
- **Training Method:** Transfer learning from ImageNet pre-trained model
- **Optimization:** Stochastic Gradient Descent (SGD)

### Model Performance Metrics

- **Training Accuracy:** Monitored during training
- **Validation Accuracy:** Evaluated on held-out validation set
- **Inference Latency:** Time per prediction (~100-500ms)
- **Throughput:** Predictions per second capacity

---

## 🔧 Technical Requirements

### AWS Services Required
- [x] AWS SageMaker (Training & Inference)
- [x] AWS Lambda (Serverless Functions)
- [x] AWS Step Functions (Orchestration)
- [x] Amazon S3 (Data Storage)
- [x] AWS IAM (Permissions)
- [x] Amazon CloudWatch (Monitoring)

### Python & Libraries

```
Python 3.8+ (Lambda runtime)
Python 3.7.10+ (SageMaker Jupyter)

Dependencies:
- boto3 (AWS SDK)
- sagemaker (SageMaker SDK)
- numpy (Array processing)
- pandas (Data manipulation)
- PIL/Pillow (Image processing)
- json (Data serialization)
- base64 (Image encoding)
```

### AWS Instance Types

| Component | Instance Type | Purpose |
|-----------|--------------|---------|
| Development | ml.t3.medium | Notebook instance for development |
| Training | ml.p3.2xlarge | GPU training for faster model convergence |
| Inference | ml.m5.large | CPU endpoint for real-time predictions |

---

## 📝 Key Concepts Covered

### 1. **Image Classification with SageMaker**
   - Using built-in image classification algorithm
   - Transfer learning for improved accuracy
   - Hyperparameter tuning and optimization

### 2. **Serverless ML Workflows**
   - AWS Lambda for lightweight computations
   - Triggered executions and event handling
   - Lambda environment configuration

### 3. **AWS Step Functions (State Machines)**
   - Defining state machine workflows in JSON
   - Sequential task execution
   - Error handling and retry logic
   - Conditional branching based on predictions

### 4. **Data Serialization & Processing**
   - Image to JSON serialization (base64 encoding)
   - RESTful API calls to SageMaker endpoints
   - JSON response parsing and filtering

### 5. **Cost Optimization**
   - On-demand inference endpoints
   - Auto-scaling configurations
   - Cleanup of unused resources
   - Monitoring and billing alerts

---

## 💡 Learning Outcomes

Upon completing this project, you will understand:

✅ End-to-end ML workflow implementation on AWS  
✅ Model training and deployment with SageMaker  
✅ Serverless data processing with Lambda  
✅ Workflow orchestration with Step Functions  
✅ Real-world ML system architecture  
✅ AWS IAM permissions and security best practices  
✅ Cost considerations for production ML systems  
✅ Model performance monitoring and optimization  

---

## 🔍 Step Function Workflow Execution Example

### Input Payload
```json
{
  "image_url": "s3://bucket-name/images/scone-1.jpg",
  "confidence_threshold": 0.8
}
```

### Execution Flow
```
1. serializeImageData
   ├─ Input: S3 image path
   ├─ Process: Load and encode image
   └─ Output: {image_data: "base64_encoded_image"}

2. Image-Classification
   ├─ Input: Serialized image
   ├─ Process: Call SageMaker endpoint
   └─ Output: {predictions: [0.92, 0.08], class_probabilities: {...}}

3. Filter Low Confidence Inferences
   ├─ Input: Predictions from step 2
   ├─ Process: Compare to threshold (0.8)
   ├─ Check: Max probability (0.92) > 0.8? ✓
   └─ Output: {status: "PASS", confidence: 0.92}
```

### Output Payload
```json
{
  "image_processed": true,
  "classification": "scone",
  "confidence_score": 0.92,
  "passed_threshold": true,
  "processing_time_ms": 245
}
```

---

## 🛠️ How to Use This Project

### Prerequisites
- AWS Account with SageMaker, Lambda, and Step Functions access
- IAM role with necessary permissions
- S3 bucket for training data and model artifacts

### Quick Start

1. **Set up SageMaker Notebook Instance**
   ```bash
   # Access AWS SageMaker console
   # Create new notebook instance (ml.t3.medium)
   # Clone or upload the Jupyter notebook
   ```

2. **Prepare Training Data**
   ```bash
   # Upload training images to S3
   # Create .lst files for image metadata
   # Verify S3 paths in notebook
   ```

3. **Run the Notebook**
   - Execute cells sequentially from top to bottom
   - Monitor training job progress in SageMaker console
   - Wait for endpoint deployment to complete

4. **Deploy Step Functions Workflow**
   - Copy contents of `step-function.json`
   - Create new state machine in AWS Step Functions
   - Paste JSON definition and configure IAM role

5. **Test the Workflow**
   - Provide test image in S3
   - Execute Step Functions workflow
   - Monitor execution in console

### Environment Setup Example
```python
import boto3
import sagemaker
from sagemaker.estimator import Estimator

# Initialize SageMaker session
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

# Verify permissions
print(f"SageMaker role: {role}")
print(f"S3 bucket: {bucket}")
```

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Training Time | ~30-45 minutes |
| Model Size | ~100-150 MB |
| Inference Latency | 100-500 ms per image |
| Endpoint Cost | ~$0.05/hour (ml.m5.large) |
| Lambda Cost | ~$0.20 per million requests |

---

## 🔐 Security & Best Practices

1. **IAM Permissions**
   - Use least-privilege access principle
   - Create dedicated roles for Lambda functions
   - Restrict S3 bucket access to required resources

2. **Data Privacy**
   - Encrypt data at rest (S3 encryption)
   - Encrypt data in transit (HTTPS/TLS)
   - Use VPC endpoints for private access

3. **Cost Optimization**
   - Delete endpoints after use
   - Use auto-scaling for variable workloads
   - Monitor CloudWatch metrics regularly

4. **Error Handling**
   - Implement retries in Lambda functions
   - Add error thresholds to Step Functions
   - Log all failures for debugging

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Lambda timeout errors | Increase timeout setting (default: 3 sec) |
| SageMaker endpoint creation fails | Check IAM role permissions for SageMaker |
| Step Function execution fails | Verify Lambda function IAM roles |
| Image serialization errors | Check image format and S3 bucket permissions |
| Endpoint not found errors | Ensure endpoint name matches in Lambda code |

### Debugging Tips
```python
# Add CloudWatch logging to Lambda
import json
print(json.dumps(event))  # Log input event

# Test endpoint connectivity
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='endpoint-name',
    Body=image_payload,
    ContentType='application/x-image'
)
```

---

## 📚 References & Resources

### AWS Documentation
- [SageMaker Image Classification Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html)
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [AWS Step Functions Documentation](https://docs.aws.amazon.com/stepfunctions/)
- [SageMaker Model Deployment](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-deployment.html)

### Related Udacity Projects
- Project 1: Predict Bike Sharing Demand with AutoGluon
- Project 3: Image Classification with PyTorch
- Project 4: Build a ML Pipeline Orchestration on SageMaker

### External Resources
- [AWS Machine Learning Path](https://aws.amazon.com/training/learn-aws-machine-learning/)
- [SageMaker Examples Repository](https://github.com/aws/amazon-sagemaker-examples)
- [Serverless ML Pattern](https://serverlessland.com/)

---

## 📋 Deliverables Checklist

- [x] Complete Jupyter notebook with all workflow steps
- [x] Lambda function implementations (Python)
- [x] Step Functions state machine JSON definition
- [x] Trained and deployed SageMaker model
- [x] Working inference endpoint
- [x] Test results and performance metrics
- [x] Documentation and architecture diagrams
- [x] Step function execution screenshot

---

## 🎓 Key Takeaways

This project demonstrates how to build a production-ready ML system that:

1. **Automates image classification** at scale using SageMaker
2. **Orchestrates complex workflows** with Lambda and Step Functions
3. **Implements filtering logic** for high-confidence predictions
4. **Handles errors gracefully** with retry and fallback mechanisms
5. **Monitors performance** through CloudWatch metrics
6. **Optimizes costs** by using serverless architecture
7. **Follows AWS best practices** for security and compliance

---

## 🤝 Contributing

This is a course project for the Udacity AWS ML Engineer Nanodegree. While it's primarily for educational purposes, improvements and extensions are welcome:

- Add data augmentation techniques
- Implement advanced hyperparameter tuning
- Add SageMaker Autopilot for automatic model selection
- Extend to multi-class classification
- Implement A/B testing for model versions

---

## 📄 License

This project is part of the Udacity AWS Machine Learning Engineer Nanodegree program.

---

## ✨ Summary

The **Workflow for Scones Unlimited on SageMaker** is a comprehensive project that showcases how to build, train, deploy, and orchestrate an end-to-end machine learning system on AWS. It combines multiple AWS services to create a scalable, cost-effective, and production-ready ML workflow for real-world business applications.

**Last Updated:** December 2025
