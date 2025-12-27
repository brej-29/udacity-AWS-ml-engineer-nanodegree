<div align="center">
  <h1>🚀 Udacity AWS ML Engineer Nanodegree</h1>
  <p><i>Complete Machine Learning engineering course with hands-on projects covering AutoML, ML Workflows, Image Classification, and Production Deployment on AWS SageMaker — built with Python, AWS Lambda, Step Functions, and industry best practices</i></p>
</div>

<br>

<div align="center">
  <a href="https://www.udacity.com/course/aws-machine-learning-engineer-nanodegree--nd189">
    <img alt="Udacity Nanodegree" src="https://img.shields.io/badge/Udacity-Nanodegree-blue?logo=udacity&logoColor=white">
  </a>
  <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue">
  <img alt="Cloud Platform" src="https://img.shields.io/badge/Cloud-AWS-orange?logo=amazonaws">
  <img alt="ML Framework" src="https://img.shields.io/badge/ML%20Framework-SageMaker-ff9900">
  <img alt="AutoML" src="https://img.shields.io/badge/AutoML-AutoGluon-brightgreen">
  <img alt="Workflow" src="https://img.shields.io/badge/Orchestration-Step%20Functions-yellow">
  <img alt="Compute" src="https://img.shields.io/badge/Serverless-Lambda-FF9900">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-success">
</div>

<div align="center">
  <br>
  <b>Built with the AWS ML stack and cutting-edge technologies:</b>
  <br><br>
  <code>Python 3.8+</code> | <code>AWS SageMaker</code> | <code>AutoGluon</code> | <code>AWS Lambda</code> | <code>AWS Step Functions</code> | <code>S3</code> | <code>IAM</code> | <code>PyTorch</code> | <code>TensorFlow</code> | <code>scikit-learn</code> | <code>Pandas</code> | <code>NumPy</code>
</div>

---

## **📚 Overview**

The **Udacity AWS ML Engineer Nanodegree** is a comprehensive 5-month intermediate-level program (5-10 hours/week) designed to equip software developers and data scientists with production-grade machine learning skills using Amazon SageMaker. This repository contains complete implementations of all course projects demonstrating:

- **Automated Machine Learning** with AutoGluon for rapid baseline development
- **End-to-end ML Workflows** combining SageMaker, Lambda, and Step Functions
- **Computer Vision** with image classification and model deployment
- **Production ML** with monitoring, optimization, and cloud-native architecture
- **Real-world Problem Solving** with Kaggle datasets and industry scenarios

---

## **📋 Program Structure**

### **Course 1: Introduction to Machine Learning**
- ML Fundamentals
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Selection & Evaluation

### **Course 2: Developing Your First ML Workflow**
- SageMaker Essentials
- Model Training & Optimization
- Deployment & Inference
- Lambda Functions & Event-Driven Architecture

### **Course 3: Deep Learning Topics**
- Computer Vision Fundamentals
- Convolutional Neural Networks (CNNs)
- Transfer Learning & Fine-tuning
- Production Image Classification

### **Course 4: Operationalizing ML Projects**
- Advanced Deployment Strategies
- Model Monitoring & Observability
- Cost Optimization
- Security & Compliance Best Practices

---

## **🎯 Projects Overview**

### **Project 1: Bike Sharing Demand Prediction with AutoGluon**
📂 **Folder:** `Bike Sharing Demand with AutoGluon/`

Predict hourly bike-sharing demand using automated machine learning:
- **Dataset:** Kaggle Bike Sharing Demand (10,000+ hourly records)
- **Techniques:** AutoGluon tabular prediction, temporal feature engineering
- **Results:** 66% improvement in RMSE (1.32 → 0.45) through iterative refinement
- **Key Skills:** Feature engineering, hyperparameter tuning, AutoML workflow

**Files:**
- `Project1_Predict-Bike-Sharing-Demand-with-AutoGluon.ipynb` - Complete notebook
- `train.csv` / `test.csv` - Kaggle dataset
- `submission_predictions.csv` - Final predictions

---

### **Project 2: ML Workflow for Scones Unlimited on SageMaker**
📂 **Folder:** `Workflow for Scones Unlimited on SageMaker/`

Build a production ML pipeline for image classification and automatic item sorting:
- **Objective:** Classify delivery items in real-time for logistics optimization
- **Architecture:** SageMaker training → Lambda processing → Step Functions orchestration
- **Key Components:**
  - Image classification using ResNet-18 (transfer learning)
  - Serverless data processing pipeline
  - Event-driven workflow with confidence filtering
- **Key Skills:** SageMaker deployment, Lambda functions, state machines, MLOps

**Files:**
- `Project2_Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker.ipynb` - Full workflow
- `Lambda.py` - Compiled Lambda function scripts
- `step-function.json` - AWS Step Functions state machine definition
- `Screenshot-of-Working-Step-Function.PNG` - Workflow visualization

---

### **Project 3: Image Classification with PyTorch** *(Additional Implementation)*
📂 **Folder:** `Image Classification/` (if present)

Deep learning approach to image classification:
- **Framework:** PyTorch + torchvision
- **Model:** Pre-trained CNN with fine-tuning
- **Deployment:** SageMaker endpoints for inference

---

## **📂 Repository Structure**

```
udacity-AWS-ml-engineer-nanodegree/
├── README.md                                          # This file
│
├── Bike Sharing Demand with AutoGluon/
│   ├── Project1_Predict-Bike-Sharing-Demand-with-AutoGluon.ipynb
│   ├── Project1_Predict-Bike-Sharing-Demand-with-AutoGluon.html
│   ├── train.csv
│   ├── test.csv
│   └── README.md                                      # Project-specific documentation
│
├── Workflow for Scones Unlimited on SageMaker/
│   ├── Project2_Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker.ipynb
│   ├── Project2_Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker.html
│   ├── Lambda.py
│   ├── step-function.json
│   ├── Screenshot-of-Working-Step-Function.PNG
│   └── README.md                                      # Project-specific documentation
│
└── [Additional Projects and Resources]/
```

---

## **🛠️ Technology Stack**

### **Cloud Platform**
- **Amazon SageMaker** - Model training, hosting, and inference management
- **AWS Lambda** - Serverless compute for event-driven processing
- **AWS Step Functions** - Workflow orchestration and state machine management
- **Amazon S3** - Data lake and model artifact storage
- **AWS IAM** - Identity and access management

### **ML & Data Science**
- **AutoGluon** - Automated tabular prediction (Project 1)
- **PyTorch** - Deep learning framework for computer vision
- **TensorFlow/Keras** - Alternative deep learning framework
- **scikit-learn** - Classical ML algorithms and preprocessing
- **Pandas & NumPy** - Data manipulation and numerical computing

### **Development & Deployment**
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development environment
- **Git & GitHub** - Version control and collaboration

---

## **🚀 Getting Started**

### **Prerequisites**
- **AWS Account** with SageMaker, Lambda, Step Functions, and S3 access
- **Python 3.8+** installed locally
- **Git** for version control
- **Jupyter Notebook** or **SageMaker Studio** for interactive development
- Familiarity with:
  - Python programming (40+ hours experience)
  - Data structures (lists, dictionaries, arrays)
  - Basic statistics and mathematics
  - ML concepts (training, testing, evaluation)

### **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/brej-29/udacity-AWS-ml-engineer-nanodegree.git
   cd udacity-AWS-ml-engineer-nanodegree
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   *Or install specific project dependencies:*
   ```bash
   pip install autogluon pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

### **AWS Setup**

1. **Create an AWS Account** (if not already done)
2. **Set up SageMaker:**
   - Navigate to SageMaker console
   - Create a notebook instance (ml.t3.medium recommended for development)
   - Select a role with SageMaker, S3, and Lambda permissions
3. **Configure AWS Credentials:**
   ```bash
   aws configure
   # Enter your AWS Access Key ID and Secret Access Key
   ```

### **Running Projects**

**Project 1: Bike Sharing Demand**
```bash
cd "Bike Sharing Demand with AutoGluon"
jupyter notebook
# Open: Project1_Predict-Bike-Sharing-Demand-with-AutoGluon.ipynb
```

**Project 2: Scones Unlimited Workflow**
```bash
cd "Workflow for Scones Unlimited on SageMaker"
jupyter notebook
# Open: Project2_Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker.ipynb
```

---

## **📊 Project Details & Learning Outcomes**

### **Project 1: AutoGluon Baseline Development**

**What You'll Learn:**
- ✅ Automated machine learning (AutoML) fundamentals
- ✅ Feature engineering for time-series data
- ✅ Hyperparameter optimization and tuning
- ✅ Model evaluation metrics (RMSE, MAE, R²)
- ✅ Iterative model improvement strategy

**Performance Metrics:**
| Stage | RMSE | Improvement |
|-------|------|------------|
| Baseline | 1.32 | — |
| Feature-Engineered | 0.47 | 64.1% ↓ |
| Optimized | 0.45 | 66% total |

**Key Insight:** Domain-driven feature engineering provided 64% of total improvement, outweighing algorithmic tuning.

---

### **Project 2: End-to-End ML Workflow**

**What You'll Learn:**
- ✅ SageMaker model training and deployment
- ✅ Creating inference endpoints for real-time predictions
- ✅ AWS Lambda function development
- ✅ Event-driven architectures with Step Functions
- ✅ Data serialization and API integration
- ✅ Serverless ML pipeline orchestration
- ✅ Monitoring and observability in production

**Architecture Components:**
```
Data Input
    ↓
Lambda 1: Serialize Image Data
    ↓
Lambda 2: Invoke SageMaker Endpoint (Inference)
    ↓
Lambda 3: Filter Low-Confidence Predictions
    ↓
Output: High-Confidence Classifications
```

**AWS Services Integrated:**
- SageMaker (model hosting)
- Lambda (data processing)
- Step Functions (workflow orchestration)
- S3 (data storage)
- CloudWatch (monitoring)

---

## **💡 Key Concepts Covered**

### **Machine Learning Fundamentals**
- Supervised vs. Unsupervised Learning
- Training/Validation/Test Split Strategies
- Overfitting & Regularization
- Cross-validation Techniques
- Feature Scaling & Normalization

### **Advanced ML Topics**
- **Ensemble Methods:** Gradient Boosting (XGBoost, LightGBM, CatBoost)
- **Transfer Learning:** Using pre-trained models for domain adaptation
- **Hyperparameter Tuning:** Grid search, random search, Bayesian optimization
- **Model Evaluation:** Precision, recall, F1, AUC-ROC, RMSE

### **AWS Cloud & Deployment**
- **Infrastructure as Code:** CloudFormation templates
- **Containerization:** Docker images for SageMaker custom algorithms
- **Monitoring:** CloudWatch logs, SageMaker Model Monitor
- **Cost Optimization:** Instance selection, auto-scaling strategies
- **Security:** IAM roles, encryption at rest/in transit

### **Production ML**
- Model versioning and artifact management
- A/B testing for model comparison
- Automated retraining pipelines
- Drift detection and monitoring
- Compliance and governance

---

## **📖 How to Use This Repository**

1. **For Learning:**
   - Start with Project 1 (AutoGluon) to understand ML fundamentals
   - Progress to Project 2 for cloud-native architectures
   - Review individual project READMEs for detailed explanations

2. **For Reference:**
   - Use notebooks as templates for your own projects
   - Adapt Lambda functions for different ML tasks
   - Reference Step Functions JSON for workflow patterns

3. **For Interviews & Portfolio:**
   - Showcase projects on GitHub for recruiters
   - Explain architecture decisions in your cover letter
   - Discuss performance improvements and lessons learned

---

## **🔗 Resources & Documentation**

### **Official AWS Documentation**
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [AWS Step Functions User Guide](https://docs.aws.amazon.com/stepfunctions/)
- [SageMaker Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

### **AutoGluon & ML Libraries**
- [AutoGluon Official Docs](https://auto.gluon.ai/)
- [AutoGluon-Tabular Paper](https://arxiv.org/abs/2003.06505)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/)

### **Kaggle Datasets**
- [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)
- [Scones Unlimited Dataset](https://www.kaggle.com/) (custom for course)

### **Udacity Resources**
- [Udacity AWS ML Nanodegree](https://www.udacity.com/course/aws-machine-learning-engineer-nanodegree--nd189)
- [Udacity ML Foundations (Free)](https://www.udacity.com/course/aws-machine-learning-foundations--cd0385)

---

## **🎓 Curriculum Map**

### **Month 1: Foundations**
- Introduction to ML concepts
- Python fundamentals review
- Data exploration and analysis
- AWS SageMaker basics

### **Month 2: ML Workflows**
- AutoML with AutoGluon (Project 1)
- Feature engineering techniques
- Model evaluation and metrics
- SageMaker training infrastructure

### **Month 3: Production Deployment**
- Model deployment and hosting
- Lambda function development
- Serverless architectures (Project 2)
- Step Functions orchestration

### **Month 4: Computer Vision & Deep Learning**
- CNN fundamentals
- Transfer learning
- Image classification pipelines
- Advanced SageMaker features

### **Month 5: Capstone & Advanced Topics**
- Model monitoring and observability
- Cost optimization strategies
- Security and compliance
- Career preparation

---

## **✨ Notable Features**

✅ **Complete End-to-End Examples**
- Data ingestion to model deployment
- Production-ready code patterns
- Best practices demonstrated

✅ **Cloud-Native Architecture**
- Serverless workflows
- Event-driven design
- Scalable infrastructure

✅ **Real-World Datasets**
- Kaggle competitions
- Industry scenarios
- Practical problem-solving

✅ **Performance Optimization**
- Hyperparameter tuning results
- Iteration improvements documented
- Lessons learned captured

✅ **Comprehensive Documentation**
- Detailed project READMEs
- Architecture diagrams
- Troubleshooting guides

---

## **🐛 Troubleshooting**

### **Common Issues**

| Issue | Solution |
|-------|----------|
| SageMaker role permissions error | Ensure IAM role includes SageMaker, Lambda, S3, and IAM permissions |
| Lambda timeout errors | Increase timeout in Lambda configuration (default: 3 seconds) |
| S3 bucket access denied | Verify bucket policy and IAM role S3 permissions |
| Model training fails | Check instance type availability in your region |
| Step Functions execution error | Review CloudWatch logs for detailed error messages |

### **Getting Help**

- Review project-specific README files
- Check AWS documentation and support
- Consult Udacity forums and Discord
- Debug using CloudWatch Logs

---

## **🤝 Contributing**

This is a personal learning repository, but suggestions for improvements are welcome:

1. Open an issue for bugs or improvements
2. Submit pull requests for code enhancements
3. Share your insights and lessons learned

---

## **📝 Project Completion Status**

- ✅ **Project 1:** Bike Sharing Demand with AutoGluon — Complete
- ✅ **Project 2:** ML Workflow for Scones Unlimited — Complete
- ✅ **Project 3:** Image Classification using AWS Sagemaker
- ✅ **Project 4:** Operationalizing an AWS ML Project

---

## **📄 License**

This repository contains course work from the Udacity AWS ML Engineer Nanodegree program. Individual project files may be subject to Udacity's terms of service. See LICENSE file for details.

---

## **👨‍💻 About the Author**

**Brejesh Balakrishnan**
- 📍 Location: Tiruppur, Tamil Nadu, India
- 🎓 Currently: AWS ML Engineer Nanodegree (In Progress)
- 🛠️ Skills: Python, AWS SageMaker, Machine Learning, Data Science
- 🔗 LinkedIn: [linkedin.com/in/brejesh-balakrishnan](https://www.linkedin.com/in/brejesh-balakrishnan-7855051b9/)
- 🐙 GitHub: [@brej-29](https://github.com/brej-29)

---

## **📞 Contact & Support**

- **Issues & Questions:** Open an issue in this GitHub repository
- **Email:** Available upon request
- **LinkedIn:** Connect for career opportunities and discussions
- **Udacity Support:** Access through your Udacity dashboard

---

<div align="center">

### 🌟 Thank you for exploring this repository! 🌟

If you find this helpful, please consider:
- ⭐ Starring this repository
- 🔗 Sharing it with others
- 💬 Providing feedback and suggestions
- 🤝 Contributing improvements

**Happy Learning & Build Great ML Systems!** 🚀

---

*Last Updated: December 2025*

*This repository is actively maintained and updated with latest AWS ML practices.*

</div>
