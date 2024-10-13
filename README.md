# Student Dropout Prediction Project(Dropout Predictor)

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Development](#model-development)
  - [Model 1: Simple Neural Network](#model-1-simple-neural-network)
  - [Model 2: Optimized Neural Network](#model-2-optimized-neural-network)
    - [Optimization Techniques](#optimization-techniques)
      - [L1 and L2 Regularization](#l1-and-l2-regularization)
      - [Early Stopping](#early-stopping)
      - [Optimizer Configuration](#optimizer-configuration)
    - [Parameter Selection and Tuning](#parameter-selection-and-tuning)
- [Model Evaluation](#model-evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Results Summary](#results-summary)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

Dropout Predictor aims to develop predictive models to determine student outcomes based on various features. By leveraging machine learning techniques, specifically neural networks, the models classify whether a student will drop out or graduate. Two models are implemented: a simple neural network and an optimized version incorporating regularization and early stopping to enhance performance and prevent overfitting.

## Project Structure

The project consists of the following main components:

- Data loading and exploration
- Data preprocessing
- Model implementation (baseline and optimized)
- Model training and evaluation
- Result visualization and analysis

## Data

The dataset used for this project is found on [Kaggle](https://www.kaggle.com/datasets/naveenkumar20bps1137/predict-students-dropout-and-academic-success?select=dataset.csv ). It contains various features related to student demographics, academic performance, and other relevant factors influencing enrollment outcomes. The target variable is `Target`, which indicates whether a student has enrolled (`Graduate`) or dropped out (`Dropout`).

## Preprocessing

Data preprocessing is crucial to ensure the quality and suitability of the dataset for model training. The following steps were undertaken:

1. **Data Loading**: The dataset is loaded using pandas.
2. **Exploratory Data Analysis (EDA)**: 
   - **Correlation Analysis**: Numeric features are analyzed to understand their correlations using a heatmap.
   - **Data Cleaning**: Instances labeled as `Enrolled` are removed from Target Column in order to focus on the binary classification between `Dropout` and `Graduate`.
   - **Handling Missing Values**: The dataset is checked for missing values, and appropriate measures are taken if any are found.
3. **Encoding Categorical Variables**: The `LabelEncoder` transforms the target variable into binary form (0 for 'Dropout', 1 for 'Graduate').
4. **Feature Scaling**: `StandardScaler` is applied to normalize feature values, ensuring that all features contribute equally to the model's performance.
5. **Data Splitting**: The dataset is divided into training and testing sets using an 80-20 split to evaluate model performance effectively.

## Model Development

Throughout the development process, multiple models were trained to find the best-performing one. Model 1 was the initial, simple version without any optimizations. From there, several models with different combinations of techniques were trained, but most of them did not outperformed the baseline. Here are some of the models that were tested:

- Model (L2, RMSprop, Early Stopping): Achieved a test loss of 0.87 and accuracy of 84%.
- Model (L2, Dropout, Early Stopping): Had a test loss of 1.15 and accuracy of 73%.
- Model (L1, Dropout, Early Stopping): Ended with a test loss of 3.01 and accuracy of 77%.
- Model (L2, Early Stopping): Scored a test loss of 0.96 and accuracy of 80%.

While each model had its unique combination of techniques, none of them improved significantly compared to the baseline model. This led to the development of `Model 2`, which incorporated a more refined approach with both L1 and L2 regularization, Adam optimizer, and early stopping. `Model 2` showed clear improvement in all performance metrics, as you'll see in the detailed results.

### Model 1: Simple Neural Network

The first model serves as a baseline and consists of a straightforward neural network architecture without any optimization techniques.

**Architecture:**
- **Input Layer**: 16 neurons with ReLU activation.
- **Hidden Layers**: Two hidden layers, each with 32 neurons and ReLU activation.
- **Output Layer**: 1 neuron with Sigmoid activation for binary classification.

**Compilation:**
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

**Training:**
- **Epochs**: 1000
- **Validation**: 20% of the data
- **Verbosity**: Silent (no output during training)

### Model 2: Optimized Neural Network

To enhance the performance of the neural network and mitigate overfitting, several optimization techniques are incorporated into the second model.

#### Optimization Techniques

##### L1 and L2 Regularization

**Principle:** Regularization techniques add a penalty to the loss function to discourage complex models, thereby preventing overfitting.

- **L1 Regularization (`l1`)**: Adds the absolute value of the weights as a penalty term. It can lead to sparse models where some weights become zero, effectively performing feature selection.
  
- **L2 Regularization (`l2`)**: Adds the squared value of the weights as a penalty term. It discourages large weights but does not enforce sparsity.

**Application in Model 2:**
- The first hidden layer uses L1 regularization with a coefficient of `0.01`.
- The subsequent hidden layers use L2 regularization with a coefficient of `0.01`.

**Relevance:** Regularization helps in generalizing the model better to unseen data by preventing it from fitting the noise in the training data.

##### Early Stopping

**Principle:** Early stopping halts the training process when the model's performance on a validation set stops improving, thereby preventing overfitting.

**Parameters:**
- **Monitor**: `val_loss` (validation loss)
- **Patience**: `5` epochs (number of epochs with no improvement after which training is stopped)
- **Restore Best Weights**: `True` (restores the model weights from the epoch with the best value of the monitored quantity)

**Relevance:** Early stopping ensures that the model does not train excessively on the training data, maintaining its ability to generalize.

##### Optimizer Configuration

**Optimizer Used:** Adam

**Parameters:**
- **Learning Rate**: `0.001`

**Relevance:** Adam optimizer combines the advantages of two other extensions of stochastic gradient descent, namely AdaGrad and RMSProp, to provide an efficient optimization algorithm. The learning rate controls how much the model weights are updated during training.

#### Parameter Selection and Tuning

- **Regularization Coefficients (`0.01` for both L1 and L2):** Chosen based on standard practices and preliminary experiments to balance the trade-off between underfitting and overfitting.
  
- **Patience (`5` epochs):** Selected to allow the model some flexibility to improve after a temporary plateau in validation loss.

- **Learning Rate (`0.001`):** A commonly used default value for the Adam optimizer, providing a good balance between convergence speed and stability.

These parameters were selected through a combination of literature review, standard best practices, and empirical testing to ensure optimal model performance.

## Model Evaluation

### Evaluation Metrics

The models are evaluated using the following metrics:

- **Confusion Matrix:** Provides a summary of prediction results, showing the number of true positives, true negatives, false positives, and false negatives.
  
- **Specificity:** Measures the proportion of actual negatives correctly identified (True Negative Rate).
  
- **F1 Score:** The harmonic mean of precision and recall, providing a balance between the two.
  
- **Classification Report:** Includes precision, recall, F1-score, and support for each class.

### Results Summary

| Model   | Test Accuracy | Specificity | F1 Score |
|---------|---------------|-------------|----------|
| Model 1 | 85.53%        | 0.8159      | 0.8827   |
| Model 2 | 91.87%        | 0.8412      | 0.9364   |

### Confusion Matrices

#### Model 1 - Confusion Matrix
|            | Predicted Dropout | Predicted Graduate |
|------------|-------------------|---------------------|
| Dropout    | 226               | 51                  |
| Graduate   | 54                | 395                 |

#### Model 2 - Confusion Matrix
|            | Predicted Dropout | Predicted Graduate |
|------------|-------------------|---------------------|
| Dropout    | 233               | 44                  |
| Graduate   | 15                | 434                 |

### Classification Reports

#### Model 1 - Classification Report
|          | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Dropout  | 0.81      | 0.82   | 0.81     | 277     |
| Graduate | 0.89      | 0.88   | 0.88     | 449     |
| Accuracy |           |        | 0.86     | 726     |
| Macro Avg| 0.85      | 0.85   | 0.85     | 726     |
| Weighted Avg| 0.86   | 0.86   | 0.86     | 726     |

#### Model 2 - Classification Report
|          | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Dropout  | 0.94      | 0.84   | 0.89     | 277     |
| Graduate | 0.91      | 0.97   | 0.94     | 449     |
| Accuracy |           |        | 0.92     | 726     |
| Macro Avg| 0.92      | 0.90   | 0.91     | 726     |
| Weighted Avg| 0.92   | 0.92   | 0.92     | 726     |

## Performance Analysis

A detailed comparison of Model 1 (baseline) and Model 2 (optimized) reveals significant improvements in predictive performance:

1. **Overall Accuracy**: 
   - Model 1: 85.53%
   - Model 2: 91.87%
   Model 2 shows a substantial improvement in overall accuracy, correctly classifying a larger proportion of instances.

2. **Specificity**:
   - Model 1: 0.8159
   - Model 2: 0.8412
   Model 2 demonstrates higher specificity, indicating better performance in correctly identifying students who will graduate.

3. **F1 Score**:
   - Model 1: 0.8827
   - Model 2: 0.9364
   The higher F1 score of Model 2 suggests a better balance between precision and recall.

4. **Precision and Recall**:
   For Dropout prediction:
   - Model 1: Precision = 0.81, Recall = 0.82
   - Model 2: Precision = 0.94, Recall = 0.84
   
   For Graduate prediction:
   - Model 1: Precision = 0.89, Recall = 0.88
   - Model 2: Precision = 0.91, Recall = 0.97
   
   Model 2 shows significant improvements, especially in precision for dropout prediction and recall for graduate prediction.

5. **Confusion Matrix Analysis**:
   - False Positives (predicting dropout when actually graduate):
     Model 1: 54, Model 2: 15
   - False Negatives (predicting graduate when actually dropout):
     Model 1: 51, Model 2: 44
   
   Model 2 substantially reduces false positives and slightly reduces false negatives.

### Reasons for Model 2's Superior Performance:

1. **Regularization**: The use of L1 and L2 regularization in Model 2 helps prevent overfitting, leading to better generalization on unseen data.

2. **Early Stopping**: This technique prevents overfitting by stopping training when validation performance starts to degrade.

3. **Adaptive Learning Rate**: The Adam optimizer used in Model 2 adapts the learning rate during training, potentially finding a better optimum in the parameter space.

4. **Balanced Improvement**: Model 2 shows improvements across all metrics, indicating a more robust and well-rounded model.

5. **Reduced False Positives**: The significant reduction in false positives (54 to 15) is particularly noteworthy, as it means fewer students are incorrectly classified as potential dropouts.

6. **High Precision for Dropout Prediction**: Model 2's precision of 0.94 for dropout prediction means that when it predicts a student will drop out, it's correct 94% of the time. This is crucial for targeted interventions.

7. **Improved Recall for Graduate Prediction**: The high recall (0.97) for graduate prediction in Model 2 means it correctly identifies 97% of all actual graduates, missing very few.

## Usage

To utilize the trained models for prediction, follow these steps:

1. **Load the Model:**
   ```python
   import pickle

   # Load Model 1
   with open('saved_models/model1.pkl', 'rb') as file:
       loaded_model1 = pickle.load(file)

   # Load Model 2
   with open('saved_models/model2.pkl', 'rb') as file:
       loaded_model2 = pickle.load(file)


## Conclusion

This project demonstrates the implementation and comparison of two neural network models for predicting student dropout rates. The optimized model (Model 2) incorporates various regularization techniques to improve performance and prevent overfitting.

Based on the comprehensive analysis, Model 2 significantly outperforms the baseline Model 1 across all key metrics. Its superior performance can be attributed to the sophisticated optimization techniques employed, which lead to better generalization, reduced overfitting, and more balanced predictions across both classes. 