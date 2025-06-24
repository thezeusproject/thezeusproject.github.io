---
title: Performance Metrics
subtitle: Performance metrics tell you if your model actually works.
author: yashasnadigsyn
---

![image](/assets/images/b3i1.png)

## Introduction

Now, that you have built your model. How do you evaluate your models? How do you know if your model is performing well or not?

All machine learning models, whether it’s linear regression, or a complex neural network, need a metric to judge it's performance.

Every Machine learning tasks can be broken down into Classification or Regression.

Classification is a supervised machine learning task where the goal is to predict a discrete class label (e.g., 'cat' or 'dog', 'fraud' or 'not fraud') based on a set of input features.

Regression is a supervised machine learning task where the goal is to predict a continuous numerical value (e.g., the price of a house, the temperature tomorrow) based on a set of input features.

Like, how we divided Machine Learning tasks into Regression and Classification. We also divide performance metrics into two types:

1. Classification Metrics
2. Regression Metrics

## Part 1: Classification Metrics

### The Confusion Matrix

The confusion matrix is the easiest way to measure and visualize the perfomance of a classification models.

It summarizes the model's predictions against the actual values. 

![image](/assets/images/b3i2.png)

The table above is a grid where the rows represent actual (true) labels and the columns represent the predicted labels by the model.

Components of the confusion matrix:
- True Positive (TP):  These are the instances where the model correctly predicted as positive and the actual label was also positive.
- True Negative (TN): These are the instances where the model correctly predicted as negative and the actual label was also negative.
- False Positives (FP): This is also called as Type I Error. These are the instances where the model incorrectly predicted as positive, but the actual label was negative.
- False Negative (FN): This is also called as Type II Error. These are the instances where the model incorrectly predicted as negative, but the actual label was positve.

Basically, the first word tells if the model was correct and the second word tells what the model predicted.

So, False Positive means the model was incorrect (False) and it predicted the positive class.

Code:
```python
def confusion_matrix(y_true, y_pred, threshold=0.5):
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

    return (tp, fp, fn, tn)
```

### Accuracy

The most straightforward way to assess the performance of a binary classifier is by computing the proportion of correctly classified instances.
This is known as accuracy.

Basically, it says, "What fraction of predictions did the model get right?"

Formula: 
$$Accuracy=(TP + TN) / (TP + TN + FP + FN)$$

Code:
```python
def accuracy(y_true, y_pred, threshold=0.5):

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred, threshold)
    N = tp+fp+fn+tn
    return (tp + tn) / N
```

### Mean misclassification error

The opposite of accuracy is Mean misclassification error (MME). It is used to assess the performance of a binary classifier is by computing the proportion of misclassified instances. 

Formula:
$$MME = (FP + FN) / (TP + TN + FP + FN)$$

Note: MME can also be calculated by accuracy.
$$MME = 1 - Accuracy$$

Code:
```python
def MME(y_true, y_pred, threshold=0.5):

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred, threshold)
    N = tp+fp+fn+tn
    return (fp + fn) / N
```

Both accuracy and MME are good performance metrics when it comes to balanced datasets. But, it doesn't work well on imbalanced datasets.

Let's use a real-world fraud detection scenario to see why. This was taken from a [real world dataset](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/Baseline_RealWorldData.html).

Imagine we have a batch of 40,000 transactions:
- Legitimate Transactions (Negative Class): 39,000
- Fraudulent Transactions (Positive Class): 1,000

This means only 0.25% of our data is fraudulent. This is a called a imbalanced dataset.

Now, let's create a model which always predicts the Negative Class (or, Not Fraud). It doesn't matter what data you provide, it always predicts "Not Fraud".

This model doesn't learn anything nor it understands the features.

Let's compute its accuracy:
- For the 39,000 legitimate transactions: It correctly predicts "Not Fraud" every time. That's 39,000 correct predictions (True Negatives).
- For the 1,000 fraudulent transactions: It incorrectly predicts "Not Fraud" every time. It misses every single fraudulent case. That's 0 correct predictions (True Positives).

Accuracy = (TP + TN) / N
So, TP = 0 and TN = 39,000 and N = 40,000

Accuracy = (0 + 39,000) / 40,000
Accuracy = 0.975

The model which doesn't learn anything and ignores all the features thrown at it have a 97.5% accuracy!

To solve this problem, we have some performance metrics.

### Precision

This is one of the most commonly used metric, also known as Positive Predicted Value. It measures, for how many of the instances the model predicted as positive are actually positive.

Formula:
$$Precision = (TP) / (TP + FP)$$

Code:
```python
def precision(y_true, y_pred, threshold=0.5):

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred, threshold)
    return (tp) / (tp+fp)
```

### Recall

This is also called as True Positive Rate (TPR). It measures, the proportion of positive instances that are correctly identified.

Informally, "Of all the actual positive cases, how many did the model find?"

This is also called as hit rate, or sensitivity.

Formula:
$$Recall = TP / (TP + FN)$$

Code:
```python
def recall(y_true, y_pred, threshold=0.5):

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred, threshold)
    return (tp) / (tp+fn)
```

### Specificity (or, True Negative Rate)

The TNR measures the proportion of negative instances that are correctly identified.

Informally, "Of all the actual negative cases, how many did the model find?"

This is also called as selectivity.

Formula:
$$TNR = (TN) / (TN + FP)$$

Code:
```python
def TNR(y_true, y_pred, threshold=0.5):

    tp, fp, fn, tn = confusion_matrix(y_true, y_pred, threshold)
    return (tn) / (tn+fp)
```

### F1-Score

An aggregate measure of the precision and recall often used in practice is the F1-score. It is the harmonic mean of precision and recall. This heavily penalizes models where one of precision or recall is very low.

Formula:
$$F1-Score = 2 * (Precision * Recall) / (Precision + Recall)$$

Code:
```python
def F1_score(y_true, y_pred, threshold=0.5):

    pre = precision(y_true, y_pred, threshold)
    rec = recall(y_true, y_pred, threshold)
    return (2 * pre * rec) / (pre + rec)
```

So far, we've looked at single numbers to judge a model. But these numbers often depend on a specific threshold we choose for making a prediction. These are called as threshold-based metrics.

What if we want to visualize the model's perfomance across all thresholds?
These are called as threshold-free metrics where we look at all thresholds to evaluate a model's perfomance.

### ROC Curve and AUC

ROC (Receiver Operating Characteristic) Curve visualizes the trade-off between the True Positive Rate (Recall) and the False Positive Rate.

Informally, It tells, "How good is my model at distinguishing between the positive and negative classes?"

AUC is called Area Under the Curve and obviously, it measures the entire two-dimensional area underneath the ROC curve.

Interpretation of AUC:
- AUC = 1.0: A perfect model.
- AUC = 0.5: A random guessing model. This is the diagonal line.

Code (You can use sklearn for this):
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### PR Curve

This is called as Precision-Recall Curve and is often used in fraudulent domain.

It visualizes the trade-off between Precision and Recall for a model.

Informally, It tells, "As we increase the recall, how much do we decrease our precision?

A PR will give you a much more honest picture of the model's perfomance when you have a highly imbalanced dataset.

Code:
```python
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Calculate PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
pr_auc = auc(recall, precision)

# Plot PR curve
plt.figure()
plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

We have covered some popular performance metrics. In addition to these, there are domain-specific metrics like Precision@k and Recall@k, which are commonly used in fraud detection and recommendation systems. There are also other metrics like Jaccard Score and Hamming Loss, which are used in multi-label classification.

## Part 2: Regression Metrics

Till now, We have looked at Classification Metrics, let us now look at Regression Metrics. Here, the model is predicting a continuous numerical value instead of a category. So, instead of asking the question, "Was the prediction correct?". Here, we ask the question, "How close is our prediction to the actual value?"

### Mean Absolute Error (MAE)

This is the most straightforward metric. It measures the average absolute difference between the predicted values and the actual values.

Formula:
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Code:
```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

### Mean Squared Error (MSE)

Here, instead of taking the absolute value of the error, we take the square of the error.

Why? MSE penalizes larger errors much more heavily than smaller ones. This is useful when you want to make the model avoid large errors.

Formula:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Code:
```python
def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))
```

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.
It combines both MAE and MSE.
- Like MAE, its value is in the original units of the target variable, making it easy to interpret.
- Like MSE, it heavily penalizes large errors.

Formula:
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Code:
```python
def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true-y_pred)))
```

### R-squared

All the metrics above mentioned tell us the size of our error but R-squared tells how well our model fits the data compared to a baseline model.
The baseline model is the one that simply predicts the mean of all the target values every single time.

Informally, R-squared answers, "What proportion of the variance in the actual values is explained by our model?"

- An $R^2$ value of 1 means the model explains 100% of the variance. A perfect model.
- An $R^2$ of 0 means the model is no better than the naive baseline of just predicting the average.
- A negative $R^2$ means the model is even worse than the naive baseline model.

Formula:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
Where, $\bar{y}$ is the mean of the true values.

Code:
```python
def r_squared(y_true, y_pred):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    
    return 1 - (ss_res / ss_tot)
```

### Adjusted R-squared

R-Squared has one major problem. It always increases or stays the same if we add more features to the model, even if the features are completely useless.This provides a more accurate measure of a model's goodness-of-fit by penalizing the inclusion of irrelevant predictors. 

Adjusted R-squared will only increase if the new feature actually improves the model's performance.

Formula:
$$R_{adj}^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$
Where, n is the number of data points and p is the number of features.

Code:
```python
def adjusted_r_squared(y_true, y_pred, p):
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
```