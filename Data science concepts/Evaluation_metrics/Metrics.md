# Evaluation Metrics

Machine learning, artificial intelligence, and deep learning models are built on a constructive feedback loop: you develop a model, evaluate it using metrics, improve it, and repeat the process until you achieve satisfactory performance. Evaluation metrics are essential, they provide the feedback needed to refine and select models that perform well in real-world scenarios.

In this article, we will cover the most important evaluation metrics that every data science professional must understand and apply. These metrics allow you to objectively compare models and ensure robustness before deploying them into production.

---

## What Are Evaluation Metrics?

Evaluation metrics are **quantitative measures** used to assess the performance and effectiveness of a machine learning model. They offer insights into:

- Predictive power
- Generalization capability
- Model robustness and reliability

Choosing the right evaluation metric depends on the problem type (classification, regression, ranking, etc.), dataset characteristics, and business objectives.

> ⚠️ **Note:** Building a predictive model is *not* the end goal. The aim is to develop a model that performs **well on out-of-sample (unseen) data**. Skipping model evaluation can result in misleading predictions and flawed decision-making.

---

## Why Metrics Matter

Many beginner analysts or data scientists rush to apply models on test sets without understanding their robustness. A good model is not just about achieving high performance on training data, it must generalize well to new data.

To do this, we must **evaluate models using proper metrics** before trusting their predictions.

Different models require different evaluation strategies, and choosing the wrong metric may hide performance issues or overstate model success. Metrics are also vital during:

- Model comparison
- Hyperparameter tuning
- Cross-validation
- Business decision-making

---

## Topics Covered in This Article

This guide includes explanations and formulas for the most widely used evaluation metrics, grouped by task type:

- [Classification Metrics (Binary & Multiclass)](#classification-metrics)
- [Regression Metrics](#regression-evaluation-metrics)
- [Clustering Metrics](#clustering-metrics)
- [Cross-Validation Principles](#cross-validation-principles)

We’ll cover both **interpretation** and **use cases** to help you choose the right metric for any project.

---
## <a id="classification-metrics"></a>Classification Metrics

In a **classification task**, the objective is to predict a **discrete target variable** (e.g., "Yes"/"No", "Spam"/"Not Spam"). Evaluating the performance of a classification model involves several metrics, each offering unique insight into different aspects of the model’s behavior.

### Common Evaluation Metrics for Classification

- [Accuracy](#accuracy)
- [Recall](#recall)
- [False Positive Rate (FPR)](#false-positive-rate-fpr)
- [Precision](#precision)
- [F1 Score](#f1-score)
- [Confusion Matrix](#confusion-matrix)
- [Area Under Curve (AUC)](#area-under-curve-auc)
- [Matthews Correlation Coefficient (MCC)](#matthews-correlation-coefficient-mcc)
- [Brier Score](#brier-score)
---

### <a id="accuracy"></a>Accuracy

**Definition**:  
Accuracy is a fundamental metric for evaluating the performance of a classification model, providing a quick snapshot of how well the model is performing in terms of correct predictions. It is calculated as the **ratio of correct predictions to the total number of input samples**.
$$
\text{Accuracy} = \frac{\text{correct classifications}}{\text{total classifications}} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Interpretation**:  
Accuracy gives a quick snapshot of model performance but is often **misleading for imbalanced datasets**.

**Example**:
Let’s assume:
- 90% of the training data belongs to **Class A**
- 10% belongs to **Class B**

A model predicting **only Class A** would still achieve 90% accuracy on this imbalanced dataset, even though it **completely fails to identify Class B**.

If tested on a more balanced test set (e.g., 60% Class A, 40% Class B), the model’s accuracy would drop, revealing its poor generalization.

**Limitation**:
> Accuracy does **not differentiate** between types of errors (false positives vs. false negatives) and may provide a **false sense of high performance** when the dataset is imbalanced.

---

### <a id="recall"></a>Recall (True Positive Rate, TPR)

**Definition**:  
Recall is the **proportion of actual positive cases that were correctly identified by the model**. It’s also referred to as the **True Positive Rate (TPR)** or **Probability of Detection**.

$$
\text{Recall (or TPR)} = \frac{\text{correctly classified actual positives}}{\text{all actual positives}} = \frac{TP}{TP + FN}
$$


**Explanation**:  
False negatives are **actual positives** that were **misclassified as negatives**, which is why they appear in the denominator.

In the context of spam classification:
> Recall = "What fraction of spam emails are detected by this model?"

A perfect model would have **zero false negatives**, resulting in a recall of **1.0 (100%)**.

**When to use**:
- Recall is **more informative than accuracy** in **imbalanced datasets**
- Critical in domains like **disease diagnosis** or **fraud detection** where **missing a positive case is costly**

---

### False Positive Rate (FPR)

**Definition**:  
The **False Positive Rate (FPR)** is the proportion of all actual negatives that were **incorrectly classified as positives**. It is also known as the **Probability of False Alarm**.
$$
\text{FPR} = \frac{\text{incorrectly classified actual negatives}}{\text{all actual negatives}} = \frac{FP}{FP + TN}
$$


**Explanation**:  
False positives are **actual negatives** misclassified as positives.

In the context of spam classification:
> FPR = "What fraction of legitimate emails are falsely flagged as spam?"

A perfect model would have **zero false positives**, giving an FPR of **0.0 (0%)**.

**When to use**:
- Useful for evaluating the **false alarm rate**
- Less meaningful in extremely imbalanced datasets where the number of negatives is very small

---

### Precision

**Definition**:  
Precision is the **proportion of positive predictions that are actually correct**. It measures **how reliable a model’s positive predictions are**.
$$
\text{Precision} = \frac{\text{correctly classified actual positives}}{\text{everything classified as positive}} = \frac{TP}{TP + FP}
$$


**Explanation**:  
False positives reduce precision, so a high precision means the model rarely predicts "positive" incorrectly.

In the context of spam classification:
> Precision = "Out of all emails flagged as spam, how many are actually spam?"

A perfect model would have **zero false positives**, resulting in a precision of **1.0 (100%)**.

**When to use**:
- Important when **false positives are costly**
- Less meaningful if positive class is extremely rare

---

### Precision vs. Recall Trade-off

- **Precision improves** as **false positives decrease**
- **Recall improves** as **false negatives decrease**
- Adjusting the classification threshold **inversely affects** these two metrics:
  - Raising the threshold → higher precision, lower recall
  - Lowering the threshold → higher recall, lower precision

Use **Precision-Recall curves** or the **F1 Score** (next section) to balance these metrics effectively.

---

### F1 Score

**Definition**:  
The **F1 Score** is the **harmonic mean** of **Precision** and **Recall**. It provides a single metric that balances the trade-off between the two.
$$
F_1 = 2 \cdot \frac{1}{\frac{1}{\text{precision}} + \frac{1}{\text{recall}}}
$$


**Range**:  
The F1 score lies between **0 and 1**, where 1 indicates **perfect precision and recall**, and 0 indicates the worst performance.

**Why Harmonic Mean?**  
The **harmonic mean** is used instead of the arithmetic mean because it handles ratios like precision and recall more effectively. A high F1 score means:

- The model is **precise** (few false positives)
- The model is **robust** (few false negatives)

> F1 ensures that **both precision and recall must be high** to yield a high score. A model with high precision but low recall or vice versa will still have a low F1.

**When to use**:
- Especially useful in **imbalanced datasets**
- Ideal when you want to balance **false positives and false negatives**
- Common in **NLP**, **fraud detection**, **medical diagnosis**, and more

**Example**:  
A model with precision = 0.90 and recall = 0.50 will have a significantly lower F1 score, reflecting that it misses many true positives.

---

### Confusion Matrix

**Definition**:  
A **Confusion Matrix** is a performance measurement tool for classification problems. It summarizes the prediction results of a classifier by showing the correct and incorrect predictions broken down by each class.

It creates an **N × N matrix**, where **N** is the number of target classes. For binary classification, **N = 2**, resulting in a **2 × 2 matrix**.

#### Example

Let's say we tested our binary classification model on **165 samples** and got the following results:

|                | **Predicted: NO** | **Predicted: YES** |
|----------------|-------------------|---------------------|
| **Actual: NO** | 50                | 10                  |
| **Actual: YES**| 5                 | 100                 |

Let’s define the four key components:

- **True Positives (TP)** = 100 → Predicted **YES**, actually **YES**
- **True Negatives (TN)** = 50 → Predicted **NO**, actually **NO**
- **False Positives (FP)** = 10 → Predicted **YES**, actually **NO**
- **False Negatives (FN)** = 5  → Predicted **NO**, actually **YES**

---

### <a id="area-under-curve-auc"></a> AUC ROC Curve

**Definition**: **AUC-ROC curve** is a graph used to check how well a binary classification model works. It helps us to understand how well the model separates the positive cases like people with a disease from the negative cases like people without the disease at different threshold level. It shows how good the model is at telling the difference between the two classes by plotting:
- True Positive Rate (TPR): how often the model correctly predicts the positive cases also known as Sensitivity or Recall.
$$
\text{TPR} = \frac{TP}{TP + FN}
$$
- False Positive Rate (FPR): how often the model incorrectly predicts a negative case as positive.
$$
\text{FPR} = \frac{FP}{FP + TN}
$$

- Specificity: measures the proportion of actual negatives that the model correctly identifies. It is calculated as 1 - FPR.
$$
\text{TNR} = \frac{TN}{TN + FP}
$$


![alt text](image.png)
#### ROC Curve

- **ROC (Receiver Operating Characteristic) Curve** plots **TPR vs. FPR** at various classification thresholds.
- Shows the **trade-off between sensitivity and specificity**.

> The **steeper the curve**, the better the model is at classification.

---

#### AUC (Area Under the Curve)

- **AUC** quantifies the entire area under the ROC curve.
- **Higher AUC = better model performance**.


- **AUC = 1.0** → Perfect classifier  
- **AUC = 0.5** → No skill (random guessing)  
- **AUC < 0.5** → Worse than random

---

#### How AUC-ROC Works (Example Insight)

Imagine a dataset with 6 points:

- 3 positives (Class 1 – disease)
- 3 negatives (Class 0 – no disease)

Steps:
1. Randomly select a pair: one positive, one negative.
2. Check if the model gives the **positive** a higher predicted probability.
3. Repeat this for **all possible pairs**.
4. AUC is the **fraction of correct rankings** out of all possible positive-negative pairs.

![alt text](image-1.png)
---

#### When to Use AUC-ROC

Use AUC-ROC when:
- The **dataset is balanced**.
- You care equally about **false positives and false negatives**.
- You want to evaluate the model across **all thresholds**.

**Caution**: On **highly imbalanced datasets**, AUC-ROC can be **overly optimistic**. In such cases, consider using the **Precision-Recall Curve** instead.

---
#### Model Interpretation Summary

| AUC Score  | Interpretation                         |
|------------|-----------------------------------------|
| ~1.0       | Excellent separability                  |
| ~0.5       | No discriminative power (random)        |
| < 0.5      | Poor – model is confusing the classes   |

---

_In short, AUC-ROC gives a **threshold-independent** view of model performance and is widely used to compare classification models objectively._

---
### <a id="matthews-correlation-coefficient-mcc"></a> Matthews Correlation Coefficient (MCC)

**Definition**:  
The **Matthews Correlation Coefficient (MCC)** is a performance metric for **binary classification** problems. Unlike Accuracy or F1 Score, MCC takes into account **true and false positives and negatives** and is regarded as a **balanced metric**, even when class sizes are **unequal** (imbalanced).

---

#### MCC Range and Interpretation

The MCC value ranges from **-1 to +1**:

| MCC Score | Meaning                          |
|-----------|----------------------------------|
| +1        | Perfect prediction               |
| 0         | Random prediction                |
| -1        | Completely incorrect prediction  |

A key strength of MCC is its **symmetry**: the metric does **not favor any class** and treats positive and negative labels equally. This makes MCC particularly robust for **imbalanced datasets**, where metrics like accuracy can be misleading.

---

#### MCC Formula

The MCC is defined as:
$$
\text{MCC} = \frac{(TP \cdot TN) - (FP \cdot FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

The formula effectively computes the correlation between predicted and actual classes.

---

#### Why Use MCC?

- It’s one of the **most informative single-score metrics** for binary classification.
- Especially useful when:
  - The dataset is **highly imbalanced**
  - You want a metric that **balances all confusion matrix components**
- Common in **bioinformatics**, **medical diagnostics**, and **fraud detection** where **minority class performance is critical**

> MCC is generally more reliable than F1 Score or Accuracy when class distributions are uneven.

_📌 Use MCC when you want a **balanced and reliable metric** for binary classification—especially when working with rare event detection or skewed class distributions._

---
### <a id="brier-score"></a>Brier Score

**Definition**  
The **Brier Score (BS)** is a **strictly proper scoring rule** that measures the **accuracy of probabilistic predictions**.  
For a set of *N* forecasts, it is the **mean squared difference** between the predicted probability *f*<sub>t</sub> of an event and the actual outcome *o*<sub>t</sub> (0 = did not occur, 1 = occurred).

$$
\text{BS} = \frac{1}{N} \sum_{t=1}^{N} (f_t - o_t)^2
$$


**Range & Interpretation**

| BS Value | Interpretation                              |
|----------|---------------------------------------------|
| 0        | Perfect calibration and discrimination      |
| 0‒0.25   | Very good                                   |
| 0.25‒0.50| Moderate / chance level (depends on base rate) |
| 1        | Worst possible (for binary BS in [0, 1])    |

---

#### Why Use the Brier Score?

- **Calibration + Discrimination** in one number: lower BS means predicted probabilities are closer to observed frequencies.
- **Strictly proper**: forecasters are incentivised to report their **true beliefs**.
- **Model-agnostic**: works with any algorithm that outputs probabilities.
- **Complementary** to Log-Loss / Cross-Entropy: BS penalises large errors quadratically (MSE style), whereas Log-Loss penalises via the log function.


#### Worked Example

| Forecast *P(rain)* | Outcome | Calculation | BS  |
|--------------------|---------|-------------|-----|
| 1.00 (100 %)       | Rain    | (1 − 1)²    | 0.00|
| 1.00               | No rain | (1 − 0)²    | 1.00|
| 0.70               | Rain    | (0.70 − 1)² | 0.09|
| 0.70               | No rain | (0.70 − 0)² | 0.49|
| 0.30               | Rain    | (0.30 − 1)² | 0.49|
| 0.50               | Either  | (0.50 − 1)² or (0.50 − 0)² = 0.25 |

> **Lower is better**: over many forecasts, an average BS closer to **0** means better-calibrated probabilities.
---

## Regression Evaluation Metrics

In a **regression task**, the goal is to predict a **continuous target variable**. Unlike classification, where the output is a discrete label, regression metrics evaluate how closely the model’s predicted values match the actual values.

Below are the most commonly used evaluation metrics for regression:

- [Mean Absolute Error (MAE)](#mae)
- [Mean Squared Error (MSE)](#mse)
- [Root Mean Squared Error (RMSE)](#rmse)
- [Root Mean Squared Logarithmic Error (RMSLE)](#rmsle)
- [R² Score (Coefficient of Determination)](#r2)


---

### <a id="mae"></a> Mean Absolute Error (MAE)

**Definition**:  
MAE measures the **average absolute difference** between actual and predicted values. It tells us how far, on average, the predictions are from the actual values.
$$
\text{MAE} = \frac{1}{N} \sum_{j=1}^{N} \left| y_j - \hat{y}_j \right|
$$


**Notes**:
- Easy to interpret.
- Does **not indicate the direction** of error (over-predict vs. under-predict).
- All errors are treated equally.

---

### <a id="mse"></a> Mean Squared Error (MSE)

**Definition**:  
MSE calculates the **average of the squared differences** between predicted and actual values.
$$
\text{MSE} = \frac{1}{N} \sum_{j=1}^{N} \left( y_j - \hat{y}_j \right)^2
$$


**Notes**:
- Penalizes **larger errors** more than smaller ones.
- Helps in focusing on larger errors.
- Commonly used in optimization due to smooth gradient.

---

### <a id="rmse"></a> Root Mean Squared Error (RMSE)

**Definition**:  
RMSE is the **square root of MSE**, and brings the error metric back to the same unit as the original values.
$$
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{j=1}^{N} \left( y_j - \hat{y}_j \right)^2 }
$$


**Notes**:
- More interpretable than MSE.
- **Sensitive to outliers** due to squaring.

---

### <a id="rmsle"></a> Root Mean Squared Logarithmic Error (RMSLE)

**Definition**:  
RMSLE is used when the target variable spans a **wide range**, and **underestimating** is worse than overestimating. It reduces the impact of large predicted values when they are close to actual values.
$$
\text{RMSLE} = \sqrt{ \frac{1}{N} \sum_{j=1}^{N} \left( \log(y_j + 1) - \log(\hat{y}_j + 1) \right)^2 }
$$


**Notes**:
- Adds a **log transform**, which makes it suitable for datasets with exponential growth patterns.
- Ideal when **relative differences** matter more than absolute ones.

---

### <a id="r2"></a> R² Score (Coefficient of Determination)

**Definition**:  
R² measures how well the **predicted values explain the variability** of the actual values. It represents the proportion of variance in the dependent variable that is predictable from the independent variables.
$$
R^2 = 1 - \frac{ \sum_{j=1}^{n} (y_j - \hat{y}_j)^2 }{ \sum_{j=1}^{n} (y_j - \bar{y})^2 }
$$


**Range**:
- 1.0: Perfect prediction
- 0.0: Model does no better than the mean
- < 0.0: Model is worse than the mean

**Notes**:
- Common in linear regression analysis.
- Not always reliable with **nonlinear models** or **imbalanced datasets**.

---
