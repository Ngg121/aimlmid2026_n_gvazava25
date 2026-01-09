ASSignment 1:






Pearson Correlation Coefficient

Repository name: **aimlmid2026_n_gvazava25**  
Author: Nino Gvazava

---

## Task Description

The goal of this task is to analyze the relationship between two variables (X and Y)
displayed as blue points on an online graph and compute **Pearson’s correlation
coefficient**.  
The process, calculation, and visualization must be reproducible using Python.

---

## Data Source

The data points were obtained from the following page:

**max.ge/aiml_midterm/24957_html**

Each blue point displays its `(x, y)` coordinates when hovering the mouse over it.
These values were manually extracted and stored in a dataset.

---

## Dataset

The extracted dataset consists of 10 data points:

| Point | X    | Y   |
|------:|-----:|----:|
| A | -8.5 | 7 |
| B | -8.0 | 5.5 |
| C | -6.0 | 3.5 |
| D | -3.0 | 4 |
| E | -1.7 | 2 |
| F | 1.2 | 0.9 |
| G | 3.5 | -2.5 |
| H | 5.0 | -3 |
| I | 7.0 | -4 |
| J | 9.0 | -5 |

---

## Methodology

Pearson’s correlation coefficient was computed using the standard formula:

\[
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
\]

The calculation was implemented in Python to ensure accuracy and reproducibility.

---

## Result

The computed Pearson correlation coefficient is:

**r = -0.9819**

This value indicates a **very strong negative linear correlation** between X and Y.
As X increases, Y decreases almost linearly.

---

## Visualization

In the attachments you may find visual graph - scatter plot visualizes the dataset along with a best-fit line,
illustrating the correlation:

<img width="550" height="437" alt="image" src="https://github.com/user-attachments/assets/12d48458-4be5-4c25-9614-d6129e8fb0ad" /> 



















Assignment 2: 


1. Dataset

The dataset used for this project is publicly available on GitHub: https://github.com/Ngg121/aimlmid2026_n_gvazava25/blob/main/1.%20Datasets.xlsx
Raw versio is:https://raw.githubusercontent.com/Ngg121/aimlmid2026_n_gvazava25/main/1.%20Datasets.xlsx

The dataset contains 2,500 email records, each represented as a single comma-delimited row with the following numeric features:

words, links, capital_words, spam_word_count, is_spam


Where:

words – total number of words in the email

links – number of URLs in the email

capital_words – number of fully capitalized words

spam_word_count – number of words matching a predefined spam dictionary

is_spam – class label (1 = spam, 0 = legitimate)




2. Model Training and Validation Description
Data Loading

The training script loads the dataset directly from the local machine:

/Users/ninogvazava/Desktop/2. Dataset.csv


The first four columns are used as input features, and the last column (is_spam) is used as the target label.

Train/Test Split

The dataset is split as follows:

70% (1,750 samples) for training

30% (750 samples) for validation

A stratified split is used when possible to preserve the original class distribution.

Model Used

The classifier is a Binary Logistic Regression model implemented using scikit-learn.





3. Logistic Regression Model Results
Model Summary
LOGISTIC REGRESSION SPAM CLASSIFIER REPORT - Updated as an attachment file.
==================================================
Dataset: /Users/ninogvazava/Desktop/2. Dataset.csv
Rows: 2500 | Features columns: 4 | Label column: is_spam
Train/Test split: 1750/750 (70/30)

Intercept
b0 = 1.970884

Feature Coefficients
Feature	Coefficient
capital_words	3.414614
links	2.145914
spam_word_count	1.970693
words	1.457101

Interpretation:

Emails with more capitalized words and links strongly increase the probability of being spam.

Presence of known spam keywords also significantly contributes to spam classification.







4. Model Evaluation: Confusion Matrix and Accuracy
Confusion Matrix (Test Set – 30%)
[[357  10]
 [ 19 364]]


Where:

True Negatives (TN): 357 legitimate emails correctly classified

False Positives (FP): 10 legitimate emails misclassified as spam

False Negatives (FN): 19 spam emails misclassified as legitimate

True Positives (TP): 364 spam emails correctly classified

Accuracy
Accuracy = 0.96


The model correctly classifies 96% of unseen emails, demonstrating strong generalization performance.






5. Email Parser Description

The email parser converts raw email text into the same numeric feature format used in the dataset.

Extracted Features

words: total number of word tokens

links: count of URLs (http://, https://, www)

capital_words: number of fully uppercase words (length ≥ 2)

spam_word_count: number of words found in a predefined spam dictionary

Parser Code
def parse_email(text: str):
    tokens = WORD_RE.findall(text)
    words = len(tokens)
    links = len(URL_RE.findall(text))
    capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
    spam_word_count = sum(1 for t in tokens if t.lower() in SPAM_WORDS)
    return words, links, capital_words, spam_word_count


The output format exactly matches the dataset feature order.




6. Testing with Real Emails (Spam and Non-Spam)

Two separate text files were attached:

Spam email sample

Non-spam (legitimate) email sample

The code parses each email, extracts features, and passes them to the trained model.

When tested with a legitimate project-related email, the model correctly classified it as non-spam, returning a low spam probability, confirming correct behavior on real-world input.








7. Confusion Matrix Heatmap

A confusion matrix heatmap was generated and saved as an image:

confusion_matrix.png


The heatmap visually represents classification performance:

Darker diagonal cells indicate correct predictions

Lighter off-diagonal cells indicate misclassifications

This visualization makes it easier to understand model errors and overall balance between spam and legitimate detection.

