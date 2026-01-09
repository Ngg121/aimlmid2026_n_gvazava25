<img width="550" height="437" alt="image" src="https://github.com/user-attachments/assets/12d48458-4be5-4c25-9614-d6129e8fb0ad" />Pearson Correlation Coefficient

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



