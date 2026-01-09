import numpy as np

# Dataset
x = np.array([
    -8.5, -8, -6, -3, -1.7,
     1.2,  3.5,  5,  7,  9
])

y = np.array([
     7,  5.5,  3.5,  4,  2,
     0.9, -2.5, -3, -4, -5
])

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sqrt(
    np.sum((x - x_mean) ** 2) *
    np.sum((y - y_mean) ** 2)
)

r = numerator / denominator

print(f"Pearson correlation coefficient (r) = {r:.6f}")
