
import numpy as np
import matplotlib.pyplot as plt

# ================================
# DATA GENERATION
# ================================

def generate_dataset(n=10, noise_std=0.1, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise_std, size=n)
    return x.reshape(-1, 1), y.reshape(-1, 1)

# ================================
# FEATURE ENGINEERING
# ================================

def build_polynomial_features(x, degree):
    # TODO: build Vandermonde-style matrix including bias term
    #x is (n, 1)
    n = x.shape[0]

    # bias column
    X = np.ones((n, degree + 1))

    for d in range(1, degree +1):
        X[:, d] = x[:, 0] ** d

    return X

x_train, y_train = generate_dataset(n=10)
degree = 20
X_matrix = build_polynomial_features(x_train, degree)

# Verification
print(f"Feature Matrix Shape: {X_matrix.shape}")
print(f"First row (x^0 to x^{degree}):\n{X_matrix[0, :5]} ...")

# ================================
# NORMAL EQUATION
# ================================

def normal_equation(X, y):
    # TODO: implement (X^T X)^(-1) X^T y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xt_y = X.T @ y
    w = XtX_inv @ Xt_y
    return w

# 2. Fitting a 20-degree polynomial to 10 points
w = normal_equation(X_matrix, y_train)

# 3. Generating 100 test points for a smooth plot
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
X_test_poly = np.hstack([x_test**i for i in range(21)])
y_pred = X_test_poly @ w

# 4. Plotting the results
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color='red', label='Training Data (10 pts)')
plt.plot(x_test, y_pred, color='blue', label='Fitted Curve (Degree 20)')
plt.ylim(-2, 2)
plt.legend()
plt.title("Task A2: 20-Degree Polynomial Fit via Normal Equation")
plt.show()


# 1. Print coefficient magnitudes (weights)
print("Coefficient Magnitudes (first 5):")
print(w[:5].flatten())
print(f"L2 Norm of weights: {np.linalg.norm(w):.2e}")

# 2. Compute condition number of (X^T X)
XTX = X_matrix.T @ X_matrix
cond_number = np.linalg.cond(XTX)
print(f"Condition Number of (X^T X): {cond_number:.2e}")


# ================================
# RIDGE REGRESSION
# ================================

def ridge_regression(X, y, lambda_):
    # TODO:
    # Implement (X^T X + λI)^(-1) X^T y
    # Do NOT regularize bias term
    pass

# ================================
# ANALYSIS UTILITIES
# ================================

def compute_condition_number(X):
    return np.linalg.cond(X.T @ X)

def plot_results(x_train, y_train, x_plot, y_plot, title="Model Fit"):
    plt.scatter(x_train, y_train)
    plt.plot(x_plot, y_plot)
    plt.title(title)
    plt.show()

# ================================
# UNIT TESTS
# ================================

def test_polynomial_shape():
    x, _ = generate_dataset()
    X = build_polynomial_features(x, 20)
    assert X.shape[1] == 21, "Feature matrix should include bias + 20 degrees"

def test_normal_equation_solution():
    x, y = generate_dataset()
    X = build_polynomial_features(x, 2)
    w = normal_equation(X, y)
    assert w.shape[0] == X.shape[1], "Weight dimension mismatch"

def test_ridge_reduces_norm():
    x, y = generate_dataset()
    X = build_polynomial_features(x, 20)
    w_no_reg = normal_equation(X, y)
    w_reg = ridge_regression(X, y, 1.0)
    assert np.linalg.norm(w_reg) < np.linalg.norm(w_no_reg),         "Ridge should reduce coefficient magnitude"

def run_all_tests():
    test_polynomial_shape()
    test_normal_equation_solution()
    test_ridge_reduces_norm()
    print("All tests passed.")

if __name__ == "__main__":
    run_all_tests()
