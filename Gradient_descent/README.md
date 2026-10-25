# Gradient Descent for Linear Regression

This directory contains a small, fully visible implementation of gradient descent for a two-feature linear regression model. The accompanying notebook is designed both as an executable demonstration and as a presentation aid: every major quantity is printed at each update so that the mathematical process can be followed from the initial parameters to the final predictions.

## Contents

- `gradient_descent.ipynb`: implementation, numerical derivative check, iteration log, final predictions, and plots.
- `README.md`: explanation of the model, equations, results, and suggested presentation flow.

## Learning objective

The notebook demonstrates how gradient descent minimizes mean squared error (MSE) by repeatedly:

1. Computing predictions from the current weights and bias.
2. Measuring the prediction error.
3. Computing the gradient of the cost function.
4. Moving the parameters in the direction that reduces the cost.

It also compares the analytically derived gradient with a numerical derivative calculated using SciPy. This provides a direct check that the matrix-calculus implementation is correct.

## Model and data

The model has two input features and one output:

```text
y_hat = X m + b
```

The notebook uses the following small dataset:

```text
X = [[1,  3],
     [4, 10]]

y = [[5],
     [6]]
```

The parameter vector and shared bias are initialized as:

```text
m = [[-1],
     [ 2]]

b = [[1],
     [1]]
```

There are two rows in `X`, so `n = 2`. The bias is stored as a two-entry vector because there is one prediction per observation. Both entries contain the same scalar bias value.

The learning rate is `0.01`, and the notebook performs three parameter updates. The displayed iteration zero is the initial state, so the notebook reports iterations zero through three.

## Cost function

For each observation, the prediction error is:

```text
e = X m + b - y
```

The cost function is the mean squared error:

```text
J(m, b) = (1/n) e^T e
```

Because `e^T e` is a one-by-one matrix, the implementation uses `.item()` to return the cost as a scalar Python value.

## Gradient equations

For the weight vector, the gradient is:

```text
dJ/dm = (2/n) X^T e
```

The notebook treats the bias as one shared scalar repeated for every row. Therefore, the scalar bias derivative is based on the sum of the errors:

```text
dJ/db = (2/n) 1^T e
```

The implementation expands this value back to the bias-vector shape using `ones`, so it can update `b` without changing the matrix representation:

```text
grad_b = (2/n) (ones^T e) ones
```

The gradient descent update rule is:

```text
m = m - learning_rate * dJ/dm
b = b - learning_rate * dJ/db
```

The subtraction is essential: the gradient points in the direction of greatest increase in the cost, so moving against it reduces the cost for a suitable learning rate.

## Numerical derivative check

The function `derivative_function` wraps SciPy's numerical derivative routine. `scipy_gradient` varies one parameter at a time while keeping the others fixed, then estimates:

- the derivative with respect to the first weight;
- the derivative with respect to the second weight;
- the derivative with respect to the shared bias.

At each update, these numerical values are printed beside the matrix-form gradients. In the recorded notebook output, the values agree at every checked step:

```text
Iteration 0: (45.00, 113.00, 12.00)
Iteration 1: (-11.84, -31.08, -5.18)
Iteration 2: (3.7962, 8.5618, -0.444)
```

This agreement confirms that the gradient formulas used for the updates match the numerical derivative of the implemented cost function.

## Recorded results

The initial and updated states are:

| Iteration | Weights `m` | Bias `b` | Predictions | MSE `J` |
| ---: | --- | --- | --- | ---: |
| 0 | `[-1, 2]` | `[1, 1]` | `[6, 17]` | `61.0000` |
| 1 | `[-1.45, 0.87]` | `[0.88, 0.88]` | `[2.04, 3.78]` | `6.8450` |
| 2 | `[-1.3316, 1.1808]` | `[0.9318, 0.9318]` | `[3.1426, 7.4134]` | `2.72381716` |
| 3 | `[-1.369562, 1.095182]` | `[0.93624, 0.93624]` | `[2.852224, 6.409812]` | `2.39044381076` |

The cost falls from `61.0` to approximately `2.39` over the three updates. This shows that the selected learning rate moves the parameters toward a lower-error solution. The final predictions are closer to the targets `[5, 6]` than the initial predictions, although the model has not been trained to convergence.

## Plots

The notebook creates two plots:

1. **Parameter values over iterations:** shows how `m1`, `m2`, and `b` change after each update.
2. **Error over iterations:** shows the MSE decreasing across the recorded iterations.

The plots make the numerical output easier to interpret during a presentation. The first plot explains the parameter trajectory; the second plot shows the optimization objective directly.

## How to run

1. Open `gradient_descent.ipynb` in Jupyter Notebook, JupyterLab, or Visual Studio Code.
2. Select a Python kernel with the required packages installed.
3. Run the cells from top to bottom.
4. Confirm that the numerical derivative check agrees with the analytical gradient.
5. Review the printed iteration table and the two generated plots.

The notebook uses:

- Python
- NumPy
- Matplotlib
- SciPy

If the imports fail, install the missing packages in the active environment:

```bash
pip install numpy matplotlib scipy
```

