# Gradient Descent for Linear Regression

This directory contains the implementation of a simple linear regression model using gradient descent. The notebook contains the Python implementation and plots that show the optimization results visually.

## Contents

- `gradient_descent.ipynb`: implementation, numerical derivative check, iteration log, final predictions, and plots.
- `README.md`: explanation of the model, equations, and results.

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

The parameter vector and independent bias coordinates are initialized as:

```text
m = [[-1],
     [ 2]]

b = [[1],
     [1]]
```

There are two rows in `X`, so `n = 2`. The bias is stored as a two-entry vector:

```text
b = [b1, b2]^T
```

The initial values of both coordinates are `1.0`, but they are treated as independent parameters. `b1` affects the first prediction and `b2` affects the second prediction. They are not forced to remain equal after an update.

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

Because each bias coordinate affects only its corresponding prediction, the bias gradient is computed element by element. For:

```text
dJ/db_i = (2/n) e_i
```

the vector form is:

```text
grad_b = (2/n) e
```

Unlike the previous shared-bias implementation, this calculation does not sum all errors into one common gradient. Each bias coordinate is updated using only its corresponding error.

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
- the derivative with respect to `b1`;
- the derivative with respect to `b2`.

At each update, these numerical values are printed beside the matrix-form gradients. In the recorded notebook output, the values agree at every checked step:

```text
Iteration 0: (45.00, 113.00, 1.00, 11.00)
Iteration 1: (-11.69, -30.65, -2.85, -2.21)
Iteration 2: (3.5937, 8.0917, -1.7851, 1.3447)
```

The four values on each line are, in order, `dm1`, `dm2`, `db1`, and `db2`. Their agreement confirms that the gradient formulas used for the updates match the numerical derivatives of the implemented cost function.

## Recorded results

The initial and updated states are:

| Iteration | Weights `m` | Bias `b` | Predictions | MSE `J` |
| ---: | --- | --- | --- | ---: |
| 0 | `[-1, 2]` | `[1, 1]` | `[6, 17]` | `61.0000` |
| 1 | `[-1.45, 0.87]` | `[0.99, 0.89]` | `[2.15, 3.79]` | `6.5033` |
| 2 | `[-1.3331, 1.1765]` | `[1.0185, 0.9121]` | `[3.2149, 7.3447]` | `2.49740005` |
| 3 | `[-1.369037, 1.095583]` | `[1.036351, 0.898653]` | `[2.954063, 6.378335]` | `2.164497790097` |

The cost falls from `61.0` to approximately `2.16` over the three updates. This shows that the selected learning rate moves the parameters toward a lower-error solution. The final predictions are closer to the targets `[5, 6]` than the initial predictions, although the model has not been trained to convergence.

## Plots

The notebook creates two plots:

1. **Parameter values over iterations:** shows how `m1`, `m2`, `b1`, and `b2` change after each update.
2. **Error over iterations:** shows the MSE decreasing across the recorded iterations.

The plots make the numerical output easier to interpret during a presentation. The first plot explains the parameter trajectory; the second plot shows the optimization objective directly.

## How to run

1. Open `gradient_descent.ipynb` in Jupyter Notebook, JupyterLab, or Visual Studio Code.
2. Select a Python kernel with the required packages installed.
3. Run the cells from top to bottom.
4. Confirm that the numerical derivative check agrees with the analytical gradient for both weights and both bias coordinates.
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

## Interpretation and limitations

The MSE decreases at every recorded update, from `61.0` to approximately `2.16`. The final parameters after three updates are:

```text
m = [-1.369037, 1.095583]
b = [1.036351, 0.898653]
```

The final predictions are `[2.954063, 6.378335]`, compared with the target values `[5, 6]`. The model is improving, but only three updates are performed, so the parameters have not been trained to convergence.

The independent bias vector is useful for demonstrating coordinate-wise gradient updates. It is different from the usual linear regression model, which uses one shared scalar intercept for every observation. With independent biases, each observation has its own additive adjustment, so this example should be understood as an instructional demonstration rather than a standard regression model.

The dataset contains only two observations. It is therefore suitable for showing the mechanics of gradient descent, but not for evaluating generalization or predictive performance.
