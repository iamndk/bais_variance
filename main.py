import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go

# Better dataset (more points + noise)
np.random.seed(42)
X = np.linspace(0, 10, 30).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # noisy sine wave

# Models
# 1. High Bias (too simple)
model_simple = LinearRegression()
model_simple.fit(X, y)

# 2. Balanced (good fit)
model_balanced = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
model_balanced.fit(X, y)

# 3. High Variance (overfit)
model_complex = make_pipeline(PolynomialFeatures(degree=12), LinearRegression())
model_complex.fit(X, y)

# Smooth curve
X_test = np.linspace(0, 10, 200).reshape(-1, 1)

y_simple = model_simple.predict(X_test)
y_balanced = model_balanced.predict(X_test)
y_complex = model_complex.predict(X_test)

# Plot
fig = go.Figure()

# Data points
fig.add_trace(go.Scatter(
    x=X.flatten(),
    y=y,
    mode='markers',
    name='Actual Data',
    hovertemplate="Actual Data<br>X: %{x:.2f}<br>Y: %{y:.2f}"
))

# High Bias
fig.add_trace(go.Scatter(
    x=X_test.flatten(),
    y=y_simple,
    mode='lines',
    name='High Bias (Underfit)',
    hovertemplate="High Bias<br>Too simple, misses pattern"
))

# Balanced
fig.add_trace(go.Scatter(
    x=X_test.flatten(),
    y=y_balanced,
    mode='lines',
    name='Balanced Model',
    hovertemplate="Balanced<br>Captures pattern well"
))

# High Variance
fig.add_trace(go.Scatter(
    x=X_test.flatten(),
    y=y_complex,
    mode='lines',
    name='High Variance (Overfit)',
    hovertemplate="High Variance<br>Too complex, overfits noise"
))

# Layout
fig.update_layout(
    title="Bias-Variance Tradeoff (Clear Visualization)",
    xaxis_title="X",
    yaxis_title="y",
    hovermode="closest"
)

fig.show()
