# 📊 Bias-Variance Tradeoff Visualization

This project demonstrates the **Bias-Variance Tradeoff** in Machine Learning using an interactive visualization.

---

## 🚀 Overview

Understanding the balance between **underfitting** and **overfitting** is critical when building machine learning models.

This project visualizes:

* High Bias (Underfitting)
* Balanced Model (Optimal Fit)
* High Variance (Overfitting)

using a noisy dataset and interactive graphs.

---

## 📌 Key Concepts

### 🔴 High Bias (Underfitting)

* Model is too simple
* Fails to capture underlying patterns
* Poor performance on both training and test data

### 🟢 Balanced Model

* Captures the true pattern
* Generalizes well to unseen data

### 🔵 High Variance (Overfitting)

* Model is too complex
* Fits noise instead of actual pattern
* Performs well on training but poorly on new data

---

## 📈 Visualization

![Bias Variance Graph](images/bias_variance_plot.png)

> Hover over the graph (in interactive version) to see explanations for each model.

---

## 🧑‍💻 Tech Stack

* Python
* NumPy
* Scikit-learn
* Plotly (for interactive visualization)

---

## ⚙️ How It Works

1. Generate a noisy dataset using a sine function
2. Train three models:

   * Linear Regression (High Bias)
   * Polynomial Degree 4 (Balanced)
   * Polynomial Degree 12 (High Variance)
3. Plot predictions using Plotly
4. Add hover explanations for better understanding

---

## ▶️ How to Run

```bash
pip install numpy scikit-learn plotly
python main.py
```

---

## 🎯 Key Takeaways

* Simple models → High Bias → Underfitting
* Complex models → High Variance → Overfitting
* Best models balance both

---

## 📌 Future Improvements

* Add training vs testing error graph
* Build a Streamlit dashboard
* Add slider to control model complexity

---

## 🙌 Author

Nikhil Khandalkar

---
