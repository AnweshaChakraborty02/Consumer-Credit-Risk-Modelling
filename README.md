# Consumer Credit Risk Modeling & Empirical Probability of Default (PD) Estimation

An advanced, Basel-compliant credit risk framework built to estimate consumer default risks using structural feature binning and machine learning approaches. This project implements a full credit scorecard lifecycle leveraging over 1.3 million consumer credit observations.

## 🎯 Project Objectives
* **Probability of Default (PD) Modeling:** Predict the empirical likelihood of a borrower defaulting on consumer loans.
* **Econometric Feature Selection:** Optimize predictive performance using industrial Credit Bureau metrics: Weight of Evidence (WoE) and Information Value (IV).
* **Model Benchmarking:** Evaluate traditional parametric estimators (Logistic Scorecard) against non-parametric machine learning models (Random Forest).

## 🔬 Methodology & Econometrics
* **Target Optimization:** The dependent variable (`loan_status`) was structurally mapped to a binary representation: `1` for default events (`Charged Off`, `Default`) and `0` for performing credit (`Fully Paid`). 
* **Leakage Control:** Post-origination and collection features were systematically omitted prior to ingestion to eliminate hindsight bias.
* **Weight of Evidence (WoE):** Continuous and dense fields were parsed into distinct data groups to maximize predictive stability and linearize relationship mechanics with loan outcomes:

  **WoE = ln( % of Performing Borrowers in Bin / % of Defaulted Borrowers in Bin )**

## 📊 Empirical Performance Summary

| Predictive Framework | ROC-AUC | Gini Coefficient (2 × AUC - 1) |
| :--- | :--- | :--- |
| **Logistic Regression Scorecard** | 0.7085 | 0.4169 |
| **Random Forest Challenger** | 0.7096 | 0.4192 |


## 🤖 Machine Learning Implementation
Implemented an ensemble **Machine Learning (ML)** pipeline using `scikit-learn`:
* **Algorithm:** Random Forest Classifier (`n_estimators=100`, `max_depth=10`, stratified split).
* **Objective:** Capture non-linear customer behavioral interactions.

## **Key Finding:** The traditional econometric WoE Logistic framework captures nearly identical risk gradients compared to the advanced Random Forest setup, highlighting the immense value of strategic variable binning.

## Requirements
* `pandas`
* `numpy`
* `statsmodels`
* `scikit-learn`
* `matplotlib`
