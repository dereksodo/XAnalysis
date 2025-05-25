Cross-Indicator Prediction of Major Economic Indicators Using World Bank Data

1. Introduction

Macroeconomic indicators provide a comprehensive lens for assessing the economic and social development of countries. This study leverages World Bank panel data from 1960 to 2020 to identify key national indicators and evaluate their predictability using machine learning. The focus is on cross-indicator prediction: can one major indicator be reliably predicted from the others?

2. Data and Methods

2.1 Data Source
	•	World Bank Open Data, 1960–2020, including G20 and other major economies.
	•	After cleaning and imputation, Principal Component Analysis (PCA) was used to select 13 relatively independent indicators, denoted as {F₁, F₂, …, F₁₃}.

2.2 Data Preprocessing
	•	Interpolated missing values and standardized all features (zero mean, unit variance).
	•	Constructed a country-year-feature panel: each row is a unique (country, year) pair.

2.3 Prediction Task

For each indicator $F_k$, we predict its value for each country-year using the remaining 12 indicators as input features. The process is repeated for all $k = 1, …, 13$.

2.4 Machine Learning Models

The following models are compared:
	•	Linear Regression (LR)
	•	Ridge Regression
	•	Lasso Regression
	•	Elastic Net
	•	Support Vector Regression (SVR)
	•	Random Forest (RF)
	•	K-Nearest Neighbors (KNN)
	•	XGBoost
	•	Locally Weighted Regression (LWR)

2.5 Experimental Setup
	•	Year ranges:
(a) Full period: 1960–2020
(b) Recent period: 2010–2020
	•	Cross-Validation:
5-fold cross-validation is used for each prediction, averaging metrics across folds.
	•	Evaluation Metrics:
	•	Root Mean Squared Error (RMSE)
	•	Standardized error (RMSE/STD)
	•	Coefficient of Determination ($R^2$)
	•	Visualization:
For each model and year range, bar plots of RMSE/STD and $R^2$ are generated, with feasible region thresholds indicated.

3. Results

Insert here: summary tables and figures for each model, e.g.:

	•	Table 1: Average RMSE/STD and $R^2$ for each model (1960–2020)
	•	Figure 1: RMSE/STD for each feature and model (bar plot)
	•	Figure 2: $R^2$ for each feature and model (bar plot)
	•	Table 2: Same metrics for 2010–2020

(Sample Table Template)