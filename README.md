# Cross-Country Overnight Interest Rate Prediction

## Project Overview

This project focuses on forecasting overnight interest rate (OIR) changes for major global economies using machine learning techniques and macro-structural features. The analysis covers G20 countries (excluding the African Union) and uses annual data from 1960 to 2020. Our objective is to predict both the timing and magnitude of overnight rate changes, leveraging demographic and economic indicators.

## Motivation

Overnight interest rates play a crucial role in monetary policy and financial markets. Traditional forecasting approaches usually target single countries or rely heavily on financial market signals. Here, we investigate whether demographic and macro-structural features—such as population trends and economic indicators—can provide predictive power in a cross-country, long-term context.

## Dataset

- **Countries**: G20 countries, excluding the African Union
- **Years**: 1960–2020 (annual data)
- **Target**: Overnight interest rate (OIR) for each country and year
- **Features**:
    - Population (total, growth rate)
    - Gender ratio
    - Birth cohort statistics (e.g., population born in specific years, population aged 15, etc.)
    - GDP, GDP growth rate
    - Inflation rate
    - Unemployment rate
    - Fiscal balance and other macroeconomic indicators
    - Lagged features (e.g., previous year's OIR)
    - Country and time fixed effects

All macroeconomic and demographic data are sourced from [World Bank Data by Indicators](https://github.com/light-and-salt/World-Bank-Data-by-Indicators).

## Methodology

- **Baseline Models**: Autoregressive models (AR), linear regression
- **Machine Learning Models**: SVMs and others
- **Evaluation Metrics**:
    - Classification: Accuracy, Precision, Recall, F1-score (predicting if OIR will change)
    - Regression: MAE, RMSE (predicting the magnitude of OIR change)
- **Analysis**: Feature importance, cross-country comparison, error analysis

## How to Use

1. Clone this repository.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Explore the workflow in the `notebooks/` directory for data exploration, modeling, and analysis.
4. For automated experiments, use the scripts in the `src/` directory.

## License

For academic and research purposes only. See LICENSE for details.

## Contact

For questions or collaboration, please contact Jiadong Zhang at dereksodo@gmail.com

## Data Sources

- Main dataset: [World Bank Data by Indicators](https://github.com/light-and-salt/World-Bank-Data-by-Indicators) (GitHub repository)
- Supplementary sources: National central banks, IMF, OECD, United Nations

## Acknowledgements

We gratefully acknowledge the maintainers of [World Bank Data by Indicators](https://github.com/light-and-salt/World-Bank-Data-by-Indicators) for providing comprehensive global economic and demographic data.