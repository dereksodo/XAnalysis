# Cross‑Country Macroeconomic Indicator Forecasting  
_A World‑Bank Panel (1960 – 2020)_

## Project Overview
This repository contains the code, data pipelines, and LaTeX manuscript for a three‑stage machine‑learning framework that forecasts ten key national indicators (life expectancy, urbanisation share, energy use, trade ratios, GDP level & growth, etc.) across G‑20 economies using the World Bank annual panel from **1960 – 2020**.

The research corresponds to the essay **“Predicting National Indicators from World Bank Data: A Machine‑Learning Pipeline (1960–2020)”**.  
The workflow proceeds in three chapters:

| Chapter | Task | Models |
|---------|------|--------|
| 4 | **Cross‑indicator** prediction: each indicator forecast from the other nine | Linear/Ridge/Lasso, SVR, KNN, Random Forest, XGBoost |
| 5 | **Country‑level time‑series** forecasting with lags | Naïve, ARIMA, XGB lag‑2, LSTM lag‑2, rolling variants |
| 6 | **Cross‑country latent‑structure ridge**: couples countries through import/export links | Block‑ridge with tuned \(\lambda_A,\lambda_M\), VAR baseline |

A unified **Guiding Score** (weighted RMSE/STD, \(R^2\), MASE, Directional Accuracy) defines when a forecast is statistically *feasible*.

## Repository Layout
```
essay/          # LaTeX source of the paper
figures/        # Generated plots
data/           # Cleaned World‑Bank panels (CSV)
src/
  DataProcessing/
  Chapter4/
  Chapter5/
  Chapter6/
```

## Data
* Primary source: **[World Bank Data by Indicators](https://github.com/light-and-salt/World-Bank-Data-by-Indicators)**  
* Coverage: G‑20 economies (minus African Union placeholder)  
* Ten indicators kept where non‑missing ratio ≥ 60 %.  
* Missing annual values are **linearly interpolated** along the time axis.


Generated figures appear in `figures/ChapterX/` and are automatically picked up by the LaTeX build.

## Key Results
* Ensemble trees (XGB, RF) beat linear baselines on **7 / 10** indicators;  
  GDP growth remains the hardest, aligning with literature on the “growth‑forecast puzzle”.
* Country‑aware ridge captures modest cross‑border signal—imports & exports drive most spill‑over gains.
* Naïve lag‑1 forecast still wins **Directional Accuracy** for slow‑moving structural features.

## Limitations
See `essay.tex` § Experimental Limitations for details—robustness checks, wider hyper‑parameter search, and probabilistic forecasts are deferred to future work.

## Future Work
The “Summary and Future Work” chapter sketches:
* Block hold‑out & noise‑injection robustness budget
* Quantile XGBoost for 80 % prediction intervals
* Lightweight neural nets (TCN, country‑aware GRU)
* Bayesian shrinkage for calibrated uncertainty

## License
For academic, non‑commercial use only. See `LICENSE`.

## Author
Jiadong Zhang — dereksodo@gmail.com  
CS229 final project (page‑limit accidentally exceeded 🎉). ChatGPT o3 assisted in code and manuscript drafting.