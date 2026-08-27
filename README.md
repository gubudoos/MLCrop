# MLCrop

Machine learning solution for the Kaggle **[Crop Yield Prediction Challenge](https://www.kaggle.com/competitions/crop-yield-prediction-challenge/overview)** — a regression task that predicts crop yield (tons/hectare, `yield_tpha`) from soil, weather, and field-management features.

## Project structure

```
MLCrop/
├── algo_main.ipynb    # ⭐ Main notebook — feature engineering, model comparison, and tuned CatBoost pipeline
├── crop.ipynb              # Exploratory data analysis notebook (pandas / seaborn / matplotlib)
├── crop_yield_train.csv    # Training data (features + yield_tpha target)
├── crop_yield_test.csv     # Test data (features only, for Kaggle submission)
├── sample_submission.csv   # Kaggle submission format template
└── README.md
```

## Main file: `algo_main.ipynb`

This is the core modeling notebook. It loads `crop_yield_train.csv` / `crop_yield_test.csv` and runs the full pipeline below.

**Feature engineering**
- Parses `harvest_date` into `harvest_month`, `harvest_dayofyear`, and `harvest_week`.
- Normalizes soil metrics (`soil_ph`, `soil_moisture`, `nitrogen_content`, `phosphorus_content`, `potassium_content`) **per region**, using the train-set region mean/std applied consistently to both train and test.
- Adds an engineered `rainfall_sunlight_index` feature (`total_rainfall × sunlight_hours`).
- Drops the raw `harvest_date` column once its parts have been extracted.

**Preprocessing**
- `ColumnTransformer` combining `OneHotEncoder` (categorical columns: `crop_type`, `region`, `season`, `field_id`) and `StandardScaler` (numeric columns).
- 5-fold `KFold` cross-validation with a custom RMSE scorer (`make_scorer` around `mean_squared_error`).

**Model comparison (Part 1)** — each wrapped in a `Pipeline` and tuned with `GridSearchCV`:

| Algorithm | Best CV RMSE | Best Parameters |
|---|---|---|
| Ridge | 0.6443 | `alpha=100` |
| Lasso | 0.6429 | `alpha=0.01` |
| ElasticNet | 0.6428 | `alpha=0.01, l1_ratio=0.5` |
| Random Forest | 0.6526 | `max_depth=10, min_samples_split=5, n_estimators=500` |
| XGBoost | 0.6446 | `colsample_bytree=0.8, learning_rate=0.05, max_depth=4, n_estimators=500, reg_lambda=10, subsample=1.0` |
| LightGBM | 0.6507 | `colsample_bytree=0.8, learning_rate=0.05, max_depth=4, n_estimators=500, num_leaves=31, reg_lambda=10, subsample=0.8` |
| **CatBoost** | **0.6368 (best)** | `depth=4, iterations=500, l2_leaf_reg=10, learning_rate=0.05` |

**Final model (Part 2) — tuned CatBoost:** CatBoost is selected as the best base algorithm and refined further with `RandomizedSearchCV` (3-fold CV, 20 sampled combinations), passing categorical columns natively via `cat_features` instead of one-hot encoding. Best result: **CV RMSE ≈ 0.6355**, train RMSE ≈ 0.5960, with `learning_rate=0.05, l2_leaf_reg=1, iterations=400, depth=4, border_count=32`. The notebook also prints the full per-fold RMSE breakdown for every sampled hyperparameter combination.

> `crop.ipynb` is a separate, earlier exploratory-analysis notebook (distributions, correlations, and visualizations of the raw features) and isn't required to reproduce the modeling results above.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gubudoos/MLCrop.git
   cd MLCrop
   ```

2. **Create and activate a virtual environment** (Python 3.10+ recommended)
   ```bash
   conda create -n mlcrop python=3.10 -y
   conda activate mlcrop
   ```
   or with `venv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install the required packages**
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm catboost jupyter
   ```
   To also run the exploratory notebook (`crop.ipynb`):
   ```bash
   pip install matplotlib seaborn
   ```

4. **Launch Jupyter and open the main notebook**
   ```bash
   jupyter notebook "algo_main.ipynb"
   ```
   Run all cells top to bottom — the CSV files are read from the repository root (`crop_yield_train.csv`, `crop_yield_test.csv`), so no path changes are needed if you run the notebook from the cloned repo directory.

> **Note (LightGBM on Windows):** the notebook was originally run on Windows; if `lgb.LGBMRegressor` raises multiprocessing/subprocess errors during `GridSearchCV`, try setting `n_jobs=1` on the LightGBM grid search or running from a plain terminal instead of certain IDE consoles — this is an environment quirk unrelated to the modeling code.

## Package versions

The notebook doesn't pin exact library versions, so install the latest compatible releases (all are widely backward-compatible for this pipeline):

| Package | Purpose | Suggested version |
|---|---|---|
| Python | Runtime | 3.10 – 3.13 (developed with 3.13.5) |
| pandas | Data loading & manipulation | latest |
| numpy | Numeric operations | latest |
| scikit-learn | Preprocessing, pipelines, `GridSearchCV`/`RandomizedSearchCV`, linear & ensemble models | latest |
| xgboost | Gradient boosting (`XGBRegressor`) | latest |
| lightgbm | Gradient boosting (`LGBMRegressor`) | latest |
| catboost | Gradient boosting with native categorical support (`CatBoostRegressor`) | latest |
| matplotlib | Plotting *(`crop.ipynb` only)* | latest |
| seaborn | Statistical plotting *(`crop.ipynb` only)* | latest |
| jupyter | Notebook environment | latest |

If you want a fully reproducible environment, freeze your installed versions after a successful run:
```bash
pip freeze > requirements.txt
```

## Data

- **`crop_yield_train.csv`** — training rows with features (`soil_ph`, `soil_moisture`, `avg_temperature`, `total_rainfall`, `fertilizer_amount`, `pesticide_usage`, `sunlight_hours`, `nitrogen_content`, `phosphorus_content`, `potassium_content`, `irrigation_frequency`, `crop_type`, `region`, `season`, `harvest_date`, `field_id`) plus the target `yield_tpha`.
- **`crop_yield_test.csv`** — same features, no target, used to generate predictions for Kaggle submission.
- **`sample_submission.csv`** — expected submission format (`id, yield_tpha`).
