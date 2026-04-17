# zrive-ds

Base repository for the [Zrive Applied Data Science](https://zriveapp.com/cursos/zrive-applied-data-science) program — a hands-on, end-to-end course covering the full data science lifecycle: from raw data exploration and feature engineering through model training, evaluation, and production deployment.

---

## Setup

For consistency, everyone uses **Python 3.11.0**, managed with `pyenv` for version control and `poetry` for virtual environments and dependencies.

> ⚠️ **Never use your system Python.** Always work inside a dedicated virtual environment to avoid cross-project conflicts.

### Steps

```bash
# 1. Install Python 3.11.0 via pyenv
pyenv install 3.11.0

# 2. Set the local version inside the repo
pyenv local 3.11.0

# 3. Verify (should show 3.11.0 with an asterisk)
pyenv versions

# 4. Install dependencies
poetry install

# 5. Add new dependencies as needed
poetry add <package>
```

> **Ubuntu/WSL users:** Poetry may not pick up the local `pyenv` version automatically. Force it with:
> ```bash
> poetry env use 3.11.0
> ```

### Code quality tools

| Tool | Purpose |
|------|---------|
| `black` | Opinionated, zero-config code formatter |
| `flake8` | Linter for syntax errors, bugs, and style issues |
| `mypy` | Static type checker (available but not enforced in `make test`) |

Run linting and tests with:

```bash
make lint
make test
```

---

## Repository Structure

```
zrive-ds
├── bin/
├── data/
├── src/
│   ├── __init__.py
│   ├── module_1/                        # External API integration & climate data analysis
│   │   ├── module_1_meteo_api.py
│   │   └── climate_evolution.png
│   ├── module_2/                        # Exploratory Data Analysis (EDA)
│   │   └── eda.ipynb
│   ├── module_3/                        # Binary classification with Logistic Regression
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── utils.py
│   │   ├── tdd.ipynb
│   │   ├── models/
│   │   └── solution/
│   │       ├── train_2.py
│   │       ├── inference_2.py
│   │       ├── utils_2.py
│   │       └── linear_push.ipynb
│   ├── module_4/                        # Gradient Boosted Trees & ML pipeline design
│   │   ├── train.py
│   │   ├── inferences.py
│   │   ├── utils.py
│   │   ├── push_notifications.ipynb
│   │   ├── models/
│   │   ├── predictions/
│   │   └── solution/
│   │       ├── module_4_fit.py
│   │       ├── module_4_predict.py
│   │       ├── push_model.py
│   │       ├── utils.py
│   │       └── push_notifications.ipynb
│   ├── module_5/                        # Model diagnosis & equity portfolio optimization
│   │   └── solution/
│   │       └── module5.ipynb
│   └── module_6/                        # REST API deployment & production system design
│       ├── basket_model_api.md
│       ├── basket_model/
│       │   ├── app.py
│       │   ├── dependencies/
│       │   ├── exceptions/
│       │   ├── models/
│       │   │   └── schemas.py
│       │   ├── routers/
│       │   │   ├── predict.py
│       │   │   ├── metrics.py
│       │   │   └── status.py
│       │   ├── services/
│       │   │   ├── basket_model.py
│       │   │   └── feature_store.py
│       │   └── utils/
│       │       ├── features.py
│       │       └── loaders.py
│       └── solution/
│           ├── app.py
│           ├── routes.py
│           ├── metrics.py
│           ├── data_model.py
│           ├── exceptions.py
│           ├── basket_model/
│           ├── handlers/
│           ├── utils/
│           ├── postman/
│           ├── profiling/
│           │   ├── attack.sh
│           │   ├── report.sh
│           │   ├── payload.json
│           │   └── targets.txt
│           └── system_design/
│               └── system_design.png
├── tests/
├── Makefile
├── pyproject.toml
├── poetry.lock
├── setup.cfg
└── README.md
```

---

## Modules

### Module 1 — External API Integration & Climate Data Analysis

Connects to the [Open-Meteo Archive API](https://archive-api.open-meteo.com) to retrieve historical daily weather data (temperature, precipitation, wind speed) for Madrid, London, and Rio de Janeiro across 2010–2020.

Key components:
- `make_api_call_with_cool_off`: resilient HTTP client with exponential backoff and retry logic for connection, HTTP, and request errors.
- `get_data_meteo`: builds and fires parameterized API requests per city.
- `get_processed_df_by_month`: aggregates daily data into monthly statistics (mean, min, max, std) grouped by city.
- `plot_df`: generates a multi-panel visualization saved as `climate_evolution.png`.

**Skills:** API integration, error handling, pandas aggregation, matplotlib visualization.

---

### Module 2 — Exploratory Data Analysis (EDA)

Deep-dive EDA on a grocery e-commerce dataset stored in S3 (Parquet format). Covers five datasets: `orders`, `regulars`, `abandoned_carts`, `inventory`, and `users`.

Key analyses:
- User segmentation and behaviour patterns (order frequency, regular items, household composition).
- Product prevalence: comparing inventory distribution vs. products saved as regulars, with vendor and product-type breakdowns.
- Regional analysis of user segments across UK NUTS1 regions.
- Price distribution comparison between regularly purchased items and overall inventory.

**Skills:** Pandas, Seaborn, exploratory statistics, business insight extraction.

---

### Module 3 — Binary Classification with Logistic Regression

Trains a push notification propensity model to predict whether a user will purchase a product, using a Ridge-regularized Logistic Regression.

Key components:
- `build_feature_frame` / `push_relevant_orders`: filters orders with at least 5 purchased items and constructs the feature set (`ordered_before`, `abandoned_before`, `global_popularity`, `set_as_regular`).
- `ridge_model_selection`: performs time-aware train/validation splitting (by order date) and grid-searches over regularization strengths (`C ∈ {1e-8, 1e-6, 1e-4, 1e-2}`), selecting the best model by PR AUC.
- `evaluate_model`: reports both PR AUC and ROC AUC.
- `save_model`: persists the best model with a timestamped filename using `joblib`.
- `inference.py`: loads a saved model and runs evaluation on the full dataset.

**Skills:** Sklearn pipelines, temporal train/test split, model selection, PR AUC evaluation.

---

### Module 4 — Gradient Boosted Trees & ML Pipeline Design

Extends the propensity model to XGBoost with a production-grade, event-driven pipeline architecture.

Key components:
- `PushModel`: encapsulates an `XGBClassifier` with feature extraction, probabilistic prediction, and threshold-based binary output.
- `handler_fit`: orchestrates training end-to-end — loads data, instantiates `PushModel`, fits, and persists the model via `joblib`. Handles path collisions with timestamp suffixes.
- `handler_predict`: loads a serialized model and runs inference on incoming user feature dictionaries.
- `utils.py`: shared data loading and feature frame construction with order-size filtering.

The module also includes a Jupyter notebook (`push_notifications.ipynb`) exploring the full training and evaluation workflow interactively.

**Skills:** XGBoost, ML pipeline abstraction, event-driven handler design, model serialization.

---

### Module 5 — Model Diagnosis & Equity Portfolio Optimization

Applies the full model diagnosis and improvement cycle to a financial dataset, with the goal of building an equity portfolio that outperforms the S&P 500.

Problem setup:
- Features: quarterly financial reports (fundamentals + technical indicators from Macrotrends).
- Target: binary — whether a stock outperforms the S&P 500 over the following year (`stock_change_div_365 - sp500_change_365 > 0`).
- Metric: weighted portfolio return of the top-N predicted tickers per quarter.

Key components:
- Time-series train/test split by `execution_date` (no data leakage).
- LightGBM classifier trained across rolling quarterly windows.
- Model diagnosis loop: performance tracking across periods, feature importance analysis, hyperparameter tuning.

**Skills:** LightGBM, financial ML, temporal validation, model diagnosis, performance attribution.

---

### Module 6 — REST API Deployment & Production System Design

Productionizes the basket recommendation model as a fully structured REST API using FastAPI, following production software engineering best practices.

API endpoints:
| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/status` | Health check |
| `POST` | `/predict` | Returns product recommendations for a given user/order |
| `GET` | `/metrics` | Exposes model performance and API usage metrics |

Architecture:
- **Routers** (`routers/`): request routing and HTTP layer.
- **Services** (`services/`): business logic — `basket_model.py` for inference, `feature_store.py` for feature retrieval.
- **Utils** (`utils/`): feature engineering (`features.py`) and data loaders (`loaders.py`).
- **Models** (`models/schemas.py`): Pydantic request/response schemas.
- **Exceptions** (`exceptions/`): structured error handling.

Also includes:
- `basket_model_api.md`: full API documentation.
- `postman/`: Postman collection for manual endpoint testing.
- `profiling/`: shell scripts (`attack.sh`, `report.sh`) for load testing the API and analyzing throughput and latency under stress.
- `system_design/`: Excalidraw diagram illustrating the end-to-end production system design.

**Skills:** FastAPI, REST API design, Pydantic, load testing, system design, software engineering best practices.

---

## Data

All datasets are stored in AWS S3 and should be downloaded locally before working. **Do not commit data to the repository.**

### Setup

```bash
# Configure AWS credentials (store keys in .env, never commit them)
aws configure

# List available datasets
aws s3 ls s3://zrive-ds-data/ --recursive
```

### Available datasets

**`groceries/sampled-datasets/`**

| File | Description |
|------|-------------|
| `orders.parquet` | Order history — each row is an order with a list of purchased `item_ids` |
| `regulars.parquet` | Items users have set as regular purchases |
| `abandoned_carts.parquet` | Items added to basket but not purchased |
| `inventory.parquet` | Product metadata (price, type, vendor, etc.) |
| `users.parquet` | User attributes (segment, region, household composition) |

**`groceries/box_builder_dataset/`**

Full-size dataset where each row is a `(order, user, product)` triplet. Designed to train a basket pre-population model — predicting which products to include in a user's next basket before they arrive.

- Binary label `outcome`: 1 if the product was purchased in that order, 0 otherwise.
- Excludes first-time orders and long-tail products.
- Can also be used for: time-series sales forecasting, repurchase probability modeling, and next-basket value prediction.

### Local download (recommended)

```python
import boto3

s3 = boto3.client("s3")
s3.download_file("zrive-ds-data", "groceries/sampled-datasets/orders.parquet", "data/orders.parquet")
```

---

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
