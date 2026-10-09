# Store Item Demand Forecasting

14-day demand forecasting for 10 stores x 50 items, built as a client-style project: a grocery chain owner wants to order the right quantity per branch and product, with fewer empty shelves and less wasted stock.

**Live demo:** _add your Streamlit link here_

## Problem

Ordering is currently done by gut feeling at each branch. Under-ordering means lost sales (empty shelves); over-ordering means excess stock. The goal is a forecast of daily unit sales for the next 14 days per store and item.

## Data

Kaggle "Store Item Demand Forecasting Challenge": daily unit sales for 10 stores x 50 items, 2013-01-01 to 2017-12-31 (about 913,000 rows, columns: `date, store, item, sales`).

The data has no prices, promotions, or stock levels, so stock-outs and promotions are not modeled.

## Approach

1. **EDA:** strong weekly seasonality (Sunday highest, Monday lowest, about 50% gap), strong yearly seasonality (July highest, January lowest, about 89% gap), and a growth trend (about 35% from 2013 to 2017).
2. **Baseline:** sales of the same weekday one year earlier (364 days back).
3. **Model:** XGBoost on calendar features (store, item, day of week, month, day, year) and lag/rolling features.
4. **No leakage:** every lag is at least 14 days, matching the 14-day forecast horizon, so the model never uses sales that would not be known at forecast time.
5. **Evaluation:** time-based split, repeated over 5 backtest windows (90 days each) instead of one test period.

Features: `lag_14, lag_21, lag_28, lag_35, lag_42, lag_364`, 28-day rolling mean and std (shifted by 14), mean of the same weekday over the last 4 weeks, plus the calendar features.

## Results

Metric: WAPE (total absolute error divided by total sales). Lower is better.

| Backtest window | Baseline | XGBoost |
|---|---|---|
| 2016-10 | 16.56% | 11.33% |
| 2017-01 | 16.43% | 12.24% |
| 2017-04 | 13.68% | 10.27% |
| 2017-07 | 13.53% | 10.00% |
| 2017-10 | 15.25% | 11.17% |
| **Average** | **15.09%** | **11.00%** |

The model beats the baseline in all 5 windows, with about 27% lower error on average. Bias stays within +/-1.6% in every window (the baseline under-forecasts by about 4% because sales keep growing).

Error is spread fairly evenly across stores (about 10% to 14%) and days of the week. Items with lower sales have higher relative error (correlation of about -0.96 between item WAPE and average sales), which is expected noise on small numbers rather than a model problem.

### Lost sales vs. excess stock (Oct-Dec 2017 test window)

| | Lost sales | Excess stock |
|---|---|---|
| Baseline | 9.7% | 5.6% |
| Model | 5.6% | 5.5% |

Lost sales drop by about 42% without increasing excess stock.

## Limitations

- The simulation assumes ordering exactly the forecast, with no safety stock, and treats all items equally. Real costs differ (perishables spoil, staples lose customers when out of stock).
- No prices or costs in the data, so the impact is in units and percentages, not currency.
- January is the weakest window: WAPE 12.24% and a slightly positive bias (+1.0%).
- The demo uses a fixed history and does not retrain or update automatically.

## Possible next steps

- Quantile forecasts (a higher quantile for staples, a lower one for perishables) to control stock risk per product type.
- Promotions, holidays, and stock-out handling when real data is available.
- Hyperparameter tuning and comparison with LightGBM.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `store item.ipynb`: full analysis, baseline, model, and backtest
- `app.py`: Streamlit demo
- `model.json`: trained XGBoost model
- `history.csv`: last 400 days of sales used as input for the demo
- `requirements.txt`: dependencies
