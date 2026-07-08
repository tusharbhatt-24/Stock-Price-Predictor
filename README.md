# StockVision AI

A premium, 3D-styled web application for tracking, screening, and predicting stock prices using machine learning.

## Features

- **Dashboard**: Real-time overview with dynamic 3D elements and quick search.
- **AI Predictions**: Ridge-regression based machine learning model trained on historical OHLCV data to predict short-term stock movements.
- **Screener**: Filter US Stocks, US ETFs, and Indian (NSE) stocks by price and percentage change. Quick presets for gainers, losers, and penny stocks.
- **Watchlist & Portfolio**: Track your favorite stocks and simulated portfolio performance.
- **Broad Market Support**: Full support for US Markets and Indian Markets (NSE) out of the box, with built-in currency awareness ($ and ₹).

## Architecture

- **Backend**: Python with Flask, using a custom-built Ridge Regression model (`stock_price_predictor.py`) that uses only the Python standard library.
- **Frontend**: Vanilla HTML/JS/CSS single-page application (SPA) with a sleek glassmorphism design, Chart.js for data visualization, and dynamic routing.
- **Data Collection**: Relies on local CSV archives of historical data (`archive/`), falling back to live data via `yfinance` when needed. Got dataset of stocks from Kaggle

## Setup and Run

This project uses `uv` for isolated package management and execution.

### 1. Run the Web App

Launch the Flask development server:

```bash
uv run --with flask --with yfinance python3 app.py
```

Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 2. Update Indian Stocks Data (Optional)

The project includes a script to bulk-download 10 years of historical data for 80+ top Indian stocks (Nifty 50 + Nifty Next 50):

```bash
uv run --with yfinance python3 download_indian_stocks.py
```

## Legacy CLI Tool

If you prefer to run the prediction model purely via the command line on a specific CSV:

```bash
uv run python3 stock_price_predictor.py --csv path/to/history.csv
```
