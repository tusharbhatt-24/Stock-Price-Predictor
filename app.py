"""
StockVision AI – Full Flask Web Application
Pages: Dashboard, Analytics, Predictions, History, Portfolio, Watchlist, Screener, Notifications, Profile
"""
import sys
import yfinance as yf
import os
import csv
import json
import math
import statistics
import random
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, session
from stock_price_predictor import (
    download_with_yfinance,
    build_dataset,
    train_ridge_regression,
    train_test_split,
    evaluate,
    read_price_csv,
    PriceRow,
)

app = Flask(__name__)
app.secret_key = "stockvision-ai-secret-2024"

ARCHIVE_STOCKS  = Path(__file__).parent / "archive" / "stocks"
ARCHIVE_ETFS    = Path(__file__).parent / "archive" / "etfs"
ARCHIVE_INDIAN  = Path(__file__).parent / "archive" / "indian_stocks"
META_CSV        = Path(__file__).parent / "archive" / "symbols_valid_meta.csv"

# ── Indian stocks metadata (NSE) ─────────────────────────────────────────────
INDIAN_STOCKS_META = {
    "RELIANCE":   {"name": "Reliance Industries Ltd",          "yf": "RELIANCE.NS",  "exchange": "NSE"},
    "TCS":        {"name": "Tata Consultancy Services",        "yf": "TCS.NS",        "exchange": "NSE"},
    "HDFCBANK":   {"name": "HDFC Bank Ltd",                   "yf": "HDFCBANK.NS",   "exchange": "NSE"},
    "INFY":       {"name": "Infosys Ltd",                     "yf": "INFY.NS",        "exchange": "NSE"},
    "ICICIBANK":  {"name": "ICICI Bank Ltd",                  "yf": "ICICIBANK.NS",  "exchange": "NSE"},
    "HINDUNILVR": {"name": "Hindustan Unilever Ltd",          "yf": "HINDUNILVR.NS", "exchange": "NSE"},
    "SBIN":       {"name": "State Bank of India",             "yf": "SBIN.NS",        "exchange": "NSE"},
    "BAJFINANCE": {"name": "Bajaj Finance Ltd",               "yf": "BAJFINANCE.NS", "exchange": "NSE"},
    "BHARTIARTL": {"name": "Bharti Airtel Ltd",               "yf": "BHARTIARTL.NS", "exchange": "NSE"},
    "KOTAKBANK":  {"name": "Kotak Mahindra Bank",             "yf": "KOTAKBANK.NS",  "exchange": "NSE"},
    "LT":         {"name": "Larsen & Toubro Ltd",             "yf": "LT.NS",          "exchange": "NSE"},
    "AXISBANK":   {"name": "Axis Bank Ltd",                   "yf": "AXISBANK.NS",   "exchange": "NSE"},
    "ASIANPAINT": {"name": "Asian Paints Ltd",                "yf": "ASIANPAINT.NS", "exchange": "NSE"},
    "MARUTI":     {"name": "Maruti Suzuki India",             "yf": "MARUTI.NS",      "exchange": "NSE"},
    "WIPRO":      {"name": "Wipro Ltd",                       "yf": "WIPRO.NS",       "exchange": "NSE"},
    "ULTRACEMCO": {"name": "UltraTech Cement Ltd",            "yf": "ULTRACEMCO.NS", "exchange": "NSE"},
    "TITAN":      {"name": "Titan Company Ltd",               "yf": "TITAN.NS",       "exchange": "NSE"},
    "SUNPHARMA":  {"name": "Sun Pharmaceutical Industries",   "yf": "SUNPHARMA.NS",  "exchange": "NSE"},
    "HCLTECH":    {"name": "HCL Technologies Ltd",            "yf": "HCLTECH.NS",     "exchange": "NSE"},
    "NESTLEIND":  {"name": "Nestle India Ltd",                "yf": "NESTLEIND.NS",  "exchange": "NSE"},
    "POWERGRID":  {"name": "Power Grid Corp of India",        "yf": "POWERGRID.NS",  "exchange": "NSE"},
    "TECHM":      {"name": "Tech Mahindra Ltd",               "yf": "TECHM.NS",       "exchange": "NSE"},
    "NTPC":       {"name": "NTPC Ltd",                        "yf": "NTPC.NS",        "exchange": "NSE"},
    "ONGC":       {"name": "Oil and Natural Gas Corp",        "yf": "ONGC.NS",        "exchange": "NSE"},
    "JSWSTEEL":   {"name": "JSW Steel Ltd",                   "yf": "JSWSTEEL.NS",   "exchange": "NSE"},
    "TATAMOTORS": {"name": "Tata Motors Ltd",                 "yf": "TATAMOTORS.NS", "exchange": "NSE"},
    "TATASTEEL":  {"name": "Tata Steel Ltd",                  "yf": "TATASTEEL.NS",  "exchange": "NSE"},
    "ADANIENT":   {"name": "Adani Enterprises Ltd",           "yf": "ADANIENT.NS",   "exchange": "NSE"},
    "ADANIPORTS": {"name": "Adani Ports & SEZ Ltd",           "yf": "ADANIPORTS.NS", "exchange": "NSE"},
    "COALINDIA":  {"name": "Coal India Ltd",                  "yf": "COALINDIA.NS",  "exchange": "NSE"},
    "HINDALCO":   {"name": "Hindalco Industries Ltd",         "yf": "HINDALCO.NS",   "exchange": "NSE"},
    "DRREDDY":    {"name": "Dr. Reddys Laboratories",         "yf": "DRREDDY.NS",    "exchange": "NSE"},
    "INDUSINDBK": {"name": "IndusInd Bank Ltd",               "yf": "INDUSINDBK.NS", "exchange": "NSE"},
    "BAJAJFINSV": {"name": "Bajaj Finserv Ltd",               "yf": "BAJAJFINSV.NS", "exchange": "NSE"},
    "M_M":        {"name": "Mahindra & Mahindra Ltd",         "yf": "M&M.NS",         "exchange": "NSE"},
    "CIPLA":      {"name": "Cipla Ltd",                       "yf": "CIPLA.NS",       "exchange": "NSE"},
    "GRASIM":     {"name": "Grasim Industries Ltd",           "yf": "GRASIM.NS",      "exchange": "NSE"},
    "DIVISLAB":   {"name": "Divi's Laboratories",             "yf": "DIVISLAB.NS",   "exchange": "NSE"},
    "APOLLOHOSP": {"name": "Apollo Hospitals Enterprise",     "yf": "APOLLOHOSP.NS", "exchange": "NSE"},
    "EICHERMOT":  {"name": "Eicher Motors Ltd",               "yf": "EICHERMOT.NS",  "exchange": "NSE"},
    "HEROMOTOCO": {"name": "Hero MotoCorp Ltd",               "yf": "HEROMOTOCO.NS", "exchange": "NSE"},
    "BRITANNIA":  {"name": "Britannia Industries Ltd",        "yf": "BRITANNIA.NS",  "exchange": "NSE"},
    "TATACONSUM": {"name": "Tata Consumer Products",          "yf": "TATACONSUM.NS", "exchange": "NSE"},
    "BPCL":       {"name": "Bharat Petroleum Corp",           "yf": "BPCL.NS",        "exchange": "NSE"},
    "SHRIRAMFIN": {"name": "Shriram Finance Ltd",             "yf": "SHRIRAMFIN.NS", "exchange": "NSE"},
    "SBILIFE":    {"name": "SBI Life Insurance Co",           "yf": "SBILIFE.NS",     "exchange": "NSE"},
    "HDFCLIFE":   {"name": "HDFC Life Insurance Co",          "yf": "HDFCLIFE.NS",    "exchange": "NSE"},
    "BAJAJ_AUTO": {"name": "Bajaj Auto Ltd",                  "yf": "BAJAJ-AUTO.NS",  "exchange": "NSE"},
    "BEL":        {"name": "Bharat Electronics Ltd",          "yf": "BEL.NS",          "exchange": "NSE"},
    "TRENT":      {"name": "Trent Ltd",                       "yf": "TRENT.NS",        "exchange": "NSE"},
    "ZOMATO":     {"name": "Zomato Ltd",                      "yf": "ZOMATO.NS",       "exchange": "NSE"},
    "IRCTC":      {"name": "Indian Railway Catering & Tourism","yf": "IRCTC.NS",       "exchange": "NSE"},
    "HAL":        {"name": "Hindustan Aeronautics Ltd",        "yf": "HAL.NS",          "exchange": "NSE"},
    "IRFC":       {"name": "Indian Railway Finance Corp",      "yf": "IRFC.NS",         "exchange": "NSE"},
    "RVNL":       {"name": "Rail Vikas Nigam Ltd",             "yf": "RVNL.NS",         "exchange": "NSE"},
    "PIDILITIND": {"name": "Pidilite Industries Ltd",          "yf": "PIDILITIND.NS",   "exchange": "NSE"},
    "MARICO":     {"name": "Marico Ltd",                      "yf": "MARICO.NS",        "exchange": "NSE"},
    "HAVELLS":    {"name": "Havells India Ltd",                "yf": "HAVELLS.NS",       "exchange": "NSE"},
    "SIEMENS":    {"name": "Siemens Ltd",                     "yf": "SIEMENS.NS",       "exchange": "NSE"},
    "VOLTAS":     {"name": "Voltas Ltd",                      "yf": "VOLTAS.NS",         "exchange": "NSE"},
    "MPHASIS":    {"name": "Mphasis Ltd",                     "yf": "MPHASIS.NS",        "exchange": "NSE"},
    "LTIM":       {"name": "LTIMindtree Ltd",                 "yf": "LTIM.NS",           "exchange": "NSE"},
    "PERSISTENT": {"name": "Persistent Systems Ltd",          "yf": "PERSISTENT.NS",    "exchange": "NSE"},
    "COFORGE":    {"name": "Coforge Ltd",                     "yf": "COFORGE.NS",        "exchange": "NSE"},
    "TATAPOWER":  {"name": "Tata Power Co Ltd",               "yf": "TATAPOWER.NS",      "exchange": "NSE"},
    "ADANIGREEN": {"name": "Adani Green Energy Ltd",          "yf": "ADANIGREEN.NS",    "exchange": "NSE"},
    "GAIL":       {"name": "GAIL India Ltd",                  "yf": "GAIL.NS",           "exchange": "NSE"},
    "IOC":        {"name": "Indian Oil Corp Ltd",             "yf": "IOC.NS",            "exchange": "NSE"},
    "CANBK":      {"name": "Canara Bank",                     "yf": "CANBK.NS",          "exchange": "NSE"},
    "PNB":        {"name": "Punjab National Bank",            "yf": "PNB.NS",            "exchange": "NSE"},
    "BANKBARODA": {"name": "Bank of Baroda",                  "yf": "BANKBARODA.NS",     "exchange": "NSE"},
    "FEDERALBNK": {"name": "Federal Bank Ltd",                "yf": "FEDERALBNK.NS",     "exchange": "NSE"},
    "IDFCFIRSTB": {"name": "IDFC First Bank Ltd",             "yf": "IDFCFIRSTB.NS",     "exchange": "NSE"},
    "DIXON":      {"name": "Dixon Technologies India Ltd",    "yf": "DIXON.NS",           "exchange": "NSE"},
    "BALKRISIND": {"name": "Balkrishna Industries Ltd",       "yf": "BALKRISIND.NS",      "exchange": "NSE"},
    "MOTHERSON":  {"name": "Samvardhana Motherson Intl",      "yf": "MOTHERSON.NS",       "exchange": "NSE"},
}

# ── In-memory "database" (session-backed for demo) ──────────────────────────
DEFAULT_PORTFOLIO = [
    {"ticker": "AAPL", "shares": 50,  "avg_cost": 165.0},
    {"ticker": "MSFT", "shares": 30,  "avg_cost": 310.0},
    {"ticker": "GOOGL","shares": 15,  "avg_cost": 128.0},
    {"ticker": "NVDA", "shares": 20,  "avg_cost": 450.0},
    {"ticker": "TSLA", "shares": 25,  "avg_cost": 230.0},
]
DEFAULT_WATCHLIST = ["AMZN","META","NFLX","AMD","COIN","PLTR","SOFI"]
DEFAULT_NOTIFICATIONS = [
    {"id":1,"type":"alert","title":"AAPL crossed $310","body":"Apple hit your price target of $310","time":"2m ago","read":False},
    {"id":2,"type":"prediction","title":"MSFT Prediction ready","body":"New 1-day prediction: $425.80","time":"15m ago","read":False},
    {"id":3,"type":"news","title":"Fed Rate Decision","body":"Markets rally after dovish Fed statement","time":"1h ago","read":True},
    {"id":4,"type":"alert","title":"NVDA Volume Spike","body":"NVDA volume 3x above average","time":"2h ago","read":True},
    {"id":5,"type":"prediction","title":"TSLA Prediction","body":"Predicted: $185.40 | Upside: +3.2%","time":"3h ago","read":True},
]

# ── Symbol metadata cache ────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_symbol_meta():
    meta = {}
    if META_CSV.exists():
        with open(META_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("Symbol","").strip()
                if sym:
                    meta[sym] = {
                        "name": row.get("Security Name","").strip(),
                        "exchange": row.get("Listing Exchange","").strip(),
                        "etf": row.get("ETF","N").strip() == "Y",
                    }
    return meta

def get_name(ticker: str) -> str:
    """Return display name for a ticker (US or Indian)."""
    # Check Indian stocks first (bare symbol without .NS)
    t = ticker.upper()
    safe = t.replace(".", "_").replace("-", "_")
    if safe in INDIAN_STOCKS_META:
        return INDIAN_STOCKS_META[safe]["name"]
    if t in INDIAN_STOCKS_META:
        return INDIAN_STOCKS_META[t]["name"]
    meta = load_symbol_meta()
    return meta.get(t, {}).get("name", t)

def is_indian_ticker(ticker: str) -> bool:
    """Return True if ticker belongs to Indian market (NSE)."""
    t = ticker.upper()
    safe = t.replace(".", "_").replace("-", "_")
    return t in INDIAN_STOCKS_META or safe in INDIAN_STOCKS_META

def resolve_indian_ticker(ticker: str) -> tuple[str, str]:
    """
    Given a bare symbol like 'RELIANCE' or 'TCS', return
    (safe_csv_name, yf_ticker) e.g. ('RELIANCE', 'RELIANCE.NS').
    Returns (None, None) if not found.
    """
    t = ticker.upper().replace(".", "_").replace("-", "_")
    if t in INDIAN_STOCKS_META:
        yf_ticker = INDIAN_STOCKS_META[t]["yf"]
        csv_name  = t  # filename without .NS
        return csv_name, yf_ticker
    # Also try the raw ticker key
    raw = ticker.upper()
    if raw in INDIAN_STOCKS_META:
        yf_ticker = INDIAN_STOCKS_META[raw]["yf"]
        return raw, yf_ticker
    return None, None

def get_currency_symbol(ticker: str) -> str:
    """Return ₹ for Indian stocks, $ for US stocks."""
    return "₹" if is_indian_ticker(ticker) else "$"

# ── Local CSV reader ─────────────────────────────────────────────────────────
def load_local_or_yfinance(ticker: str, period: str = "2y") -> list[PriceRow]:
    """Try local archive first (US stocks, ETFs, Indian stocks), fall back to yfinance."""
    period_days = {"6mo":180,"1y":365,"2y":730,"5y":1825,"max":99999}
    days = period_days.get(period, 730)
    cutoff = datetime.now() - timedelta(days=days)

    def trim(rows):
        trimmed = [r for r in rows if r.date >= cutoff]
        return trimmed if len(trimmed) >= 30 else rows[-max(30, days):]

    # 1. Check US stocks and ETFs
    for base in [ARCHIVE_STOCKS, ARCHIVE_ETFS]:
        csv_path = base / f"{ticker.upper()}.csv"
        if csv_path.exists():
            rows = read_price_csv(csv_path)
            if rows:
                return trim(rows)

    # 2. Check Indian stocks (bare symbol → safe filename)
    csv_name, yf_ticker = resolve_indian_ticker(ticker)
    if csv_name is not None:
        # Try local CSV first
        safe_file = csv_name.replace("&", "_").replace("-", "_")
        for candidate in [safe_file, csv_name]:
            csv_path = ARCHIVE_INDIAN / f"{candidate}.csv"
            if csv_path.exists():
                rows = read_price_csv(csv_path)
                if rows:
                    return trim(rows)
        # Fall back to yfinance with .NS suffix
        return download_with_yfinance(yf_ticker, period)

    # 3. If ticker already has .NS suffix, download directly
    if ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO"):
        return download_with_yfinance(ticker, period)

    # 4. Default yfinance fallback
    return download_with_yfinance(ticker, period)

# ── Core prediction engine ───────────────────────────────────────────────────
def run_prediction(ticker: str, period: str = "2y") -> dict:
    rows = load_local_or_yfinance(ticker, period)
    if len(rows) < 30:
        raise ValueError(f"Not enough data ({len(rows)} rows) for '{ticker}'.")

    horizon, alpha, train_ratio = 1, 1.0, 0.8
    dates, features, targets = build_dataset(rows, horizon)
    (dates_train, feat_train, tgt_train,
     dates_test, feat_test, tgt_test) = train_test_split(dates, features, targets, train_ratio)

    model = train_ridge_regression(feat_train, tgt_train, horizon, alpha)
    y_pred_test = model.predict(feat_test)

    split_idx = max(1, min(len(features)-1, int(len(features)*train_ratio)))
    baseline_start = split_idx + horizon
    baseline = [rows[i].close for i in range(baseline_start, baseline_start + len(tgt_test))]
    metrics = evaluate(tgt_test, y_pred_test, baseline)

    next_pred = model.predict([features[-1]])[0]
    latest_close = rows[-1].close
    prior_close = rows[-2].close if len(rows) >= 2 else latest_close
    pct_change = ((latest_close - prior_close) / prior_close) * 100

    chart_slice = list(zip(dates_test, tgt_test, y_pred_test))[-90:]
    chart_data = [{"date": d.strftime("%b %d"), "actual": round(a,2), "predicted": round(p,2)}
                  for d, a, p in chart_slice]

    # Candlestick-style OHLC for last 30 days
    ohlc_data = [
        {"date": r.date.strftime("%b %d"), "open": round(r.open,2),
         "high": round(r.high,2), "low": round(r.low,2), "close": round(r.close,2),
         "volume": int(r.volume)}
        for r in rows[-30:]
    ]

    # Compute MA lines
    closes = [r.close for r in rows]
    ma20 = [round(sum(closes[max(0,i-19):i+1])/min(20,i+1), 2) for i in range(len(closes))]
    ma50 = [round(sum(closes[max(0,i-49):i+1])/min(50,i+1), 2) for i in range(len(closes))]
    ma_chart = [
        {"date": rows[i].date.strftime("%b %d"),
         "close": round(closes[i],2), "ma20": ma20[i], "ma50": ma50[i]}
        for i in range(max(0,len(rows)-90), len(rows))
    ]

    disp_ticker = ticker.upper()
    # For display, strip .NS suffix
    disp_name = disp_ticker.replace(".NS","").replace(".BO","")
    return {
        "ticker": disp_name,
        "name": get_name(disp_name),
        "exchange": "NSE" if is_indian_ticker(disp_name) else "NASDAQ/NYSE",
        "currency": "INR" if is_indian_ticker(disp_name) else "USD",
        "currency_symbol": get_currency_symbol(disp_name),
        "period": period,
        "latest_close": round(latest_close, 2),
        "latest_date": rows[-1].date.strftime("%b %d, %Y"),
        "open": round(rows[-1].open, 2),
        "high": round(rows[-1].high, 2),
        "low": round(rows[-1].low, 2),
        "volume": int(rows[-1].volume),
        "next_pred": round(next_pred, 2),
        "implied_upside": round(((next_pred - latest_close) / latest_close) * 100, 2),
        "pct_change": round(pct_change, 2),
        "rows_loaded": len(rows),
        "train_samples": len(feat_train),
        "holdout_samples": len(feat_test),
        "metrics": {
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "mape": round(metrics["mape"], 4),
            "direction_accuracy": round(metrics["direction_accuracy"], 2),
        },
        "chart_data": chart_data,
        "ohlc_data": ohlc_data,
        "ma_chart": ma_chart,
    }

def quick_quote(ticker: str) -> dict:
    """Fast quote using local CSV (US, ETF, or Indian)."""
    t = ticker.upper()
    display = t.replace(".NS","").replace(".BO","")
    is_indian = is_indian_ticker(display)

    def _make_quote(rows, sym):
        last = rows[-1]; prev = rows[-2]
        chg = ((last.close - prev.close) / prev.close) * 100
        return {"ticker": sym, "name": get_name(sym),
                "price": round(last.close,2), "change": round(chg,2),
                "volume": int(last.volume), "high": round(last.high,2),
                "low": round(last.low,2),
                "currency_symbol": get_currency_symbol(sym),
                "exchange": "NSE" if is_indian_ticker(sym) else "NASDAQ/NYSE"}

    # US stocks and ETFs
    for base in [ARCHIVE_STOCKS, ARCHIVE_ETFS]:
        csv_path = base / f"{t}.csv"
        if csv_path.exists():
            rows = read_price_csv(csv_path)
            if len(rows) >= 2:
                return _make_quote(rows, display)

    # Indian stocks
    csv_name, yf_ticker = resolve_indian_ticker(display)
    if csv_name:
        safe = csv_name.replace("&","_").replace("-","_")
        for candidate in [safe, csv_name]:
            csv_path = ARCHIVE_INDIAN / f"{candidate}.csv"
            if csv_path.exists():
                rows = read_price_csv(csv_path)
                if len(rows) >= 2:
                    return _make_quote(rows, display)
        # Live fetch from yfinance
        try:
            rows = download_with_yfinance(yf_ticker, "5d")
            if len(rows) >= 2:
                return _make_quote(rows, display)
        except Exception:
            pass

    # Generic yfinance fallback
    try:
        rows = download_with_yfinance(t, "5d")
        if len(rows) >= 2:
            return _make_quote(rows, display)
    except Exception:
        pass
    return {"ticker": display, "name": get_name(display),
            "price": 0.0, "change": 0.0, "volume": 0, "high": 0.0, "low": 0.0,
            "currency_symbol": get_currency_symbol(display), "exchange": "NSE" if is_indian else "—"}


# ════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", page="dashboard")

@app.route("/analytics")
def analytics():
    return render_template("index.html", page="analytics")

@app.route("/predictions")
def predictions():
    return render_template("index.html", page="predictions")

@app.route("/history")
def history():
    return render_template("index.html", page="history")

@app.route("/portfolio")
def portfolio():
    return render_template("index.html", page="portfolio")

@app.route("/watchlist")
def watchlist():
    return render_template("index.html", page="watchlist")

@app.route("/screener")
def screener():
    return render_template("index.html", page="screener")

@app.route("/notifications")
def notifications():
    return render_template("index.html", page="notifications")

@app.route("/profile")
def profile():
    return render_template("index.html", page="profile")


# ── API: Predict ─────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    ticker = (data.get("ticker") or "AAPL").strip().upper()
    period = (data.get("period") or "2y").strip()
    try:
        result = run_prediction(ticker, period)
        # Store in prediction history
        hist = session.get("pred_history", [])
        hist.insert(0, {
            "ticker": result["ticker"], "period": period,
            "price": result["latest_close"], "pred": result["next_pred"],
            "upside": result["implied_upside"],
            "mae": result["metrics"]["mae"],
            "dir_acc": result["metrics"]["direction_accuracy"],
            "date": datetime.now().strftime("%b %d, %Y %H:%M"),
        })
        session["pred_history"] = hist[:50]
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


# ── API: Analytics ───────────────────────────────────────────────────────────
@app.route("/api/analytics", methods=["POST"])
def api_analytics():
    data = request.get_json(force=True)
    ticker = (data.get("ticker") or "AAPL").strip().upper()
    period = (data.get("period") or "2y").strip()
    try:
        rows = load_local_or_yfinance(ticker, period)
        closes = [r.close for r in rows]
        returns = [(closes[i]-closes[i-1])/closes[i-1]*100 for i in range(1,len(closes))]

        # Monthly returns
        monthly = {}
        for r in rows:
            key = r.date.strftime("%Y-%m")
            monthly.setdefault(key, []).append(r.close)
        monthly_ret = []
        keys = sorted(monthly.keys())
        for i in range(1, len(keys)):
            prev_closes = monthly[keys[i-1]]
            curr_closes = monthly[keys[i]]
            ret = (curr_closes[-1] - prev_closes[-1]) / prev_closes[-1] * 100
            monthly_ret.append({"month": keys[i], "return": round(ret,2)})

        # Volatility windows
        vol_30 = round(statistics.pstdev(returns[-30:]) if len(returns)>=30 else statistics.pstdev(returns), 4)
        vol_90 = round(statistics.pstdev(returns[-90:]) if len(returns)>=90 else statistics.pstdev(returns), 4)

        # RSI-14
        def compute_rsi(closes_list, period=14):
            if len(closes_list) < period+1: return 50.0
            gains, losses = [], []
            for i in range(1, len(closes_list)):
                delta = closes_list[i] - closes_list[i-1]
                gains.append(max(0,delta)); losses.append(max(0,-delta))
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            if avg_loss == 0: return 100.0
            rs = avg_gain / avg_loss
            return round(100 - 100/(1+rs), 2)

        rsi = compute_rsi(closes)

        # MACD
        def ema(data, n):
            k = 2/(n+1); e = data[0]
            result = [e]
            for v in data[1:]: e = v*k + e*(1-k); result.append(e)
            return result

        ema12 = ema(closes, 12); ema26 = ema(closes, 26)
        macd_line = [round(e12-e26, 4) for e12,e26 in zip(ema12,ema26)]
        signal = ema(macd_line, 9)
        macd_hist = [round(m-s, 4) for m,s in zip(macd_line,signal)]

        # Price distribution (histogram)
        price_min, price_max = min(closes), max(closes)
        n_bins = 10
        bin_size = (price_max - price_min) / n_bins
        hist_bins = [0]*n_bins
        for c in closes:
            idx = min(int((c - price_min) / bin_size), n_bins-1)
            hist_bins[idx] += 1
        distribution = [
            {"range": f"${round(price_min + i*bin_size,0):.0f}-${round(price_min+(i+1)*bin_size,0):.0f}",
             "count": hist_bins[i]}
            for i in range(n_bins)
        ]

        # Chart data: last 90 days OHLCV
        chart_rows = rows[-90:]
        ma20 = [round(sum([r.close for r in rows[max(0,i-19):i+1]])/min(20,i+1),2) for i in range(len(rows))]
        ma50 = [round(sum([r.close for r in rows[max(0,i-49):i+1]])/min(50,i+1),2) for i in range(len(rows))]

        chart_data = [
            {"date": rows[i].date.strftime("%b %d"), "close": round(rows[i].close,2),
             "ma20": ma20[i], "ma50": ma50[i],
             "macd": macd_line[i] if i < len(macd_line) else 0,
             "signal": signal[i] if i < len(signal) else 0,
             "macd_hist": macd_hist[i] if i < len(macd_hist) else 0,
             "volume": int(rows[i].volume)}
            for i in range(max(0,len(rows)-90), len(rows))
        ]

        return jsonify({"ok": True, "data": {
            "ticker": ticker,
            "name": get_name(ticker),
            "latest_close": round(closes[-1],2),
            "pct_change": round(returns[-1] if returns else 0, 2),
            "vol_30": vol_30,
            "vol_90": vol_90,
            "rsi": rsi,
            "52w_high": round(max(closes[-252:] if len(closes)>=252 else closes),2),
            "52w_low": round(min(closes[-252:] if len(closes)>=252 else closes),2),
            "avg_volume": int(sum(r.volume for r in rows[-30:]) / min(30,len(rows))),
            "monthly_returns": monthly_ret[-18:],
            "distribution": distribution,
            "chart_data": chart_data,
        }})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


# ── API: History ─────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def api_history():
    hist = session.get("pred_history", [])
    return jsonify({"ok": True, "history": hist})

@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    session["pred_history"] = []
    return jsonify({"ok": True})


# ── API: Portfolio ────────────────────────────────────────────────────────────
@app.route("/api/portfolio", methods=["GET"])
def api_portfolio():
    holdings = session.get("portfolio", DEFAULT_PORTFOLIO.copy())
    enriched = []
    total_value = 0; total_cost = 0
    for h in holdings:
        q = quick_quote(h["ticker"])
        cost_basis = h["shares"] * h["avg_cost"]
        mkt_value  = h["shares"] * q["price"]
        gain = mkt_value - cost_basis
        gain_pct = (gain / cost_basis * 100) if cost_basis else 0
        total_value += mkt_value; total_cost += cost_basis
        enriched.append({**h, "price": q["price"], "change": q["change"],
                          "name": q["name"], "mkt_value": round(mkt_value,2),
                          "cost_basis": round(cost_basis,2),
                          "gain": round(gain,2), "gain_pct": round(gain_pct,2)})
    return jsonify({"ok": True, "holdings": enriched,
                    "total_value": round(total_value,2),
                    "total_cost": round(total_cost,2),
                    "total_gain": round(total_value-total_cost,2),
                    "total_gain_pct": round((total_value-total_cost)/total_cost*100,2) if total_cost else 0})

@app.route("/api/portfolio/add", methods=["POST"])
def api_portfolio_add():
    data = request.get_json(force=True)
    holdings = session.get("portfolio", DEFAULT_PORTFOLIO.copy())
    ticker = data["ticker"].upper()
    shares = float(data.get("shares",1))
    avg_cost = float(data.get("avg_cost",0))
    existing = next((h for h in holdings if h["ticker"]==ticker), None)
    if existing:
        total_shares = existing["shares"] + shares
        existing["avg_cost"] = (existing["shares"]*existing["avg_cost"] + shares*avg_cost) / total_shares
        existing["shares"] = total_shares
    else:
        holdings.append({"ticker":ticker,"shares":shares,"avg_cost":avg_cost})
    session["portfolio"] = holdings
    return jsonify({"ok": True})

@app.route("/api/portfolio/remove", methods=["POST"])
def api_portfolio_remove():
    data = request.get_json(force=True)
    ticker = data["ticker"].upper()
    holdings = session.get("portfolio", DEFAULT_PORTFOLIO.copy())
    holdings = [h for h in holdings if h["ticker"] != ticker]
    session["portfolio"] = holdings
    return jsonify({"ok": True})


# ── API: Watchlist ────────────────────────────────────────────────────────────
@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    tickers = session.get("watchlist", DEFAULT_WATCHLIST.copy())
    quotes = [quick_quote(t) for t in tickers]
    return jsonify({"ok": True, "watchlist": quotes})

@app.route("/api/watchlist/add", methods=["POST"])
def api_watchlist_add():
    data = request.get_json(force=True)
    ticker = data["ticker"].upper()
    wl = session.get("watchlist", DEFAULT_WATCHLIST.copy())
    if ticker not in wl:
        wl.append(ticker)
    session["watchlist"] = wl
    return jsonify({"ok": True})

@app.route("/api/watchlist/remove", methods=["POST"])
def api_watchlist_remove():
    data = request.get_json(force=True)
    ticker = data["ticker"].upper()
    wl = session.get("watchlist", DEFAULT_WATCHLIST.copy())
    wl = [t for t in wl if t != ticker]
    session["watchlist"] = wl
    return jsonify({"ok": True})


# ── API: Screener ─────────────────────────────────────────────────────────────
@app.route("/api/screener", methods=["POST"])
def api_screener():
    data = request.get_json(force=True)
    min_change = float(data.get("min_change", -999))
    max_change = float(data.get("max_change", 999))
    min_price  = float(data.get("min_price", 0))
    max_price  = float(data.get("max_price", 999999))
    asset_type = data.get("asset_type", "stocks")  # stocks | etfs | all
    limit      = int(data.get("limit", 40))

    if asset_type == "indian":
        csv_files = list(ARCHIVE_INDIAN.glob("*.csv")) if ARCHIVE_INDIAN.exists() else []
        # Also include live Indian stocks if no local files yet
        if not csv_files:
            # Return metadata from our embedded dict
            results = []
            for sym, meta in list(INDIAN_STOCKS_META.items())[:limit]:
                results.append({"ticker": sym, "name": meta["name"],
                                 "price": 0.0, "change": 0.0, "volume": 0,
                                 "high": 0.0, "low": 0.0,
                                 "currency_symbol": "₹", "exchange": "NSE",
                                 "note": "Download in progress"})
            return jsonify({"ok": True, "results": results})
    elif asset_type == "etfs":
        csv_files = list(ARCHIVE_ETFS.glob("*.csv"))
    elif asset_type == "all":
        csv_files = (list(ARCHIVE_STOCKS.glob("*.csv")) +
                     list(ARCHIVE_ETFS.glob("*.csv")) +
                     (list(ARCHIVE_INDIAN.glob("*.csv")) if ARCHIVE_INDIAN.exists() else []))
    else:  # stocks (default)
        csv_files = list(ARCHIVE_STOCKS.glob("*.csv"))

    random.shuffle(csv_files)
    results = []

    for csv_path in csv_files:
        if len(results) >= limit: break
        try:
            rows = read_price_csv(csv_path)
            if len(rows) < 2: continue
            last = rows[-1]; prev = rows[-2]
            chg = (last.close - prev.close) / prev.close * 100
            if not (min_change <= chg <= max_change): continue
            if not (min_price <= last.close <= max_price): continue
            ticker = csv_path.stem
            is_ind = csv_path.parent.name == "indian_stocks"
            results.append({
                "ticker": ticker,
                "name": get_name(ticker),
                "price": round(last.close,2),
                "change": round(chg,2),
                "volume": int(last.volume),
                "high": round(last.high,2),
                "low": round(last.low,2),
                "currency_symbol": "₹" if is_ind else "$",
                "exchange": "NSE" if is_ind else "US",
            })
        except Exception:
            continue

    results.sort(key=lambda x: abs(x["change"]), reverse=True)
    return jsonify({"ok": True, "results": results[:limit]})


# ── API: Notifications ────────────────────────────────────────────────────────
@app.route("/api/notifications", methods=["GET"])
def api_notifications():
    notifs = session.get("notifications", DEFAULT_NOTIFICATIONS.copy())
    return jsonify({"ok": True, "notifications": notifs,
                    "unread": sum(1 for n in notifs if not n["read"])})

@app.route("/api/notifications/read", methods=["POST"])
def api_notifications_read():
    data = request.get_json(force=True)
    nid = data.get("id")
    notifs = session.get("notifications", DEFAULT_NOTIFICATIONS.copy())
    for n in notifs:
        if nid is None or n["id"] == nid:
            n["read"] = True
    session["notifications"] = notifs
    return jsonify({"ok": True})

@app.route("/api/notifications/add_alert", methods=["POST"])
def api_notifications_add_alert():
    data = request.get_json(force=True)
    notifs = session.get("notifications", DEFAULT_NOTIFICATIONS.copy())
    new_id = max((n["id"] for n in notifs), default=0) + 1
    notifs.insert(0, {
        "id": new_id,
        "type": "alert",
        "title": f"{data.get('ticker','?')} Price Alert",
        "body": f"Alert set at ${data.get('price','?')}",
        "time": "just now",
        "read": False,
    })
    session["notifications"] = notifs
    return jsonify({"ok": True})


# ── API: Profile ──────────────────────────────────────────────────────────────
@app.route("/api/profile", methods=["GET"])
def api_profile():
    profile_data = session.get("profile", {
        "name": "Alex Morgan",
        "email": "alex.morgan@stockvision.ai",
        "plan": "Pro",
        "joined": "Jan 2024",
        "avatar_color": "#6366f1",
        "risk_tolerance": "Moderate",
        "preferred_sectors": ["Technology","Finance","Healthcare"],
        "alert_email": True,
        "alert_push": True,
        "dark_mode": False,
    })
    return jsonify({"ok": True, "profile": profile_data})

@app.route("/api/profile/update", methods=["POST"])
def api_profile_update():
    data = request.get_json(force=True)
    profile_data = session.get("profile", {})
    profile_data.update(data)
    session["profile"] = profile_data
    return jsonify({"ok": True, "profile": profile_data})


# ── API: Quote (quick) ────────────────────────────────────────────────────────
@app.route("/api/quote/<ticker>", methods=["GET"])
def api_quote(ticker):
    q = quick_quote(ticker.upper())
    return jsonify({"ok": True, **q})

@app.route("/api/search", methods=["GET"])
def api_search():
    q = request.args.get("q","").upper()
    if len(q) < 1:
        return jsonify({"ok": True, "results": []})

    results = []

    # 1. Indian stocks (searched first so Indian tickers surface fast)
    for sym, info in INDIAN_STOCKS_META.items():
        if sym.startswith(q) or q in info["name"].upper():
            results.append({"ticker": sym, "name": info["name"][:55],
                            "etf": False, "exchange": "NSE", "flag": "🇮🇳"})
        if len(results) >= 8:
            break

    # 2. US stocks
    meta = load_symbol_meta()
    us_results = [
        {"ticker": k, "name": v["name"][:55], "etf": v["etf"],
         "exchange": v.get("exchange","US"), "flag": "🇺🇸"}
        for k,v in meta.items()
        if k.startswith(q) or q in v["name"].upper()
    ][:max(1, 15 - len(results))]
    results.extend(us_results)

    return jsonify({"ok": True, "results": results[:15]})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
