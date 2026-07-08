"""
Download Indian stock data (NSE) via yfinance and save as CSV to archive/indian_stocks/
Covers: Nifty 50 + Nifty Next 50 key stocks + popular midcaps
Run: uv run --with yfinance python3 download_indian_stocks.py
"""
import csv
import sys
import time
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance"); sys.exit(1)

OUTPUT_DIR = Path(__file__).parent / "archive" / "indian_stocks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Indian stocks: symbol → display name ────────────────────────────────────
INDIAN_STOCKS = {
    # Nifty 50 — Large Cap
    "RELIANCE.NS":     "Reliance Industries Ltd",
    "TCS.NS":          "Tata Consultancy Services",
    "HDFCBANK.NS":     "HDFC Bank Ltd",
    "INFY.NS":         "Infosys Ltd",
    "ICICIBANK.NS":    "ICICI Bank Ltd",
    "HINDUNILVR.NS":   "Hindustan Unilever Ltd",
    "SBIN.NS":         "State Bank of India",
    "BAJFINANCE.NS":   "Bajaj Finance Ltd",
    "BHARTIARTL.NS":   "Bharti Airtel Ltd",
    "KOTAKBANK.NS":    "Kotak Mahindra Bank",
    "LT.NS":           "Larsen & Toubro Ltd",
    "AXISBANK.NS":     "Axis Bank Ltd",
    "ASIANPAINT.NS":   "Asian Paints Ltd",
    "MARUTI.NS":       "Maruti Suzuki India",
    "WIPRO.NS":        "Wipro Ltd",
    "ULTRACEMCO.NS":   "UltraTech Cement Ltd",
    "TITAN.NS":        "Titan Company Ltd",
    "SUNPHARMA.NS":    "Sun Pharmaceutical Industries",
    "HCLTECH.NS":      "HCL Technologies Ltd",
    "NESTLEIND.NS":    "Nestle India Ltd",
    "POWERGRID.NS":    "Power Grid Corp of India",
    "TECHM.NS":        "Tech Mahindra Ltd",
    "NTPC.NS":         "NTPC Ltd",
    "ONGC.NS":         "Oil and Natural Gas Corp",
    "JSWSTEEL.NS":     "JSW Steel Ltd",
    "TATAMOTORS.NS":   "Tata Motors Ltd",
    "TATASTEEL.NS":    "Tata Steel Ltd",
    "ADANIENT.NS":     "Adani Enterprises Ltd",
    "ADANIPORTS.NS":   "Adani Ports & SEZ Ltd",
    "COALINDIA.NS":    "Coal India Ltd",
    "HINDALCO.NS":     "Hindalco Industries Ltd",
    "DRREDDY.NS":      "Dr Reddys Laboratories",
    "INDUSINDBK.NS":   "IndusInd Bank Ltd",
    "BAJAJFINSV.NS":   "Bajaj Finserv Ltd",
    "M&M.NS":          "Mahindra & Mahindra Ltd",
    "CIPLA.NS":        "Cipla Ltd",
    "GRASIM.NS":       "Grasim Industries Ltd",
    "DIVISLAB.NS":     "Divi s Laboratories",
    "APOLLOHOSP.NS":   "Apollo Hospitals Enterprise",
    "EICHERMOT.NS":    "Eicher Motors Ltd",
    "HEROMOTOCO.NS":   "Hero MotoCorp Ltd",
    "BRITANNIA.NS":    "Britannia Industries Ltd",
    "TATACONSUM.NS":   "Tata Consumer Products",
    "BPCL.NS":         "Bharat Petroleum Corp",
    "SHRIRAMFIN.NS":   "Shriram Finance Ltd",
    "SBILIFE.NS":      "SBI Life Insurance Co",
    "HDFCLIFE.NS":     "HDFC Life Insurance Co",
    "BAJAJ-AUTO.NS":   "Bajaj Auto Ltd",
    "BEL.NS":          "Bharat Electronics Ltd",
    "TRENT.NS":        "Trent Ltd",
    # Nifty Next 50 & popular midcaps
    "ZOMATO.NS":       "Zomato Ltd",
    "PAYTM.NS":        "One97 Communications (Paytm)",
    "NYKAA.NS":        "FSN E-Commerce Ventures (Nykaa)",
    "POLICYBZR.NS":    "PB Fintech (PolicyBazaar)",
    "DELHIVERY.NS":    "Delhivery Ltd",
    "IRFC.NS":         "Indian Railway Finance Corp",
    "RVNL.NS":         "Rail Vikas Nigam Ltd",
    "HAL.NS":          "Hindustan Aeronautics Ltd",
    "IRCTC.NS":        "Indian Railway Catering & Tourism",
    "MARICO.NS":       "Marico Ltd",
    "PIDILITIND.NS":   "Pidilite Industries Ltd",
    "HAVELLS.NS":      "Havells India Ltd",
    "SIEMENS.NS":      "Siemens Ltd",
    "ABB.NS":          "ABB India Ltd",
    "VOLTAS.NS":       "Voltas Ltd",
    "MPHASIS.NS":      "Mphasis Ltd",
    "LTIM.NS":         "LTIMindtree Ltd",
    "PERSISTENT.NS":   "Persistent Systems Ltd",
    "COFORGE.NS":      "Coforge Ltd",
    "TATAPOWER.NS":    "Tata Power Co Ltd",
    "ADANIGREEN.NS":   "Adani Green Energy Ltd",
    "ADANIPOWER.NS":   "Adani Power Ltd",
    "ATGL.NS":         "Adani Total Gas Ltd",
    "GAIL.NS":         "GAIL India Ltd",
    "IOC.NS":          "Indian Oil Corp Ltd",
    "CANBK.NS":        "Canara Bank",
    "PNB.NS":          "Punjab National Bank",
    "BANKBARODA.NS":   "Bank of Baroda",
    "FEDERALBNK.NS":   "Federal Bank Ltd",
    "IDFCFIRSTB.NS":   "IDFC First Bank Ltd",
    "BANDHANBNK.NS":   "Bandhan Bank Ltd",
    "MOTHERSON.NS":    "Samvardhana Motherson Intl",
    "BALKRISIND.NS":   "Balkrishna Industries Ltd",
    "SOLARINDS.NS":    "Solar Industries India Ltd",
    "DIXON.NS":        "Dixon Technologies India Ltd",
    "AAPL.NS":         None,  # skip — AAPL is US stock
}

# Remove any None values
INDIAN_STOCKS = {k: v for k, v in INDIAN_STOCKS.items() if v}

def save_stock(ticker: str, name: str) -> bool:
    """Download 10 years of history and save as CSV."""
    safe_name = ticker.replace(".NS", "").replace("&", "_").replace("-", "_")
    out_path = OUTPUT_DIR / f"{safe_name}.csv"

    if out_path.exists():
        print(f"  ✓ {ticker} already exists, skipping")
        return True

    try:
        df = yf.download(ticker, period="10y", progress=False,
                         auto_adjust=False, multi_level_index=False)
        if df.empty or len(df) < 10:
            print(f"  ✗ {ticker}: no data")
            return False

        # Drop rows where Close is NaN
        df = df.dropna(subset=["Close"])

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
            for date, row in df.iterrows():
                try:
                    writer.writerow([
                        date.strftime("%Y-%m-%d"),
                        round(float(row["Open"]),  4),
                        round(float(row["High"]),  4),
                        round(float(row["Low"]),   4),
                        round(float(row["Close"]), 4),
                        round(float(row["Adj Close"]) if "Adj Close" in row else float(row["Close"]), 4),
                        int(row["Volume"]) if row["Volume"] == row["Volume"] else 0,
                    ])
                except Exception:
                    continue

        print(f"  ✓ {ticker}: {len(df)} rows → {out_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")
        return False


if __name__ == "__main__":
    print(f"\n📥 Downloading {len(INDIAN_STOCKS)} Indian stocks (NSE)…")
    print(f"   Output: {OUTPUT_DIR}\n")
    ok = fail = 0
    for ticker, name in INDIAN_STOCKS.items():
        if save_stock(ticker, name):
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)  # be polite to Yahoo

    print(f"\n✅ Done: {ok} downloaded, {fail} failed")
    print(f"   Files in {OUTPUT_DIR}: {len(list(OUTPUT_DIR.glob('*.csv')))}")
