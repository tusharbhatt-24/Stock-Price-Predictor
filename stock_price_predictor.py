#!/usr/bin/env python3
"""Train a stock-price predictor from historical OHLCV data.

The default model is ridge-regularized linear regression implemented with the
Python standard library, so the project works even in a fresh environment.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REQUIRED_PRICE_COLUMNS = ("date", "open", "high", "low", "close", "volume")
FEATURE_NAMES = (
    "close",
    "open",
    "high",
    "low",
    "volume",
    "daily_return",
    "range_pct",
    "lag_1_return",
    "lag_2_return",
    "lag_3_return",
    "lag_5_return",
    "ma_5_ratio",
    "ma_10_ratio",
    "volatility_5",
    "volume_5_ratio",
    "day_of_week",
)


@dataclass(frozen=True)
class PriceRow:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StandardScaler:
    means: list[float]
    scales: list[float]

    @classmethod
    def fit(cls, rows: list[list[float]]) -> "StandardScaler":
        columns = list(zip(*rows))
        means = [statistics.fmean(column) for column in columns]
        scales = []
        for column in columns:
            std = statistics.pstdev(column)
            scales.append(std if std > 1e-12 else 1.0)
        return cls(means=means, scales=scales)

    def transform_one(self, row: list[float]) -> list[float]:
        return [(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)]

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [self.transform_one(row) for row in rows]


@dataclass
class RidgeLinearRegression:
    feature_names: tuple[str, ...]
    coefficients: list[float]
    intercept: float
    scaler: StandardScaler
    horizon: int

    def predict_one(self, raw_features: list[float]) -> float:
        features = self.scaler.transform_one(raw_features)
        return self.intercept + sum(weight * value for weight, value in zip(self.coefficients, features))

    def predict(self, raw_features: list[list[float]]) -> list[float]:
        return [self.predict_one(row) for row in raw_features]


def parse_date(value: str) -> datetime:
    cleaned = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(cleaned)


def normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "_")


def read_price_csv(path: Path) -> list[PriceRow]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")

        header_map = {normalize_header(name): name for name in reader.fieldnames}
        if "adj_close" in header_map and "close" not in header_map:
            header_map["close"] = header_map["adj_close"]

        missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in header_map]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    PriceRow(
                        date=parse_date(row[header_map["date"]]),
                        open=float(row[header_map["open"]]),
                        high=float(row[header_map["high"]]),
                        low=float(row[header_map["low"]]),
                        close=float(row[header_map["close"]]),
                        volume=float(row[header_map["volume"]]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Could not parse row {line_number} in {path}: {exc}") from exc

    rows.sort(key=lambda item: item.date)
    return rows


def download_with_yfinance(ticker: str, period: str) -> list[PriceRow]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Ticker downloads require yfinance. Install it with: python3 -m pip install yfinance"
        ) from exc

    frame = yf.download(ticker, period=period, progress=False, auto_adjust=False, multi_level_index=False)
    if frame.empty:
        raise RuntimeError(f"No historical data returned for ticker {ticker!r}.")

    rows = []
    for timestamp, values in frame.iterrows():
        rows.append(
            PriceRow(
                date=timestamp.to_pydatetime(),
                open=float(values["Open"]),
                high=float(values["High"]),
                low=float(values["Low"]),
                close=float(values["Adj Close"] if "Adj Close" in values else values["Close"]),
                volume=float(values["Volume"]),
            )
        )
    rows.sort(key=lambda item: item.date)
    return rows


def percent_change(current: float, previous: float) -> float:
    if abs(previous) < 1e-12:
        return 0.0
    return (current - previous) / previous


def rolling_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def build_dataset(rows: list[PriceRow], horizon: int) -> tuple[list[datetime], list[list[float]], list[float]]:
    lookback = 10
    if len(rows) <= lookback + horizon:
        raise ValueError(f"Need more than {lookback + horizon} rows; received {len(rows)}.")

    dates: list[datetime] = []
    features: list[list[float]] = []
    targets: list[float] = []

    for index in range(lookback, len(rows) - horizon):
        row = rows[index]
        closes = [item.close for item in rows]
        volumes = [item.volume for item in rows]
        returns = [percent_change(closes[i], closes[i - 1]) for i in range(1, len(closes))]

        ma_5 = rolling_mean(closes[index - 4 : index + 1])
        ma_10 = rolling_mean(closes[index - 9 : index + 1])
        volatility_window = returns[max(0, index - 5) : index]
        volume_5 = rolling_mean(volumes[index - 4 : index + 1])

        feature_row = [
            row.close,
            row.open,
            row.high,
            row.low,
            row.volume,
            percent_change(row.close, row.open),
            (row.high - row.low) / row.close if abs(row.close) > 1e-12 else 0.0,
            percent_change(closes[index], closes[index - 1]),
            percent_change(closes[index - 1], closes[index - 2]),
            percent_change(closes[index - 2], closes[index - 3]),
            percent_change(closes[index - 4], closes[index - 5]),
            row.close / ma_5 if abs(ma_5) > 1e-12 else 1.0,
            row.close / ma_10 if abs(ma_10) > 1e-12 else 1.0,
            statistics.pstdev(volatility_window) if len(volatility_window) > 1 else 0.0,
            row.volume / volume_5 if abs(volume_5) > 1e-12 else 1.0,
            float(row.date.weekday()),
        ]

        dates.append(row.date)
        features.append(feature_row)
        targets.append(rows[index + horizon].close)

    return dates, features, targets


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(size)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("The regression system is singular; try a larger ridge alpha.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= pivot_value

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]

    return [augmented[row][-1] for row in range(size)]


def train_ridge_regression(
    features: list[list[float]], targets: list[float], horizon: int, alpha: float
) -> RidgeLinearRegression:
    scaler = StandardScaler.fit(features)
    scaled = scaler.transform(features)
    design = [[1.0] + row for row in scaled]
    columns = len(design[0])

    xtx = [[0.0 for _ in range(columns)] for _ in range(columns)]
    xty = [0.0 for _ in range(columns)]
    for row, target in zip(design, targets):
        for i in range(columns):
            xty[i] += row[i] * target
            for j in range(columns):
                xtx[i][j] += row[i] * row[j]

    for diagonal in range(1, columns):
        xtx[diagonal][diagonal] += alpha

    weights = solve_linear_system(xtx, xty)
    return RidgeLinearRegression(
        feature_names=FEATURE_NAMES,
        coefficients=weights[1:],
        intercept=weights[0],
        scaler=scaler,
        horizon=horizon,
    )


def train_test_split(
    dates: list[datetime], features: list[list[float]], targets: list[float], train_ratio: float
) -> tuple[list[datetime], list[list[float]], list[float], list[datetime], list[list[float]], list[float]]:
    split_index = max(1, min(len(features) - 1, int(len(features) * train_ratio)))
    return (
        dates[:split_index],
        features[:split_index],
        targets[:split_index],
        dates[split_index:],
        features[split_index:],
        targets[split_index:],
    )


def evaluate(actual: list[float], predicted: list[float], baseline: list[float]) -> dict[str, float]:
    errors = [prediction - target for prediction, target in zip(predicted, actual)]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    pct_errors = [abs(error / target) for error, target in zip(errors, actual) if abs(target) > 1e-12]

    directional_matches = 0
    for target, prediction, previous_close in zip(actual, predicted, baseline):
        actual_direction = target >= previous_close
        predicted_direction = prediction >= previous_close
        if actual_direction == predicted_direction:
            directional_matches += 1

    return {
        "mae": statistics.fmean(abs_errors),
        "rmse": math.sqrt(statistics.fmean(squared_errors)),
        "mape": statistics.fmean(pct_errors) * 100 if pct_errors else 0.0,
        "direction_accuracy": directional_matches / len(actual) * 100 if actual else 0.0,
    }


def write_predictions(path: Path, dates: list[datetime], actual: list[float], predicted: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "actual_close", "predicted_close", "error"])
        for date, target, forecast in zip(dates, actual, predicted):
            writer.writerow([date.date().isoformat(), f"{target:.4f}", f"{forecast:.4f}", f"{forecast - target:.4f}"])


def save_model(path: Path, model: RidgeLinearRegression) -> None:
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def load_rows(args: argparse.Namespace) -> list[PriceRow]:
    if args.csv:
        return read_price_csv(Path(args.csv))
    if args.ticker:
        return download_with_yfinance(args.ticker, args.period)
    raise ValueError("Provide either --csv path/to/prices.csv or --ticker SYMBOL.")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stock-price predictor from historical OHLCV data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Path to a CSV with Date, Open, High, Low, Close, Volume columns.")
    source.add_argument("--ticker", help="Ticker to download with yfinance, e.g. AAPL or MSFT.")
    parser.add_argument("--period", default="5y", help="yfinance period when using --ticker. Default: 5y.")
    parser.add_argument("--horizon", type=int, default=1, help="Predict this many trading days ahead. Default: 1.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Chronological train split ratio. Default: 0.8.")
    parser.add_argument("--alpha", type=positive_float, default=1.0, help="Ridge regularization strength. Default: 1.0.")
    parser.add_argument("--model-out", default="model.pkl", help="Where to save the trained model. Default: model.pkl.")
    parser.add_argument(
        "--predictions-out",
        default="predictions.csv",
        help="Where to write holdout predictions. Default: predictions.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1.")
    if not 0.1 <= args.train_ratio <= 0.95:
        raise ValueError("--train-ratio must be between 0.1 and 0.95.")

    rows = load_rows(args)
    dates, features, targets = build_dataset(rows, horizon=args.horizon)
    _, train_x, train_y, test_dates, test_x, test_y = train_test_split(
        dates, features, targets, args.train_ratio
    )

    model = train_ridge_regression(train_x, train_y, horizon=args.horizon, alpha=args.alpha)
    predictions = model.predict(test_x)
    baseline_closes = [row[0] for row in test_x]
    metrics = evaluate(test_y, predictions, baseline_closes)

    save_model(Path(args.model_out), model)
    write_predictions(Path(args.predictions_out), test_dates, test_y, predictions)

    latest_features = features[-1]
    latest_close = latest_features[0]
    next_prediction = model.predict_one(latest_features)

    print(f"Rows loaded: {len(rows)}")
    print(f"Training samples: {len(train_x)}")
    print(f"Holdout samples: {len(test_x)}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Direction accuracy: {metrics['direction_accuracy']:.2f}%")
    print(f"Latest close: {latest_close:.4f}")
    print(f"{args.horizon}-day-ahead predicted close: {next_prediction:.4f}")
    print(f"Saved model: {Path(args.model_out).resolve()}")
    print(f"Saved predictions: {Path(args.predictions_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
