from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from stock_price_predictor import (
    PriceRow,
    build_dataset,
    evaluate,
    read_price_csv,
    train_ridge_regression,
    train_test_split,
)


def make_rows(count: int = 80) -> list[PriceRow]:
    start = datetime(2024, 1, 1)
    rows = []
    for day in range(count):
        close = 100 + day * 0.7 + math.sin(day / 5)
        rows.append(
            PriceRow(
                date=start + timedelta(days=day),
                open=close - 0.25,
                high=close + 0.8,
                low=close - 0.9,
                close=close,
                volume=1_000_000 + day * 1000,
            )
        )
    return rows


class StockPredictorTests(unittest.TestCase):
    def test_build_dataset_and_train(self) -> None:
        dates, features, targets = build_dataset(make_rows(), horizon=1)
        self.assertEqual(len(dates), len(features))
        self.assertEqual(len(features), len(targets))

        _, train_x, train_y, _, test_x, test_y = train_test_split(dates, features, targets, 0.8)
        model = train_ridge_regression(train_x, train_y, horizon=1, alpha=1.0)
        predictions = model.predict(test_x)
        metrics = evaluate(test_y, predictions, [row[0] for row in test_x])

        self.assertEqual(len(predictions), len(test_y))
        self.assertLess(metrics["mae"], 3.0)

    def test_read_price_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text(
                "Date,Open,High,Low,Close,Volume\n"
                "2024-01-02,100,101,99,100.5,12345\n"
                "2024-01-01,99,100,98,99.5,12000\n",
                encoding="utf-8",
            )

            rows = read_price_csv(path)
            self.assertEqual([row.date.day for row in rows], [1, 2])
            self.assertEqual(rows[1].close, 100.5)


if __name__ == "__main__":
    unittest.main()
