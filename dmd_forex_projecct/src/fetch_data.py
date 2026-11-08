"""
دریافت داده‌های فارکس/بورس با yfinance
اجرای مستقل: python fetch_data.py --symbol EURUSD=X
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf


def fetch(symbol: str, start: str, end: str, interval: str, out_path: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"داده‌ای برای نماد {symbol} برنگشت. نماد یا بازه را بررسی کنید.")

    # نسخه‌های جدید yfinance ستون‌های MultiIndex برمی‌گردانند - صاف می‌کنیم
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = np.asarray(df[col]).reshape(-1)

    df.index.name = "datetime"
    df.to_csv(out_path)
    print(f"ذخیره شد: {out_path}  |  تعداد ردیف: {len(df)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="دریافت داده‌ی فارکس/بورس")
    parser.add_argument("--symbol", default="EURUSD=X")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--out", default="../data/raw_data.csv")
    args, _unknown = parser.parse_known_args()  # آرگومان‌های ناشناخته (مثل -f جوپیتر) را نادیده می‌گیرد

    fetch(args.symbol, args.start, args.end, args.interval, args.out)
