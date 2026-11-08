"""
دریافت داده‌ی EUR/USD از Twelve Data API - بدون نیاز به MetaTrader یا هیچ برنامه‌ی دیگری،
فقط یک درخواست HTTP ساده.

چرا Twelve Data؟
- کاملاً رایگان در سطح پایه (۸۰۰ درخواست در روز، ۸ درخواست در دقیقه - برای این پروژه که
  روزی یک‌بار اجرا می‌شود کاملاً کافی است)
- کندل واقعی OHLC می‌دهد (نه فقط یک نرخ لحظه‌ای مثل بعضی API های رایگان دیگر)
- شرکتش در سنگاپور ثبت شده (نه آمریکا)، و برخلاف OANDA صریحاً ایران را در فهرست
  محدودیت‌هایش نگذاشته - ولی چون نمی‌شود صد در صد از بیرون تضمین کرد، این ماژول کاملاً
  اختیاری است و اگر کار نکند خودکار به yfinance برمی‌گردیم.

پیش‌نیاز:
1. یک حساب رایگان در https://twelvedata.com/ بسازید (نیاز به کارت بانکی ندارد)
2. از داشبورد، API Key خودتان را کپی کنید
3. آن را در فایل config.ini (کنار همین فایل) بگذارید
4. pip install requests
"""

import configparser
import os

import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")

INTERVAL_MAP = {
    "1d": "1day",
    "1h": "1h",
    "30m": "30min",
    "15m": "15min",
    "5m": "5min",
}


class TwelveDataConfigError(RuntimeError):
    pass


def _load_api_key() -> str:
    if not os.path.exists(CONFIG_PATH):
        raise TwelveDataConfigError(
            f"فایل config.ini پیدا نشد در {CONFIG_PATH}. طبق README یک حساب رایگان "
            "Twelve Data بسازید و کلید API را در این فایل بگذارید."
        )
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH, encoding="utf-8")
    api_key = cfg.get("twelvedata", "api_key", fallback="").strip()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise TwelveDataConfigError(
            "کلید API در config.ini (بخش [twelvedata]) تنظیم نشده."
        )
    return api_key


def to_twelvedata_symbol(symbol: str) -> str:
    """تبدیل فرمت نماد از سبک yfinance ('EURUSD=X') به سبک Twelve Data ('EUR/USD')"""
    s = symbol.upper().replace("=X", "")
    if "/" in s:
        return s
    if len(s) == 6:
        return f"{s[:3]}/{s[3:]}"
    return s


def _parse_time_series(raw_json: dict) -> pd.DataFrame:
    """
    تبدیل پاسخ JSON خام Twelve Data به DataFrame استاندارد پروژه.
    ساختار طبق مستندات رسمی:
    {"values": [{"datetime": "...", "open": "..", "high": "..", "low": "..",
                 "close": "..", "volume": ".."}], "status": "ok"}
    """
    if raw_json.get("status") == "error":
        raise RuntimeError(f"خطای Twelve Data: {raw_json.get('message', raw_json)}")

    values = raw_json.get("values", [])
    if not values:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    rows, idx = [], []
    for v in values:
        idx.append(pd.Timestamp(v["datetime"]))
        rows.append({
            "Open": float(v["open"]),
            "High": float(v["high"]),
            "Low": float(v["low"]),
            "Close": float(v["close"]),
            "Volume": float(v.get("volume", 0) or 0),
        })
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="datetime"))
    return df.sort_index()  # Twelve Data معمولا جدیدترین را اول می‌دهد - مرتب می‌کنیم


def fetch_candles_twelvedata(symbol: str = "EUR/USD", interval: str = "1d",
                              outputsize: int = 700) -> pd.DataFrame:
    api_key = _load_api_key()
    td_symbol = to_twelvedata_symbol(symbol)
    td_interval = INTERVAL_MAP.get(interval, interval)

    params = {
        "symbol": td_symbol,
        "interval": td_interval,
        "outputsize": min(outputsize, 5000),
        "apikey": api_key,
        "format": "JSON",
    }
    resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"خطای HTTP Twelve Data ({resp.status_code}): {resp.text[:300]}")

    return _parse_time_series(resp.json())


def fetch_live_data_twelvedata(symbol: str = "EUR/USD", count: int = 700) -> pd.DataFrame:
    """معادل fetch_live_data قبلی ولی با Twelve Data - کندل روزانه"""
    df = fetch_candles_twelvedata(symbol, interval="1d", outputsize=count)
    if df.empty:
        raise RuntimeError(f"هیچ داده‌ای از Twelve Data برای {symbol} دریافت نشد.")
    return df


if __name__ == "__main__":
    try:
        daily = fetch_live_data_twelvedata(count=10)
        print("۱۰ کندل روزانه‌ی آخر از Twelve Data:")
        print(daily)
    except TwelveDataConfigError as e:
        print("خطای تنظیمات:", e)
    except Exception as e:
        print("خطا:", e)
