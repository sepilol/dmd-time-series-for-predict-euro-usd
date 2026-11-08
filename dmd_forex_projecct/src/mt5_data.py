"""
دریافت داده‌ی EUR/USD مستقیم از ترمینال MetaTrader 5 که روی سیستم خودتان نصب است.

چرا MT5 به‌جای OANDA؟
OANDA رسماً به ایران خدمات نمی‌دهد (طبق شرایط استفاده‌ی خودش، ایران را در فهرست
مناطق محدود‌شده گذاشته). این روش به‌جایش از طریق همان ترمینال MT5 که با حساب دمو
یک بروکر پشتیبان ایران (مثل LiteFinance, Alpari, IC Markets, FBS, HFM, IFC Markets)
لاگین کرده‌اید متصل می‌شود - یعنی داده از طریق همان اتصال بروکر شما می‌آید،
نه یک اتصال مستقیم و مسدودشدنی از پایتون به یک سرور آمریکایی.

پیش‌نیاز:
1. یک حساب دمو رایگان نزد بروکری که به ایران خدمت می‌دهد بسازید (مثلا از
   iniran.forex برای LiteFinance، یا سایت رسمی Alpari/IC Markets/FBS).
2. پلتفرم MetaTrader 5 (حتما MT5، نه MT4) را از همان بروکر دانلود و نصب کنید.
3. با حساب دمو خود وارد شوید و مطمئن شوید ترمینال باز است و پایین-راستش
   نشانه‌ی اتصال سبز (نه قرمز) دارد.
4. pip install MetaTrader5   (این پکیج فقط روی ویندوز کار می‌کند)
5. اگر اسم نماد در Market Watch بروکرتان دقیقا "EURUSD" نیست (مثلا "EURUSD.m" یا
   "EURUSDm")، آن را در MT5_SYMBOL در dmd_indicator.py اصلاح کنید.
"""

import pandas as pd

TIMEFRAME_ATTR_MAP = {
    "1d": "TIMEFRAME_D1",
    "1h": "TIMEFRAME_H1",
    "30m": "TIMEFRAME_M30",
    "15m": "TIMEFRAME_M15",
    "5m": "TIMEFRAME_M5",
}


class MT5ConnectionError(RuntimeError):
    pass


def _get_mt5_module():
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise MT5ConnectionError(
            "پکیج MetaTrader5 نصب نیست یا سیستم شما ویندوز نیست. "
            "روی ویندوز اجرا کنید و دستور بزنید: pip install MetaTrader5"
        ) from e
    return mt5


def _ensure_connected(mt5):
    info = mt5.terminal_info()
    if info is None:
        if not mt5.initialize():
            raise MT5ConnectionError(
                f"اتصال به ترمینال MT5 ناموفق بود (last_error={mt5.last_error()}). "
                "مطمئن شوید برنامه‌ی MetaTrader 5 باز است و با حساب دمو خود وارد شده‌اید."
            )
        info = mt5.terminal_info()
    if info is not None and not info.connected:
        raise MT5ConnectionError(
            "ترمینال MT5 باز است ولی به سرور بروکر متصل نیست (نشانه‌ی قرمز پایین-راست). "
            "اتصال اینترنت یا لاگین حساب خود را چک کنید."
        )


def _rates_to_df(rates) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "tick_volume": "Volume",
    })
    df = df.set_index("time")
    df.index.name = "datetime"
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def fetch_candles_mt5(symbol: str = "EURUSD", interval: str = "1d", count: int = 700) -> pd.DataFrame:
    mt5 = _get_mt5_module()
    _ensure_connected(mt5)

    tf_attr = TIMEFRAME_ATTR_MAP.get(interval, interval)
    timeframe = getattr(mt5, tf_attr, None)
    if timeframe is None:
        raise MT5ConnectionError(f"بازه‌ی زمانی نامعتبر: {interval}")

    if not mt5.symbol_select(symbol, True):
        raise MT5ConnectionError(
            f"نماد '{symbol}' در Market Watch بروکر شما پیدا نشد. احتمالا اسمش با یک "
            f"پسوند فرق دارد (مثلا 'EURUSD.m') - آن را در MT5_SYMBOL اصلاح کنید."
        )

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        raise MT5ConnectionError(f"دریافت کندل ناموفق بود (last_error={mt5.last_error()})")

    return _rates_to_df(rates)


def fetch_live_data_mt5(symbol: str = "EURUSD", count: int = 700) -> pd.DataFrame:
    """معادل fetch_live_data (yfinance/OANDA قبلی) ولی از MT5 - کندل روزانه"""
    return fetch_candles_mt5(symbol, interval="1d", count=count)


if __name__ == "__main__":
    # تست دستی سریع (فقط روی ویندوز با ترمینال MT5 باز و لاگین‌شده کار می‌کند)
    try:
        df = fetch_live_data_mt5(count=10)
        print("۱۰ کندل روزانه‌ی آخر از MT5:")
        print(df)
    except MT5ConnectionError as e:
        print("خطای اتصال:", e)
