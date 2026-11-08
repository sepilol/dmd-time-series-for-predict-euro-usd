"""
تشخیص دقیق اینکه کندل روزانه‌ی yfinance برای EURUSD بر چه اساس زمانی بسته می‌شود.

منطق: داده‌ی روزانه و ساعتی را همزمان می‌گیریم. Close هر کندل روزانه باید دقیقاً
برابر Close یکی از کندل‌های ساعتی باشد - همان ساعت، رول‌اوور روزانه است.

اجرا: python check_timezone.py
"""

import pandas as pd
import numpy as np
import yfinance as yf

SYMBOL = "EURUSD=X"

print("در حال دریافت داده‌ی روزانه...")
daily = yf.download(SYMBOL, period="10d", interval="1d", progress=False)
if isinstance(daily.columns, pd.MultiIndex):
    daily.columns = daily.columns.get_level_values(0)

print("در حال دریافت داده‌ی ساعتی...")
hourly = yf.download(SYMBOL, period="10d", interval="1h", progress=False)
if isinstance(hourly.columns, pd.MultiIndex):
    hourly.columns = hourly.columns.get_level_values(0)

print("\n--- اطلاعات منطقه‌ی زمانی ایندکس ---")
print("Timezone داده‌ی روزانه:", daily.index.tz)
print("Timezone داده‌ی ساعتی :", hourly.index.tz)

print("\n--- چند نمونه از کندل روزانه (خام، همان‌طور که yfinance می‌دهد) ---")
print(daily[["Open", "High", "Low", "Close"]].tail(5))

print("\n--- تطبیق Close روزانه با نزدیک‌ترین کندل ساعتی ---")
hourly_close = hourly["Close"]
for day_ts, day_row in daily.tail(5).iterrows():
    target_close = float(day_row["Close"])
    # نزدیک‌ترین کندل ساعتی از نظر مقدار Close را پیدا کن (در بازه‌ی ± 1 روز اطراف)
    window = hourly_close[(hourly_close.index >= day_ts - pd.Timedelta(hours=6)) &
                            (hourly_close.index <= day_ts + pd.Timedelta(hours=30))]
    if window.empty:
        continue
    diffs = (window - target_close).abs()
    best_ts = diffs.idxmin()
    print(f"روز {day_ts.date()} | Close روزانه={target_close:.5f} "
          f"| نزدیک‌ترین ساعت تطبیق‌یافته (UTC-index خام)= {best_ts} "
          f"| اختلاف={diffs.min():.6f}")

print("""
--- راهنمای تفسیر نتیجه ---
اگر ستون Timezone بالا خالی/None بود، یعنی ایندکس بدون منطقه‌ی زمانی (naive) است
و باید بر اساس ساعت تطبیق‌یافته در جدول آخر حدس زد. اگر آن ساعت تطبیق‌یافته نزدیک به
17:00 یا 21:00 یا 00:00 (بسته به DST) بود، رول‌اوور نیویورک را تأیید می‌کند.
اگر Timezone چیزی مثل 'America/New_York' یا 'UTC' نشان داد، مستقیم مشخص است.
""")
