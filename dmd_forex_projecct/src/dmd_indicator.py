"""
اندیکاتور معاملاتی مبتنی بر DMD برای EUR/USD
------------------------------------------------
هر بار اجرا:
1. آخرین داده‌ی EUR/USD را می‌گیرد (اول از MT5 محلی، اگر نشد از yfinance)
2. یک مدل DMD روی بازده‌ی لگاریتمی fit می‌کند
3. بازده‌ی HORIZON روز آینده را پیش‌بینی می‌کند
4. سیگنال BUY/SELL/HOLD تولید می‌کند
5. نتیجه را به data/signals_log.csv اضافه می‌کند
6. یک نمودار کندلی PNG + یک اسکریپت TradingView ذخیره می‌کند

مسیرهای data و logs به‌صورت خودکار و نسبت به محل خود این فایل پیدا می‌شوند -
یعنی این فایل باید همیشه داخل پوشه‌ی src/ پروژه بماند.

اجرای دستی: python dmd_indicator.py
اجرای خودکار روزانه: با run_daily.bat + Windows Task Scheduler
"""

import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    def fa(text: str) -> str:
        """آماده‌سازی متن فارسی برای نمایش درست (راست‌به‌چپ) در matplotlib"""
        return get_display(arabic_reshaper.reshape(text))
except ImportError:
    def fa(text: str) -> str:
        return text

from dmd_edmd import hankel_embed, DMD

# ---------------- تنظیمات ----------------
SYMBOL = "EURUSD=X"       # نماد برای yfinance (fallback)
MT5_SYMBOL = "EURUSD"     # نماد برای MT5 - اگر بروکرتان پسوند دارد اینجا اصلاح کنید (مثلا "EURUSD.m")
LOOKBACK_PERIOD = "2y"
HORIZON = 1               # افق پیش‌بینی: ۱ روز آینده (کوتاه‌تر = خطای تجمعی کمتر = دقیق‌تر)
DELAY = 20                # پیش‌فرض اولیه - هر اجرا با auto-tuning ممکن است تغییر کند
RANK = 10                 # پیش‌فرض اولیه - هر اجرا با auto-tuning ممکن است تغییر کند
AUTO_TUNE_DELAYS = (10, 15, 20, 25, 30)
AUTO_TUNE_RANKS = (5, 8, 10, 12, 15)
AUTO_TUNE_VAL_SIZE = 60   # تعداد نمونه‌ی اعتبارسنجی برای انتخاب بهترین delay/rank
FIT_WINDOW = DELAY * 8
SIGNAL_THRESHOLD_STD = 0.5

# ایران دیگر ساعت تابستانی ندارد (از شهریور ۱۴۰۱) - آفست ثابت +۳:۳۰ همیشه درست است
IRAN_TZ = timezone(timedelta(hours=3, minutes=30))
DAILY_CUTOFF_HOUR = 1     # ساعت مرجع برای تشخیص "امروز" یا "فردا" (به وقت ایران)
DAILY_CUTOFF_MINUTE = 45  # یعنی مرجع = ۰۱:۴۵ بامداد - بر اساس داده‌ی واقعی اصلاح شد (پایین توضیح داده شده)

# مسیرها نسبت به محل این فایل (src/) محاسبه می‌شوند - نیازی به تغییر دستی نیست
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
LOG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "logs"))
LOG_PATH = os.path.join(DATA_DIR, "signals_log.csv")
CHART_PATH = os.path.join(DATA_DIR, "indicator_chart.png")
OHLC_LOG_PATH = os.path.join(DATA_DIR, "ohlc_forecast_log.csv")
PINE_SCRIPT_PATH = os.path.join(DATA_DIR, "dmd_forecast_overlay.pine")
ACTUAL_CANDLES_LOG_PATH = os.path.join(DATA_DIR, "actual_candles_log.csv")
PREDICTION_ACCURACY_LOG_PATH = os.path.join(DATA_DIR, "prediction_accuracy_log.csv")


WEEKDAY_FA = {
    0: "دوشنبه", 1: "سه‌شنبه", 2: "چهارشنبه", 3: "پنجشنبه",
    4: "جمعه", 5: "شنبه", 6: "یکشنبه",
}


def weekday_fa(date) -> str:
    return WEEKDAY_FA[pd.Timestamp(date).weekday()]


def describe_forecast_timing(forecast_first_date) -> str:
    """
    پیام واضح می‌سازد که بگوید پیش‌بینی روز اول برای «امروز» است یا «فردا» -
    بر اساس مقایسه‌ی تاریخ آن با تاریخ امروز به وقت ایران (ثابت +۳:۳۰، بدون تغییر فصلی)
    و ساعت مرجع ۰۱:۴۵ بامداد (بر اساس تحلیل داده‌ی واقعی: کندل روزانه‌ی فارکس با بسته‌شدن
    نیویورک ساعت ۵ عصر finalize می‌شود که تقریباً برابر ۰۰:۳۰-۰۱:۳۰ بامداد به وقت ایران است؛
    ۰۱:۴۵ کمی حاشیه‌ی امن اضافه می‌کند تا منبع داده هم به‌روزرسانی نهایی را انجام داده باشد).

    - اگر قبل از ۰۱:۴۵ بامداد اجرا شود: کندل امروز (طبق تقویم فارکس) هنوز کامل نشده، پس
      پیش‌بینی روی «امروز» است.
    - اگر بعد از ۰۱:۴۵ بامداد اجرا شود (یا با Task Scheduler که هر روز همین ساعت اجرا می‌شود):
      کندل امروز کامل شده، پس پیش‌بینی روی «فردا» است.
    """
    now_ir = datetime.now(IRAN_TZ)
    cutoff_today = now_ir.replace(hour=DAILY_CUTOFF_HOUR, minute=DAILY_CUTOFF_MINUTE,
                                    second=0, microsecond=0)
    before_cutoff = now_ir < cutoff_today
    predicted_date = pd.Timestamp(forecast_first_date).date()
    today_ir = now_ir.date()

    time_str = now_ir.strftime("%H:%M")
    if predicted_date == today_ir:
        timing_label = "امروز"
    elif predicted_date == today_ir + timedelta(days=1):
        timing_label = "فردا"
    else:
        timing_label = f"{predicted_date}"  # حالت نادر (مثلا آخر هفته که بازار تعطیل است)

    cutoff_note = "قبل" if before_cutoff else "بعد"
    return (
        f"ساعت اجرا: {time_str} به وقت ایران ({cutoff_note} از مرجع "
        f"{DAILY_CUTOFF_HOUR}:{DAILY_CUTOFF_MINUTE:02d}) -> "
        f"این پیش‌بینی برای «{timing_label}» ({predicted_date}) است."
    )


# =====================================================================
# دریافت داده: اول MT5 محلی (کار می‌کند برای ایران، از طریق بروکر پشتیبان)
# سپس yfinance به‌عنوان جایگزین (اگر MT5 در دسترس نبود)
# =====================================================================

def fetch_live_data(symbol=SYMBOL, period=LOOKBACK_PERIOD, interval="1d"):
    try:
        from mt5_data import fetch_live_data_mt5
        df = fetch_live_data_mt5(symbol=MT5_SYMBOL, count=700)
        print(f"[منبع داده: MetaTrader5 - نماد {MT5_SYMBOL}]")
        return df
    except Exception as e:
        print(f"[هشدار] دریافت از MT5 ناموفق بود ({e}) - تلاش برای Twelve Data...")

    try:
        from twelvedata_data import fetch_live_data_twelvedata
        df = fetch_live_data_twelvedata(symbol="EUR/USD", count=700)
        print("[منبع داده: Twelve Data]")
        return df
    except Exception as e:
        print(f"[هشدار] دریافت از Twelve Data ناموفق بود ({e}) - برگشت به yfinance...")

    raw = yf.download(symbol, period=period, interval=interval, progress=False)
    if raw.empty:
        raise RuntimeError(f"هیچ داده‌ای برای {symbol} دریافت نشد. اتصال اینترنت را چک کنید.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    raw.index.name = "datetime"
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        raw[col] = np.asarray(raw[col]).reshape(-1)
    print("[منبع داده: yfinance]")
    return raw


def _validate_one_step(series: np.ndarray, delay: int, rank: int, val_size: int, horizon: int = 1) -> float:
    """RMSE پیش‌بینی یک‌گام‌جلو روی val_size نمونه‌ی آخر (با teacher forcing) - برای auto-tuning"""
    if len(series) <= val_size + delay + 5:
        raise ValueError("داده برای این delay/val_size کافی نیست")
    fit_part = series[:-val_size]
    val_part = series[-val_size:]
    X = hankel_embed(fit_part, delay=delay)
    model = DMD(rank=rank).fit(X)

    preds, history = [], list(fit_part)
    x0 = X[:, -1].copy()
    for true_val in val_part:
        p = model.forecast(x0, n_steps=1)[0]
        preds.append(p)
        history.append(true_val)
        x0 = np.array(history[-delay:])
    return float(np.sqrt(np.mean((np.array(preds) - val_part) ** 2)))


def select_best_delay_rank(log_ret: np.ndarray, delays=AUTO_TUNE_DELAYS, ranks=AUTO_TUNE_RANKS,
                            val_size=AUTO_TUNE_VAL_SIZE):
    """
    برای دقت بیشتر: به‌جای delay/rank ثابت، هر اجرا روی داده‌ی اخیر یک اعتبارسنجی
    one-step-ahead کوچک انجام می‌دهد و پارامتری را انتخاب می‌کند که کمترین خطا را دارد.
    اگر داده برای auto-tuning کافی نبود، به مقادیر پیش‌فرض (DELAY, RANK) برمی‌گردد.
    """
    best = None
    for delay in delays:
        for rank in ranks:
            try:
                rmse = _validate_one_step(log_ret, delay, rank, val_size)
            except Exception:
                continue
            if best is None or rmse < best[2]:
                best = (delay, rank, rmse)
    if best is None:
        return DELAY, RANK
    return best[0], best[1]


def forecast_series(vals: np.ndarray, delay=DELAY, rank=RANK, horizon=HORIZON, fit_window=FIT_WINDOW):
    """پیش‌بینی horizon گام آینده‌ی یک سری قیمت (با DMD روی بازده لگاریتمی)"""
    log_ret = np.diff(np.log(vals))
    fit = log_ret[-fit_window:] if len(log_ret) > fit_window else log_ret
    X = hankel_embed(fit, delay=delay)
    model = DMD(rank=rank).fit(X)
    future_rets = model.forecast(X[:, -1], n_steps=horizon)
    return vals[-1] * np.exp(np.cumsum(future_rets))


def forecast_ohlc(raw: pd.DataFrame, delay=DELAY, rank=RANK):
    """
    پیش‌بینی کندل Open/High/Low/Close برای HORIZON روز آینده.
    - Close: با DMD روی سری Close
    - High/Low: هرکدام با DMD مستقل روی سری خودشان
    - Open روز بعد = Close همان لحظه (چون فارکس شکاف قیمتی مثل بورس سهام ندارد)
    - در پایان یک اصلاح سازگاری اعمال می‌شود: High >= max(Open, Close) و Low <= min(Open, Close)
    - تاریخ‌ها با pd.bdate_range محاسبه می‌شوند که فقط دوشنبه تا جمعه برمی‌گرداند
      (شنبه و یکشنبه که بازار فارکس بسته است، خودکار رد می‌شوند)
    """
    close_vals = np.asarray(raw["Close"]).reshape(-1)
    high_vals = np.asarray(raw["High"]).reshape(-1)
    low_vals = np.asarray(raw["Low"]).reshape(-1)

    close_fc = forecast_series(close_vals, delay=delay, rank=rank)
    high_fc = forecast_series(high_vals, delay=delay, rank=rank)
    low_fc = forecast_series(low_vals, delay=delay, rank=rank)

    open_fc = np.empty(HORIZON)
    open_fc[0] = close_vals[-1]
    for i in range(1, HORIZON):
        open_fc[i] = close_fc[i - 1]

    high_fc_adj = np.maximum.reduce([high_fc, open_fc, close_fc])
    low_fc_adj = np.minimum.reduce([low_fc, open_fc, close_fc])

    future_dates = pd.bdate_range(raw.index[-1], periods=HORIZON + 1)[1:]
    return pd.DataFrame({
        "date": future_dates,
        "weekday_fa": [weekday_fa(d) for d in future_dates],
        "open": open_fc,
        "high": high_fc_adj,
        "low": low_fc_adj,
        "close": close_fc,
    })


def generate_signal(raw: pd.DataFrame, delay=DELAY, rank=RANK):
    close = np.asarray(raw["Close"]).reshape(-1)
    log_ret = np.diff(np.log(close))

    fit_data = log_ret[-FIT_WINDOW:] if len(log_ret) > FIT_WINDOW else log_ret
    X = hankel_embed(fit_data, delay=delay)
    model = DMD(rank=rank).fit(X)

    future_rets = model.forecast(X[:, -1], n_steps=HORIZON)
    cum_ret = float(np.sum(future_rets))

    hist_std = float(np.std(log_ret[-250:])) if len(log_ret) >= 250 else float(np.std(log_ret))
    threshold = SIGNAL_THRESHOLD_STD * hist_std

    if cum_ret > threshold:
        signal = "BUY"
    elif cum_ret < -threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    last_price = float(close[-1])
    predicted_price = float(last_price * np.exp(cum_ret))
    confidence = float(np.clip(abs(cum_ret) / threshold, 0, 3)) if threshold > 0 else 0.0

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "delay_used": delay,
        "rank_used": rank,
        "last_price": round(last_price, 5),
        "predicted_price_h1": round(predicted_price, 5),
        "predicted_cum_return_pct": round(cum_ret * 100, 4),
        "signal": signal,
        "confidence": round(confidence, 2),
    }
    return result, close, future_rets


def append_log(row: dict, log_path=LOG_PATH):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(log_path):
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)


def append_log_if_new(row: dict, log_path: str, key_col: str) -> bool:
    """مثل append_log ولی اگر مقدار key_col قبلا در فایل ثبت شده باشد، ردیف تکراری اضافه نمی‌کند"""
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, dtype=str)
        if key_col in existing.columns and str(row[key_col]) in existing[key_col].values:
            return False
    append_log(row, log_path)
    return True


def log_actual_candle(raw: pd.DataFrame):
    """
    کندل واقعیِ آخرین روز کامل (که همین الان از منبع داده دریافت شده) را به
    data/actual_candles_log.csv اضافه می‌کند - هر روز فقط یک‌بار (بدون تکرار).
    این یک آرشیو تاریخی مستقل از داده‌ی خام است که هر روز به‌صورت خودکار رشد می‌کند.
    """
    last_date = raw.index[-1]
    last_row = raw.iloc[-1]
    row = {
        "date": last_date.strftime("%Y-%m-%d"),
        "weekday_fa": weekday_fa(last_date),
        "open": round(float(last_row["Open"]), 5),
        "high": round(float(last_row["High"]), 5),
        "low": round(float(last_row["Low"]), 5),
        "close": round(float(last_row["Close"]), 5),
    }
    is_new = append_log_if_new(row, ACTUAL_CANDLES_LOG_PATH, key_col="date")
    return row, is_new


def check_prediction_accuracy(actual_row: dict):
    """
    اگر برای همین تاریخ (actual_row['date']) قبلا یک پیش‌بینی در ohlc_forecast_log.csv
    ثبت شده باشد (یعنی روز/روزهای قبل پیش‌بینی شده بود)، خطای هرکدام از
    Open/High/Low/Close را حساب و در data/prediction_accuracy_log.csv ثبت می‌کند.
    اگر پیش‌بینی‌ای برای این تاریخ پیدا نشود (مثلا اولین اجرای پروژه)، None برمی‌گرداند.
    """
    if not os.path.exists(OHLC_LOG_PATH):
        return None
    forecasts = pd.read_csv(OHLC_LOG_PATH)
    matches = forecasts[forecasts["forecast_date"] == actual_row["date"]]
    if matches.empty:
        return None
    f = matches.iloc[-1]  # اگر چند بار پیش‌بینی شده، آخرین (نزدیک‌ترین به آن روز) را می‌گیریم

    def pct_err(pred, actual):
        return round(abs(pred - actual) / actual * 100, 4)

    predicted_dir = "up" if f["close"] >= f["open"] else "down"
    actual_dir = "up" if actual_row["close"] >= actual_row["open"] else "down"

    return {
        "date": actual_row["date"],
        "weekday_fa": actual_row["weekday_fa"],
        "predicted_open": f["open"], "actual_open": actual_row["open"],
        "error_open_pct": pct_err(f["open"], actual_row["open"]),
        "predicted_high": f["high"], "actual_high": actual_row["high"],
        "error_high_pct": pct_err(f["high"], actual_row["high"]),
        "predicted_low": f["low"], "actual_low": actual_row["low"],
        "error_low_pct": pct_err(f["low"], actual_row["low"]),
        "predicted_close": f["close"], "actual_close": actual_row["close"],
        "error_close_pct": pct_err(f["close"], actual_row["close"]),
        "direction_correct": predicted_dir == actual_dir,
    }


def write_pine_script(forecast_df: pd.DataFrame):
    """
    یک اندیکاتور Pine Script (TradingView) می‌سازد که کندل پیش‌بینی‌شده‌ی روز بعد را
    به‌صورت دستی (با پیست کردن مقادیر در ورودی‌های اندیکاتور) روی نمودار زنده نشان می‌دهد.
    TradingView اجازه‌ی push خودکار داده از بیرون به نمودار را نمی‌دهد؛ این روش قانونی و
    رایج جایگزین است: هر روز مقادیر تازه را از ohlc_forecast_log.csv کپی می‌کنید و در
    تنظیمات اندیکاتور (Settings) روی چارت پیست می‌کنید.
    """
    row1 = forecast_df.iloc[0]
    label1 = f"{row1['weekday_fa']} {row1['date'].strftime('%Y-%m-%d')}"

    pine_code = f'''//@version=5
indicator("DMD Forecast Overlay", overlay=true)

// این مقادیر را هر روز از فایل data/ohlc_forecast_log.csv کپی و اینجا (در Settings) پیست کنید
// روز+۱ = {label1}
o1 = input.float({row1['open']:.5f}, "روز+1 Open ({label1})")
h1 = input.float({row1['high']:.5f}, "روز+1 High ({label1})")
l1 = input.float({row1['low']:.5f}, "روز+1 Low ({label1})")
c1 = input.float({row1['close']:.5f}, "روز+1 Close ({label1})")

n_bars_ahead = input.int(1, "چند کندل جلوتر")

var line hi1 = na
var line lo1 = na
var box body1 = na
var label lbl1 = na

if barstate.islast
    line.delete(hi1)
    line.delete(lo1)
    box.delete(body1)
    label.delete(lbl1)

    x1 = bar_index + n_bars_ahead

    hi1 := line.new(x1, h1, x1, l1, color=color.new(color.blue, 0), width=1)
    body1 := box.new(x1 - 1, math.max(o1, c1), x1 + 1, math.min(o1, c1),
                       bgcolor=color.new(o1 <= c1 ? color.blue : color.orange, 60),
                       border_color=color.blue)
    lbl1 := label.new(x1, l1, "{label1}", style=label.style_label_up,
                        color=color.blue, textcolor=color.white, size=size.small)
'''
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PINE_SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(pine_code)
    return PINE_SCRIPT_PATH


def save_chart(close, future_rets, window=60):
    """چارت ساده‌ی خطی (نسخه‌ی سبک‌تر - در کنار نمودار کندلی کامل‌تر)"""
    last_price = close[-1]
    future_prices = last_price * np.exp(np.cumsum(future_rets))
    plt.figure(figsize=(10, 4))
    plt.plot(range(-window, 0), close[-window:], label="قیمت اخیر")
    plt.plot(range(0, len(future_prices)), future_prices, "o--", color="crimson",
              label=f"پیش‌بینی {HORIZON} روز آینده")
    plt.axvline(0, color="gray", linestyle=":")
    plt.legend()
    plt.title(f"{SYMBOL} - اندیکاتور DMD")
    plt.tight_layout()
    os.makedirs(DATA_DIR, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=120)
    plt.close()


def _draw_candle(ax, x, o, h, l, c, width=0.6, up_color="#26a69a", down_color="#ef5350",
                  edge_style="solid", alpha=1.0):
    """رسم دستی یک کندل (بدون نیاز به کتابخونه‌ی جانبی مثل mplfinance)"""
    color = up_color if c >= o else down_color
    ax.plot([x, x], [l, h], color=color, linewidth=1, alpha=alpha, linestyle=edge_style, zorder=2)
    body_bottom, body_height = min(o, c), abs(c - o) or 1e-6
    ax.add_patch(plt.Rectangle(
        (x - width / 2, body_bottom), width, body_height,
        facecolor=color, edgecolor=color, alpha=alpha,
        linestyle=edge_style, linewidth=1.2, zorder=3,
    ))


def save_candlestick_chart(raw: pd.DataFrame, forecast_df: pd.DataFrame, n_history=30):
    """نمودار کندلی: کندل‌های تاریخی واقعی + کندل‌های پیش‌بینی‌شده با استایل متفاوت"""
    hist = raw.iloc[-n_history:]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (_, row) in enumerate(hist.iterrows()):
        _draw_candle(ax, i, row["Open"], row["High"], row["Low"], row["Close"])

    n_hist = len(hist)
    for j, (_, row) in enumerate(forecast_df.iterrows()):
        _draw_candle(
            ax, n_hist + j, row["open"], row["high"], row["low"], row["close"],
            up_color="#2962ff", down_color="#ff6d00", edge_style="dashed", alpha=0.85,
        )
        label_text = fa(f"{row['weekday_fa']} {row['date'].strftime('%Y-%m-%d')}")
        y_offset = -28 - (j % 2) * 18
        ax.annotate(label_text, xy=(n_hist + j, row["low"]), xytext=(0, y_offset),
                    textcoords="offset points", ha="center", fontsize=8, color="#2962ff")

    ax.axvline(n_hist - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.text(n_hist - 0.5, ax.get_ylim()[1], fa(" پیش‌بینی ←"), va="top", fontsize=9, color="gray")

    hist_labels = [fa(f"{weekday_fa(d)} {d.strftime('%m-%d')}") for d in hist.index]
    forecast_labels = [fa(f"{r['weekday_fa']} {r['date'].strftime('%m-%d')}") for _, r in forecast_df.iterrows()]
    all_labels = hist_labels + forecast_labels
    step = max(1, len(all_labels) // 15)
    ax.set_xticks(range(0, len(all_labels), step))
    ax.set_xticklabels([all_labels[i] for i in range(0, len(all_labels), step)], rotation=45, fontsize=8)

    ax.set_title(fa(f"{SYMBOL} - کندل‌های واقعی + پیش‌بینی DMD ({HORIZON} روز آینده)"))
    ax.set_xlim(-1, n_hist + len(forecast_df) + 2)
    ax.margins(y=0.2)
    fig.tight_layout()

    os.makedirs(DATA_DIR, exist_ok=True)
    candle_path = os.path.join(DATA_DIR, "indicator_candlestick.png")
    fig.savefig(candle_path, dpi=130)
    plt.close(fig)
    return candle_path


def main():
    print("=" * 60)
    print(f"اجرای اندیکاتور DMD | {datetime.now().isoformat()}")
    print(f"BASE_DIR : {BASE_DIR}")
    print(f"DATA_DIR : {DATA_DIR}")
    print(f"LOG_DIR  : {LOG_DIR}")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        raw = fetch_live_data()
        print(f"داده دریافت شد. تعداد کندل: {len(raw)} | آخرین کندل: {raw.index[-1]}")

        # --- ثبت کندل واقعیِ تازه‌کامل‌شده در آرشیو + بررسی دقت پیش‌بینیِ روز(های) قبل ---
        actual_row, is_new_candle = log_actual_candle(raw)
        if is_new_candle:
            print(f"\nکندل واقعی جدید ثبت شد: {actual_row['date']} ({actual_row['weekday_fa']}) "
                  f"O={actual_row['open']} H={actual_row['high']} L={actual_row['low']} C={actual_row['close']}")
            accuracy = check_prediction_accuracy(actual_row)
            if accuracy:
                append_log_if_new(accuracy, PREDICTION_ACCURACY_LOG_PATH, key_col="date")
                dir_txt = "درست ✓" if accuracy["direction_correct"] else "غلط ✗"
                print(f"مقایسه با پیش‌بینیِ قبلی -> خطای Close: {accuracy['error_close_pct']}٪ "
                      f"| جهت حرکت: {dir_txt}")
            else:
                print("(پیش‌بینی قبلی برای این تاریخ در آرشیو پیدا نشد - احتمالا اولین اجراهای پروژه‌اند)")
        else:
            print(f"\nکندل {actual_row['date']} قبلا در آرشیو ثبت شده بود - تکراری اضافه نشد.")

        # --- Auto-tuning: انتخاب بهترین delay/rank بر اساس داده‌ی اخیر (برای دقت بیشتر) ---
        close_for_tuning = np.asarray(raw["Close"]).reshape(-1)
        log_ret_for_tuning = np.diff(np.log(close_for_tuning))
        best_delay, best_rank = select_best_delay_rank(log_ret_for_tuning)
        print(f"\nپارامترهای انتخاب‌شده (auto-tuned): delay={best_delay}, rank={best_rank}")

        result, close, future_rets = generate_signal(raw, delay=best_delay, rank=best_rank)
        append_log(result)
        save_chart(close, future_rets)

        forecast_df = forecast_ohlc(raw, delay=best_delay, rank=best_rank)
        timing_msg = describe_forecast_timing(forecast_df.iloc[0]["date"])
        print(f"\n{timing_msg}")
        today_ir = datetime.now(IRAN_TZ).date()
        timing_label = "today" if pd.Timestamp(forecast_df.iloc[0]["date"]).date() == today_ir else "tomorrow"

        for _, r in forecast_df.iterrows():
            ohlc_row = {
                "run_timestamp": result["timestamp"],
                "forecast_date": r["date"].strftime("%Y-%m-%d"),
                "weekday_fa": r["weekday_fa"],
                "timing": timing_label,
                "open": round(r["open"], 5),
                "high": round(r["high"], 5),
                "low": round(r["low"], 5),
                "close": round(r["close"], 5),
            }
            append_log(ohlc_row, log_path=OHLC_LOG_PATH)

        candle_path = save_candlestick_chart(raw, forecast_df)
        pine_path = write_pine_script(forecast_df)

        print("\nنتیجه سیگنال:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("\nپیش‌بینی کندل‌های آینده (OHLC):")
        print(forecast_df[["date", "weekday_fa", "open", "high", "low", "close"]].to_string(index=False))

        print(f"\nلاگ سیگنال: {LOG_PATH}")
        print(f"لاگ OHLC پیش‌بینی: {OHLC_LOG_PATH}")
        print(f"آرشیو کندل‌های واقعی: {ACTUAL_CANDLES_LOG_PATH}")
        print(f"لاگ دقت پیش‌بینی: {PREDICTION_ACCURACY_LOG_PATH}")
        print(f"نمودار خطی: {CHART_PATH}")
        print(f"نمودار کندلی: {candle_path}")
        print(f"اسکریپت TradingView: {pine_path}")
        print("وضعیت: موفق")
    except Exception as e:
        print(f"خطا در اجرا: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    finally:
        print("=" * 60)


if __name__ == "__main__":
    main()
