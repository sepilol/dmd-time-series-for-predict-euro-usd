@echo off
setlocal

REM ============================================================
REM اجرای خودکار روزانه‌ی اندیکاتور DMD برای EUR/USD
REM این نسخه با goto نوشته شده (نه if با پرانتز) تا حتی اگر مسیر پوشه
REM شامل پرانتز () یا فاصله باشد هم مشکلی پیش نیاید.
REM ============================================================

set "PROJECT_DIR=%~dp0"
set "SRC_DIR=%PROJECT_DIR%src"

echo ------------------------------------------------------------
echo پوشه پروژه: %PROJECT_DIR%
echo پوشه src   : %SRC_DIR%
echo ------------------------------------------------------------

if not exist "%SRC_DIR%\dmd_indicator.py" goto :NOFILE

cd /d "%SRC_DIR%"

where python >nul 2>nul
if errorlevel 1 goto :TRYANACONDA
goto :RUNPY

:TRYANACONDA
if not exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" goto :NOPYTHON
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base

:RUNPY
echo در حال اجرای اسکریپت پایتون...
python dmd_indicator.py
if errorlevel 1 goto :PYERROR

echo.
echo اجرا با موفقیت انجام شد.
pause
goto :EOF

:NOFILE
echo خطا: فایل dmd_indicator.py در %SRC_DIR% پیدا نشد.
echo لطفا مطمئن شوید ساختار پوشه‌ها دقیقا مطابق راهنماست.
pause
exit /b 1

:NOPYTHON
echo خطا: نه python در PATH است و نه Anaconda در مسیر پیش‌فرض پیدا شد.
echo لطفا مسیر نصب Anaconda یا Python خودتان را در این فایل اضافه کنید.
pause
exit /b 1

:PYERROR
echo.
echo خطا: اسکریپت پایتون با خطا متوقف شد. پیام بالا را بخوانید.
echo اگر پیام ModuleNotFoundError دیدید، این را اجرا کنید:
echo     pip install yfinance pandas numpy matplotlib
pause
exit /b 1
