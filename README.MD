# ChartLensAI — Image → Trading Scenarios (FastAPI + Gemini)

Upload a trading chart (or provide a direct image URL), pick timeframe/strategy, and get **AI-generated buy/sell scenarios** — entries, stops, T1–T3, R:R, and **max_time_wait_min** to avoid sideways chop.

> ⚠️ **Educational use only. Not financial advice.**  
> Scenarios are **AI-generated** and may be wrong. **Trade at your own risk.**  
> You are solely responsible for any financial losses or damages.

![UI](web/screenshot.png) <!-- optional: add a screenshot -->

---

## Features
- 🖼️ Upload image or use **direct** image URL (png/jpg/webp)
- ⏱️ Timeframes: `5m, 15m, 30m, 60m, 1D, 1W`
- 📚 Strategies: `auto, smc, ict, smt, sr, custom`
- 🧠 Gemini returns **strict JSON** (scenarios, RR, T1–T3, max wait, confluence, confidence)
- 🖊️ Overlay renderer (draws lines/zones on top of the uploaded chart)
- 🔐 No API key in the browser; key stays server-side
- 🌐 Static UI served at `/ui`

---

## Quickstart

### 1) Requirements
- Python 3.10+ (3.12/3.13 fine)
- A **Gemini API key**

### 2) Clone & install
```bash
git clone https://github.com/<you>/chartlensai.git
cd chartlensai
cp .env.example .env            # put your real GEMINI_API_KEY here (do not commit)
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
