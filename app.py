# app.py
from datetime import datetime, timezone
import os
import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import requests
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Body,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

# Gemini SDK
from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ---------- ENV ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PORT = int(os.getenv("PORT", "8080"))
RETRY_FOR_MISSING_SIDE = os.getenv("RETRY_FOR_MISSING_SIDE", "true").lower() in {"1", "true", "yes"}

if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- ENUMS ----------
class Timeframe(str, Enum):
    m5 = "5m"
    m15 = "15m"
    m30 = "30m"
    m60 = "60m"
    d1 = "1D"
    w1 = "1W"


class Strategy(str, Enum):
    auto = "auto"
    smc = "smc"
    ict = "ict"
    smt = "smt"
    sr = "sr"  # Support & Resistance
    custom = "custom"


# ---------- SCHEMA (Request) ----------
class AnalyzeJSON(BaseModel):
    image_url: str
    timeframe: Timeframe
    strategy: Strategy = Strategy.auto
    intention: Optional[str] = Field(default=None, max_length=300)
    question: Optional[str] = Field(default=None, max_length=500)


# ---------- SCHEMA (Response) ----------
class Instrument(BaseModel):
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    asset_class: Optional[str] = None
    confidence: Optional[float] = None


class ImageQuality(BaseModel):
    readable_axes: Optional[bool] = None
    occlusions: Optional[str] = None
    notes: Optional[str] = None


class ChartInfo(BaseModel):
    styleUsed: Optional[str] = None
    timeframe: Optional[str] = None
    instrument: Optional[Instrument] = None
    image_quality: Optional[ImageQuality] = None


class Level(BaseModel):
    kind: Optional[str] = None  # Support | Resistance | FVG | Gap | Trendline | Zone
    price_text: Optional[str] = None
    price_num: Optional[float] = None
    y_rel: Optional[float] = None
    confidence: Optional[float] = None


class Pattern(BaseModel):
    name: Optional[str] = None
    confidence: Optional[float] = None


class Trend(BaseModel):
    bias: Optional[str] = None
    confidence: Optional[float] = None


class Detected(BaseModel):
    levels: Optional[List[Level]] = None
    patterns: Optional[List[Pattern]] = None
    trend: Optional[Trend] = None


class Entry(BaseModel):
    type: Optional[str] = None  # breakout | pullback | limit | between | market
    price_text: Optional[str] = None
    zone_text: Optional[str] = None


class Stop(BaseModel):
    price_text: Optional[str] = None
    rule: Optional[str] = None  # e.g., below swing low


class Target(BaseModel):
    label: Optional[str] = None  # T1 | T2 | T3
    price_text: Optional[str] = None
    note: Optional[str] = None


class RMultiple(BaseModel):
    t1: Optional[float] = None
    t2: Optional[float] = None
    t3: Optional[float] = None


class DrawCommand(BaseModel):
    op: Optional[str] = None  # line | box | label
    y1_rel: Optional[float] = None
    y2_rel: Optional[float] = None
    note: Optional[str] = None


class Scenario(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    side: Optional[str] = None  # buy | sell
    timeframe: Optional[str] = None
    strategy_used: Optional[str] = None
    entry: Optional[Entry] = None
    stop: Optional[Stop] = None
    targets: Optional[List[Target]] = None
    r_multiple: Optional[RMultiple] = None
    max_time_wait_min: Optional[float] = None
    confidence: Optional[float] = None
    confluence: Optional[List[str]] = None
    explanation: Optional[str] = None
    invalidation_reason: Optional[str] = None
    draw_commands: Optional[List[DrawCommand]] = None


class ResponseMeta(BaseModel):
    received_at: Optional[str] = None
    request_id: Optional[str] = None


class RequestEcho(BaseModel):
    timeframe: Optional[str] = None
    strategy: Optional[str] = None
    intention: Optional[str] = None
    question: Optional[str] = None


class ScenarioResponse(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    meta: Optional[ResponseMeta] = None
    request: Optional[RequestEcho] = None
    chart: Optional[ChartInfo] = None
    detected: Optional[Detected] = None
    scenarios: Optional[List[Scenario]] = None
    suggestions: Optional[List[str]] = None
    disclaimer: Optional[str] = None


# ---------- NORMALIZER ----------
_KIND_MAP = {
    "support": "Support",
    "resistance": "Resistance",
    "demand_zone": "Demand Zone",
    "supply_zone": "Supply Zone",
    "current_price": "Current Price",
    "recent_swing_low": "Recent Swing Low",
    "recent_swing_high": "Recent Swing High",
    "fvg": "Fair Value Gap (FVG)",
}

_SIDE_MAP = {"long": "buy", "short": "sell"}


def _norm_side(v):
    if isinstance(v, str):
        return _SIDE_MAP.get(v.lower(), v.lower())
    return v


def _norm_strategy_used(v):
    if not isinstance(v, str):
        return v
    v = v.lower().replace("s&r", "sr").replace("/", "_").replace(" ", "")
    return v  # e.g., "smc_sr"


def _norm_entry_type(v):
    if isinstance(v, str):
        return v.lower()
    return v


def _norm_draw_op(v):
    if not isinstance(v, str):
        return v
    vlow = v.lower()
    if vlow in ("draw_line", "line"):
        return "line"
    if vlow in ("draw_box", "box", "rectangle"):
        return "box"
    if vlow in ("label", "text"):
        return "label"
    return v


def normalize_response(d: dict) -> dict:
    out = dict(d)  # shallow copy

    # detected.levels.kind -> nice names
    detected = out.get("detected") or {}
    levels = detected.get("levels") or []
    for lv in levels:
        k = lv.get("kind")
        if isinstance(k, str):
            lv["kind"] = _KIND_MAP.get(k.strip().lower(), k.title())
    detected["levels"] = levels
    out["detected"] = detected

    # scenarios cleanup
    scenarios = out.get("scenarios") or []
    for s in scenarios:
        s["side"] = _norm_side(s.get("side"))
        s["strategy_used"] = _norm_strategy_used(s.get("strategy_used"))

        entry = s.get("entry") or {}
        if "type" in entry:
            entry["type"] = _norm_entry_type(entry["type"])
        s["entry"] = entry

        # draw_commands op mapping
        cmds = s.get("draw_commands") or []
        for c in cmds:
            if "op" in c:
                c["op"] = _norm_draw_op(c["op"])
        s["draw_commands"] = cmds

        # r_multiple to floats if possible
        rm = s.get("r_multiple") or {}
        for k in ("t1", "t2", "t3"):
            if k in rm and rm[k] is not None:
                try:
                    rm[k] = float(rm[k])
                except Exception:
                    pass
        s["r_multiple"] = rm

    out["scenarios"] = scenarios

    # fill provider/model/meta defaults if missing (safe no-op if present)
    out.setdefault("provider", "gemini")
    out.setdefault("model", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    meta = out.get("meta") or {}
    meta.setdefault("received_at", iso_now())
    out["meta"] = meta

    return out


# ---------- UTILS ----------
def part_from_bytes(image_bytes: bytes, mime: str = "image/png"):
    return types.Part.from_bytes(data=image_bytes, mime_type=mime)


def part_from_url(url: str):
    # fetch the URL
    r = requests.get(url, timeout=25, headers={"User-Agent": "ChartLensAI/1.0 (+https://example.com)"})
    if not r.ok:
        raise HTTPException(status_code=400, detail=f"image_url fetch failed (HTTP {r.status_code})")

    # sniff MIME
    mime = (r.headers.get("content-type") or "").split(";")[0].strip().lower()

    # allow only real images
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if mime not in allowed:
        # last-ditch guess from file extension
        lower = url.lower()
        if lower.endswith(".png"):
            mime = "image/png"
        elif lower.endswith(".jpg"):
            mime = "image/jpeg"
        elif lower.endswith(".jpeg"):
            mime = "image/jpeg"
        elif lower.endswith(".webp"):
            mime = "image/webp"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"image_url must point directly to an image (png/jpg/webp). Got content-type: {mime or 'unknown'}",
            )

    return types.Part.from_bytes(data=r.content, mime_type=mime)


def build_prompt(
    *,
    timeframe: Union[Timeframe, str],
    strategy: Union[Strategy, str],
    intention: Optional[str],
    question: Optional[str],
) -> str:
    tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
    st = strategy.value if isinstance(strategy, Strategy) else str(strategy)
    intent = (intention or "").strip()
    q = (question or "").strip()
    return f"""
You are an image-based trading chart analyst.

Return ONLY raw JSON (no code fences) matching this shape. IMPORTANT RULES:
- ALWAYS output exactly TWO scenarios: one "buy" and one "sell".
- If one side looks low-probability, include it anyway with lower "confidence" and set "invalidation_reason".
- Keep explanations concise (<= 2 sentences). Fill T1–T3, r_multiple (t1..t3) if estimable; else leave null.

Context:
- timeframe: {tf}
- strategy: {st} (auto = infer a blend of SMC/ICT/SMT/S&R as appropriate)
- user_intention: {intent}
- user_question: {q}

Field hints:
- scenarios[].entry.type in [breakout, pullback, limit, between, market]
- max_time_wait_min = minutes to wait before skipping (avoid sideways).
- draw_commands[].op in [line, box, label], with y*_rel in 0..1.

Return raw JSON ONLY.
""".strip()


# ---------- BUY/SELL ENFORCER HELPERS ----------
def _side_set(scenarios: Optional[List[Scenario]]) -> set:
    out = set()
    if not scenarios:
        return out
    for s in scenarios:
        if not s or not s.side:
            continue
        out.add(str(s.side).lower())
    return out


def _pick_level(levels: List[Level], *candidates: str) -> Optional[Level]:
    # fuzzy match on kind (case/space insensitive)
    def norm(x): return (x or "").lower().replace(" ", "").strip()
    want = [norm(c) for c in candidates]
    for lv in levels or []:
        k = norm(lv.kind)
        for w in want:
            if w in k:
                return lv
    return None


def _synthesize_opposite(data: ScenarioResponse, missing: str, tf_hint: Optional[str]) -> Scenario:
    """
    Heuristic, low-confidence opposite scenario from detected levels.
    """
    det = data.detected or Detected()
    levels = det.levels or []
    trend = (det.trend.bias or "").lower() if det.trend else ""

    if missing == "sell":
        # Try short near swing high / resistance
        hi = _pick_level(levels, "recent_swing_high", "swinghigh", "resistance", "supply")
        lo = _pick_level(levels, "recent_swing_low", "swinglow", "support", "demand")
        entry_lv = hi or _pick_level(levels, "current_price")
        stop_lv = hi or _pick_level(levels, "resistance")
        t1_lv = lo or _pick_level(levels, "support", "demand")
        t2_lv = _pick_level(levels, "demand") or lo
        t3_lv = _pick_level(levels, "support") or lo

        sc = Scenario(
            id="sell_auto",
            name="Heuristic Rejection",
            side="sell",
            timeframe=tf_hint,
            strategy_used="auto",
            entry=Entry(type="market", price_text=(entry_lv.price_text if entry_lv else None)),
            stop=Stop(price_text=(stop_lv.price_text if stop_lv else None), rule="Above recent swing high / supply"),
            targets=[
                Target(label="T1", price_text=t1_lv.price_text if t1_lv else None, note="Nearest support/demand"),
                Target(label="T2", price_text=t2_lv.price_text if t2_lv else None, note="Deeper level"),
                Target(label="T3", price_text=t3_lv.price_text if t3_lv else None, note="Stretch target"),
            ],
            r_multiple=RMultiple(t1=None, t2=None, t3=None),
            max_time_wait_min=60.0,
            confidence=0.30,
            confluence=[c for c in ["Heuristic opposite-side scenario", f"Trend: {trend}" if trend else None] if c],
            explanation="Opposite-side idea auto-computed from levels; validate with price action.",
            invalidation_reason="Break and hold above recent swing high.",
            draw_commands=[
                c
                for c in [
                    DrawCommand(op="line", y1_rel=entry_lv.y_rel, y2_rel=entry_lv.y_rel, note="Entry")
                    if entry_lv and entry_lv.y_rel is not None
                    else None,
                    DrawCommand(op="line", y1_rel=stop_lv.y_rel, y2_rel=stop_lv.y_rel, note="Stop")
                    if stop_lv and stop_lv.y_rel is not None
                    else None,
                    DrawCommand(op="line", y1_rel=t1_lv.y_rel, y2_rel=t1_lv.y_rel, note="T1")
                    if t1_lv and t1_lv.y_rel is not None
                    else None,
                    DrawCommand(op="line", y1_rel=t2_lv.y_rel, y2_rel=t2_lv.y_rel, note="T2")
                    if t2_lv and t2_lv.y_rel is not None
                    else None,
                    DrawCommand(op="line", y1_rel=t3_lv.y_rel, y2_rel=t3_lv.y_rel, note="T3")
                    if t3_lv and t3_lv.y_rel is not None
                    else None,
                ]
                if c
            ],
        )
        return sc

    # missing == "buy"
    lo = _pick_level(levels, "recent_swing_low", "swinglow", "support", "demand")
    hi = _pick_level(levels, "recent_swing_high", "swinghigh", "resistance", "supply")
    entry_lv = lo or _pick_level(levels, "current_price")
    stop_lv = lo or _pick_level(levels, "support")
    t1_lv = hi or _pick_level(levels, "resistance", "supply")
    t2_lv = _pick_level(levels, "supply") or hi
    t3_lv = _pick_level(levels, "resistance") or hi

    sc = Scenario(
        id="buy_auto",
        name="Heuristic Pullback",
        side="buy",
        timeframe=tf_hint,
        strategy_used="auto",
        entry=Entry(type="market", price_text=(entry_lv.price_text if entry_lv else None)),
        stop=Stop(price_text=(stop_lv.price_text if stop_lv else None), rule="Below recent swing low / demand"),
        targets=[
            Target(label="T1", price_text=t1_lv.price_text if t1_lv else None, note="Nearest resistance/supply"),
            Target(label="T2", price_text=t2_lv.price_text if t2_lv else None, note="Deeper level"),
            Target(label="T3", price_text=t3_lv.price_text if t3_lv else None, note="Stretch target"),
        ],
        r_multiple=RMultiple(t1=None, t2=None, t3=None),
        max_time_wait_min=60.0,
        confidence=0.30,
        confluence=[c for c in ["Heuristic opposite-side scenario", f"Trend: {trend}" if trend else None] if c],
        explanation="Opposite-side idea auto-computed from levels; validate with price action.",
        invalidation_reason="Break and hold below recent swing low.",
        draw_commands=[
            c
            for c in [
                DrawCommand(op="line", y1_rel=entry_lv.y_rel, y2_rel=entry_lv.y_rel, note="Entry")
                if entry_lv and entry_lv.y_rel is not None
                else None,
                DrawCommand(op="line", y1_rel=stop_lv.y_rel, y2_rel=stop_lv.y_rel, note="Stop")
                if stop_lv and stop_lv.y_rel is not None
                else None,
                DrawCommand(op="line", y1_rel=t1_lv.y_rel, y2_rel=t1_lv.y_rel, note="T1")
                if t1_lv and t1_lv.y_rel is not None
                else None,
                DrawCommand(op="line", y1_rel=t2_lv.y_rel, y2_rel=t2_lv.y_rel, note="T2")
                if t2_lv and t2_lv.y_rel is not None
                else None,
                DrawCommand(op="line", y1_rel=t3_lv.y_rel, y2_rel=t3_lv.y_rel, note="T3")
                if t3_lv and t3_lv.y_rel is not None
                else None,
            ]
            if c
        ],
    )
    return sc


def ensure_two_scenarios(
    data: ScenarioResponse,
    original_parts: List[types.Part],
    timeframe,
    strategy,
    intention: Optional[str],
    question: Optional[str],
) -> ScenarioResponse:
    """
    If only one side is present, try a quick second Gemini call for the missing side.
    If that fails (or disabled), synthesize a low-confidence opposite scenario from levels.
    """
    sides = _side_set(data.scenarios)
    if {"buy", "sell"}.issubset(sides):
        return data

    missing = None
    if "buy" in sides and "sell" not in sides:
        missing = "sell"
    elif "sell" in sides and "buy" not in sides:
        missing = "buy"
    else:
        # zero or weird: try to get both
        missing = "both"

    img_part = original_parts[0]  # we always appended image first
    tf_hint = timeframe.value if hasattr(timeframe, "value") else str(timeframe)

    if RETRY_FOR_MISSING_SIDE:
        try:
            if missing in ("buy", "sell"):
                directive = f"""
Return ONLY raw JSON with "scenarios": an array containing exactly ONE scenario on the "{missing}" side.
Follow the same fields and brevity as before.
""".strip()
            else:
                directive = """
Return ONLY raw JSON with "scenarios": exactly TWO items: one "buy" and one "sell".
""".strip()

            retry_parts = [img_part, types.Part(text=directive)]
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=ScenarioResponse
            )
            resp2 = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=retry_parts)],
                config=cfg,
            )
            new_data = getattr(resp2, "parsed", None)
            if not new_data:
                new_data = ScenarioResponse.model_validate(json.loads(resp2.text))

            # Merge scenarios
            new_scenarios = new_data.scenarios or []
            merged = list(data.scenarios or [])
            have = _side_set(merged)
            for s in new_scenarios:
                sd = (s.side or "").lower()
                if missing == "both" or sd not in have:
                    merged.append(s)
                    have.add(sd)
            data.scenarios = merged
        except Exception:
            # ignore and fall back to synth
            pass

    # If still missing, synthesize
    sides = _side_set(data.scenarios)
    if "buy" not in sides:
        data.scenarios = (data.scenarios or []) + [_synthesize_opposite(data, "buy", tf_hint)]
    if "sell" not in sides:
        data.scenarios = (data.scenarios or []) + [_synthesize_opposite(data, "sell", tf_hint)]

    # Final tidy pass through normalizer+model validation
    as_dict = json.loads(data.model_dump_json(exclude_none=True))
    normalized = normalize_response(as_dict)
    return ScenarioResponse.model_validate(normalized)


# ---------- CORE ----------
def call_gemini(parts: list) -> ScenarioResponse:
    # parts must be a list of types.Part (image + text)
    content = types.Content(role="user", parts=parts)

    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ScenarioResponse,
    )

    try:
        resp = client.models.generate_content(model=MODEL, contents=[content], config=cfg)
    except genai_errors.APIError as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")

    parsed = getattr(resp, "parsed", None)
    if parsed:
        data = parsed
    else:
        text = resp.text
        try:
            obj = json.loads(text)
        except Exception as e:
            raise HTTPException(status_code=502, detail="Gemini returned non-JSON") from e
        data = ScenarioResponse.model_validate(obj)

    if data.provider is None:
        data.provider = "gemini"
    if data.model is None:
        data.model = MODEL
    if data.meta is None:
        data.meta = ResponseMeta(received_at=iso_now())

    return data


def analyze_image(
    image_bytes: Optional[bytes],
    image_url: Optional[str],
    timeframe,
    strategy,
    intention: Optional[str],
    question: Optional[str],
) -> ScenarioResponse:
    parts: List[types.Part] = []

    # image part
    if image_bytes:
        parts.append(part_from_bytes(image_bytes))
    elif image_url:
        parts.append(part_from_url(image_url))
    else:
        raise HTTPException(status_code=400, detail="Provide image file or image_url")

    # text prompt part
    prompt_text = build_prompt(
        timeframe=timeframe, strategy=strategy, intention=intention, question=question
    )
    parts.append(types.Part(text=prompt_text))

    data = call_gemini(parts)

    # guarantee we have both sides
    data = ensure_two_scenarios(
        data,
        original_parts=parts,
        timeframe=timeframe,
        strategy=strategy,
        intention=intention,
        question=question,
    )

    # echo request meta
    data.request = RequestEcho(
        timeframe=(timeframe.value if hasattr(timeframe, "value") else str(timeframe)),
        strategy=(strategy.value if hasattr(strategy, "value") else str(strategy)),
        intention=intention,
        question=question,
    )
    if not data.disclaimer:
        data.disclaimer = "Educational use only. Not financial advice."
    return data


# ---------- API ----------
app = FastAPI(title="ChartLensAI – Gemini (Python)", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod with ALLOWED_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /ui
web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")


@app.get("/health")
def health():
    return {"ok": True}


# Transform arbitrary JSON -> our schema (normalizer)
@app.post("/transform", response_model=ScenarioResponse)
def transform_endpoint(payload: dict = Body(...)):
    normalized = normalize_response(payload)
    return ScenarioResponse.model_validate(normalized)


@app.post("/analyze/json", response_model=ScenarioResponse)
def analyze_json(payload: AnalyzeJSON = Body(...)):
    return analyze_image(
        image_bytes=None,
        image_url=payload.image_url,
        timeframe=payload.timeframe,
        strategy=payload.strategy,
        intention=payload.intention,
        question=payload.question,
    )


@app.post("/analyze/upload", response_model=ScenarioResponse)
async def analyze_upload(
    image: UploadFile = File(..., description="Chart image"),
    timeframe: Timeframe = Form(...),
    strategy: Strategy = Form(Strategy.auto),
    intention: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image")
    return analyze_image(
        image_bytes=content,
        image_url=None,
        timeframe=timeframe,
        strategy=strategy,
        intention=intention,
        question=question,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
