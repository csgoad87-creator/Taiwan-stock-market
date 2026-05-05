#!/usr/bin/env python3
"""
台股 MACD 訊號掃描器(GitHub Actions 用)
=====================================
- 從 FinMind 抓 TWSE 上市股清單(過濾掉上櫃 tpex、興櫃、ETF)
- 對每檔計算 MACD(12,26,9)
- 偵測訊號:
    * imminent_above   零軸上 · 即將金叉(多頭加速)
    * imminent_below   零軸下 · 即將金叉(底部反轉)
    * crossed_above    零軸上 · 已金叉(進行中)
    * crossed_below    零軸下 · 已金叉(進行中)
- 輸出 signals.json 供前端讀取
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
LOOKBACK_DAYS = 240          # 約 8 個月歷史(夠算 26EMA + Signal)
MIN_BARS = 70                # 至少要有 70 根 K 棒
RECENT_CROSS_DAYS = 5        # 近 5 日內的金叉算「已金叉」
IMMINENT_DAYS_MAX = 3.5      # 線性外推 ≤ 3.5 日視為「即將金叉」
TZ_TPE = timezone(timedelta(hours=8))

# 重點掃描的產業(避免掃太多冷門股)
PRIORITY_INDUSTRIES = {
    "半導體業", "電腦及週邊設備業", "電子零組件業", "光電業",
    "通信網路業", "電子通路業", "資訊服務業", "其他電子業",
    "金融保險業", "電機機械", "鋼鐵工業", "運輸物流業",
    "汽車工業", "貿易百貨業", "生技醫療業", "塑膠工業",
    "食品工業", "化學工業", "玻璃陶瓷",
}


# ---------------- 資料抓取 ----------------
def fetch_stock_list():
    """抓 TaiwanStockInfo,過濾出純 TWSE 上市的 4 碼數字股票。"""
    print("Fetching stock list...")
    r = requests.get(f"{FINMIND_API}?dataset=TaiwanStockInfo", timeout=60)
    r.raise_for_status()
    j = r.json()
    if int(j.get("status", 0)) != 200:
        raise RuntimeError(f"TaiwanStockInfo failed: {j.get('msg')}")

    seen = set()
    cleaned = []
    for s in j["data"]:
        code = (s.get("stock_id") or "").strip()
        # 嚴格過濾條件:
        #   1. 必須是 4 碼純數字(排除權證、特別股、ETF、債券)
        #   2. type 必須是 'twse'(上市,排除 tpex 上櫃 / emerging 興櫃)
        if not (code.isdigit() and len(code) == 4):
            continue
        # FinMind 的 TaiwanStockInfo 沒有 type 欄位,要從 industry 判斷
        # ETF 的 industry 是 ETF / ETN,排除掉
        industry = (s.get("industry_category") or "").strip()
        if industry in {"ETF", "ETN", "受益證券", ""}:
            continue
        if code in seen:
            continue
        seen.add(code)
        cleaned.append({
            "code": code,
            "name": (s.get("stock_name") or "").strip(),
            "industry": industry,
        })
    print(f"  → got {len(cleaned)} candidates")
    return cleaned


def fetch_prices(code, start_date, retries=2):
    """抓單一股票的日線。"""
    url = f"{FINMIND_API}?dataset=TaiwanStockPrice&data_id={code}&start_date={start_date}"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=30)
            if not r.ok:
                if attempt < retries:
                    time.sleep(1.5)
                    continue
                return None
            j = r.json()
            if int(j.get("status", 0)) != 200 or not j.get("data"):
                return None
            rows = sorted(j["data"], key=lambda x: x["date"])
            closes, vols, dates = [], [], []
            for row in rows:
                c = row.get("close")
                if c is None or float(c) <= 0:
                    continue
                closes.append(float(c))
                vols.append(float(row.get("Trading_Volume") or 0))
                dates.append(row["date"])
            if len(closes) < MIN_BARS:
                return None
            return np.array(closes), np.array(vols), dates
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5)
                continue
            print(f"  {code} fetch error: {e}", file=sys.stderr)
            return None


# ---------------- 指標計算 ----------------
def ema(arr, span):
    k = 2 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def sma(arr, n):
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = np.mean(arr[i - n + 1:i + 1])
    return out


def compute_macd(closes):
    fast = ema(closes, 12)
    slow = ema(closes, 26)
    macd = fast - slow
    signal = sma(macd, 9)
    hist = macd - signal
    return macd, signal, hist


# ---------------- 訊號判定 ----------------
def detect_signal(macd, signal, hist, vols):
    """
    判定訊號類型,回傳 dict 或 None。
    keys: type ('crossed' | 'imminent'), zone ('above' | 'below'), days, score, macd, signal, hist
    """
    if len(macd) < 30 or np.isnan(signal[-1]):
        return None

    # ① 檢查近 RECENT_CROSS_DAYS 日是否已金叉
    crossed_idx = None
    for offset in range(RECENT_CROSS_DAYS + 1):
        i = -1 - offset
        if -i > len(macd):
            break
        if i - 1 < -len(macd):
            break
        if (macd[i - 1] < signal[i - 1]) and (macd[i] >= signal[i]):
            crossed_idx = i
            break

    if crossed_idx is not None:
        days_ago = -crossed_idx - 1
        m_at = float(macd[crossed_idx])
        s_at = float(signal[crossed_idx])
        zone = "above" if (m_at > 0 and s_at > 0) else "below"
        # 評分(0~10)
        score = 5
        if zone == "below":
            score += 2
        if not np.isnan(hist[-2]) and hist[-1] > hist[-2]:
            score += 1
        if len(vols) >= 21:
            avg20 = np.mean(vols[-21:-1])
            if avg20 > 0 and vols[-1] > avg20 * 1.5:
                score += 1
        if hist[-1] > 0:
            score += 1
        return {
            "type": "crossed",
            "zone": zone,
            "days": days_ago,
            "score": min(score, 10),
            "macd": float(macd[-1]),
            "signal": float(signal[-1]),
            "hist": float(hist[-1]),
        }

    # ② 沒已金叉,看是否「即將」金叉
    # 條件:目前 MACD 在 Signal 之下,但差距正在收斂
    if hist[-1] >= 0:
        return None
    diff_now = float(signal[-1] - macd[-1])
    if diff_now <= 0:
        return None

    # 用近 5 日的 (signal - macd) 線性回歸估計交叉時間
    if len(macd) < 5:
        return None
    recent_diff = signal[-5:] - macd[-5:]
    if np.any(np.isnan(recent_diff)):
        return None
    if not np.all(recent_diff > 0):
        return None
    # 必須在縮小:最後一根要比前 3 根小
    if recent_diff[-1] >= recent_diff[-3]:
        return None
    slope = np.polyfit(range(5), recent_diff, 1)[0]
    if slope >= 0:
        return None
    days_to_cross = float(-recent_diff[-1] / slope)
    if not (0 < days_to_cross <= IMMINENT_DAYS_MAX):
        return None

    m_now = float(macd[-1])
    s_now = float(signal[-1])
    zone = "above" if (m_now > 0 and s_now > 0) else "below"
    score = 3
    if zone == "below":
        score += 1
    if not np.isnan(hist[-2]) and hist[-1] > hist[-2]:
        score += 1
    if len(vols) >= 21:
        avg20 = np.mean(vols[-21:-1])
        if avg20 > 0 and vols[-1] > avg20 * 1.3:
            score += 1
    return {
        "type": "imminent",
        "zone": zone,
        "days": round(days_to_cross, 1),
        "score": min(score, 10),
        "macd": m_now,
        "signal": s_now,
        "hist": float(hist[-1]),
    }


# ---------------- 主流程 ----------------
def main():
    today = datetime.now(TZ_TPE)
    start_date = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    stocks = fetch_stock_list()
    # 過濾優先產業(不是強制,但會減少掃描量)
    priority = [s for s in stocks if s["industry"] in PRIORITY_INDUSTRIES]
    others = [s for s in stocks if s["industry"] not in PRIORITY_INDUSTRIES]
    # 先掃優先產業,再掃剩下的(總數限制 320 檔避免 Action 跑太久)
    universe = (priority + others)[:320]
    print(f"Will scan {len(universe)} stocks ({len(priority)} priority + {len(universe) - len(priority)} others)")

    results = {
        "imminent_above": [],
        "imminent_below": [],
        "crossed_above": [],
        "crossed_below": [],
    }
    fetched = 0
    failed = 0

    for i, s in enumerate(universe, 1):
        if i % 25 == 0:
            print(f"  Progress {i}/{len(universe)} | imminent={len(results['imminent_above']) + len(results['imminent_below'])} | crossed={len(results['crossed_above']) + len(results['crossed_below'])}")
        try:
            data = fetch_prices(s["code"], start_date)
            if data is None:
                failed += 1
                continue
            closes, vols, dates = data
            macd, signal, hist = compute_macd(closes)
            sig = detect_signal(macd, signal, hist, vols)
            fetched += 1
            if sig is None:
                continue
            entry = {
                **s,
                **sig,
                "close": float(closes[-1]),
                "as_of": dates[-1],
            }
            key = f"{sig['type']}_{sig['zone']}"
            results[key].append(entry)
            time.sleep(0.35)  # 對 FinMind 友善,避免 rate limit
        except Exception as e:
            print(f"  {s['code']} unhandled error: {e}", file=sys.stderr)
            failed += 1
            continue

    # 排序:即將金叉按預估天數遞增、已金叉按分數遞減
    results["imminent_above"].sort(key=lambda x: (x["days"], -x["score"]))
    results["imminent_below"].sort(key=lambda x: (x["days"], -x["score"]))
    results["crossed_above"].sort(key=lambda x: (-x["score"], x["days"]))
    results["crossed_below"].sort(key=lambda x: (-x["score"], x["days"]))

    output = {
        "scanned_at": today.isoformat(),
        "stock_universe": len(universe),
        "fetched": fetched,
        "failed": failed,
        "summary": {k: len(v) for k, v in results.items()},
        "results": results,
        "config": {
            "macd": [12, 26, 9],
            "lookback_days": LOOKBACK_DAYS,
            "imminent_threshold_days": IMMINENT_DAYS_MAX,
            "recent_cross_window_days": RECENT_CROSS_DAYS,
            "filter": "TWSE listed only (上市), 4-digit numeric codes, ETF/ETN excluded",
        },
    }

    out_path = Path("signals.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(v) for v in results.values())
    print(f"\n✓ Wrote {out_path}")
    print(f"  Total signals: {total}")
    for k, v in output["summary"].items():
        print(f"    {k}: {v}")
    print(f"  Fetched ok: {fetched}  failed: {failed}")


if __name__ == "__main__":
    main()
