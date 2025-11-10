#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scrape ForexFactory economic calendar month-by-month (server-rendered HTML)
and save 2023–2025 data (configurable) to CSV (full + per-year).
- Single-line 0–100% progress bar
- No pyarrow (CSV output)
- Robust parsing against common FF HTML patterns

Usage (PowerShell):
  pip install requests beautifulsoup4 pandas tqdm python-dateutil
  python tools\scrape_forexfactory_2023_2025.py --out_dir "data\ff_2023_2025" --start_year 2023 --end_year 2025 --sleep 1.0
"""

import argparse
import os
import re
import time
import sys
from datetime import datetime, date
from dateutil import parser as dtparse
from calendar import month_abbr
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://www.forexfactory.com/calendar"

HEADERS = {
    # polite UA
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}

# Try to force UTC display (reduces time ambiguity). ForexFactory respects cookies for TZ.
COOKIES = {
    # 0 is UTC on FF, commonly used. If FF ignores, we still capture date & time text as shown.
    "ffTimeZone": "0"
}

# Map common impact class names/text to normalized impact string and level
IMPACT_LEVELS = {
    "High": 3,      # red
    "Medium": 2,    # orange
    "Low": 1,       # yellow
    "Holiday": 0,   # gray
    "Non-Economic": 0,
}

def month_url(year: int, month: int) -> str:
    """Construct a monthly calendar URL: e.g., ?month=Jan.2024"""
    mon_str = month_abbr[month]  # Jan, Feb, ...
    return f"{BASE_URL}?month={mon_str}.{year}"

def _text(x) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())

def parse_impact(cell: BeautifulSoup) -> (str, int):
    """
    Impact is often indicated by:
      - an icon with title/alt (e.g., 'High Impact Expected'), or
      - text label in the row
    """
    txt = _text(cell.get_text(" "))
    title = cell.get("title") or ""
    alt = ""
    img = cell.find("img")
    if img and img.has_attr("alt"):
        alt = img["alt"]

    raw = _text(title) or _text(alt) or txt

    # Normalize to High/Medium/Low/Holiday/Non-Economic
    norm = None
    lvl = 0
    # common keywords
    if re.search(r"\bhigh\b", raw, re.I):
        norm, lvl = "High Impact Expected", 3
    elif re.search(r"\bmedium\b", raw, re.I):
        norm, lvl = "Medium Impact Expected", 2
    elif re.search(r"\blow\b", raw, re.I):
        norm, lvl = "Low Impact Expected", 1
    elif re.search(r"holiday|bank holiday|non-?economic", raw, re.I):
        norm, lvl = "Non-Economic", 0

    return (norm or raw or ""), lvl

def parse_month(html: str, year: int, month: int) -> List[Dict]:
    """
    Parse a ForexFactory month page into structured rows.
    Tries to be resilient to layout variants.
    """
    soup = BeautifulSoup(html, "html.parser")

    # FF calendar rows often have classes like:
    # 'calendar__row--release' or general rows with cells:
    # columns: date, time, currency, impact, event, actual, forecast, previous
    # We'll look for the main table first.
    rows = []

    # Strategy:
    # 1) Identify the calendar table region.
    # 2) Iterate visible rows that contain event cells.
    # 3) Carry-forward date when row omits it (typical grouped date display).
    container = soup.find(attrs={"id": re.compile(r"^calendar")}) or soup
    # fallback: any table with 'calendar' in class
    table = container.find("table") or soup.find("table")

    last_date_text = None

    def extract_cells(tr):
        # get text of each cell (th/td)
        tds = tr.find_all(["td", "th"], recursive=False)
        # fallback: if none direct, allow deeper
        if not tds:
            tds = tr.find_all(["td", "th"])
        return tds

    # Collect <tr> candidates
    trs = []
    if table:
        trs = table.find_all("tr")
    if not trs:
        # fallback to any rows with a known class pattern
        trs = soup.find_all("tr")

    for tr in trs:
        cells = extract_cells(tr)
        if len(cells) < 5:
            continue  # not a data row

        # Heuristic extraction by column meaning
        # Try to map columns by best-effort keywords/classes
        ctexts = [_text(c.get_text(" ")) for c in cells]

        # Often orders like: Date | Time | Currency | Impact | Event | Actual | Forecast | Previous
        # We'll align by using first 8 cells if available.
        date_txt, time_txt, currency, impact_txt, event_txt, actual, forecast, previous = [""]*8

        # Try to locate a cell that looks like a currency (3 letters)
        currency_idx = None
        for i, t in enumerate(ctexts[:6]):
            if re.fullmatch(r"[A-Z]{3}", t):
                currency_idx = i
                break

        if currency_idx is None:
            # not a regular event row
            continue

        # We expect date & time to be before currency
        # Rough mapping around currency index
        # idx-2: date, idx-1: time, idx: CCY, idx+1: impact, idx+2: event, idx+3..: numbers
        def safe(idx):
            return cells[idx] if 0 <= idx < len(cells) else None

        # Date
        date_cell = safe(currency_idx - 2)
        if date_cell:
            dtxt = _text(date_cell.get_text(" "))
            # Keep if contains month/day pattern or weekday
            if dtxt and not re.fullmatch(r"(?:\d{1,2}:\d{2})|(?:\d{2}\:\d{2}\s*\w+)", dtxt, re.I):
                last_date_text = dtxt  # update carry-forward
        # Time
        time_cell = safe(currency_idx - 1)
        time_txt = _text(time_cell.get_text(" ")) if time_cell else ""

        # Currency
        currency = _text(cells[currency_idx].get_text(" "))

        # Impact
        impact_cell = safe(currency_idx + 1)
        impact_str = ""
        impact_lvl = 0
        if impact_cell:
            impact_str, impact_lvl = parse_impact(impact_cell)

        # Event
        event_cell = safe(currency_idx + 2)
        event_txt = _text(event_cell.get_text(" ")) if event_cell else ""

        # Actual, Forecast, Previous
        actual = _text((safe(currency_idx + 3) or {}).get_text(" ")) if safe(currency_idx + 3) else ""
        forecast = _text((safe(currency_idx + 4) or {}).get_text(" ")) if safe(currency_idx + 4) else ""
        previous = _text((safe(currency_idx + 5) or {}).get_text(" ")) if safe(currency_idx + 5) else ""

        # Build date value
        # last_date_text is something like "Mon Feb 05" or "Feb 05, 2024" etc.
        # We'll attempt to parse with current year if year is missing.
        date_val = None
        if last_date_text:
            try:
                d = dtparse.parse(last_date_text, default=datetime(year, month, 1))
                # If parser guessed a different month because of default, force requested year+month
                if d.year != year or d.month != month:
                    d = d.replace(year=year, month=month)
                date_val = d.date()
            except Exception:
                date_val = date(year, month, 1)
        else:
            # fallback to first day
            date_val = date(year, month, 1)

        # Normalize time (keep as text; also try to parse to HH:MM if possible)
        time_norm = time_txt
        # FF sometimes shows "All Day" / "Day 1" etc.
        if re.search(r"all\s*day|day\s*\d+", time_txt, re.I):
            time_norm = "00:00"

        rows.append({
            "date": date_val.isoformat(),
            "time": time_norm,
            "currency": currency,
            "impact": impact_str,
            "event": event_txt,
            "actual": actual,
            "forecast": forecast,
            "previous": previous,
            "impact_level": impact_lvl,
        })

    return rows

def fetch_month(year: int, month: int, session: requests.Session, sleep_sec: float) -> List[Dict]:
    url = month_url(year, month)
    for attempt in range(3):
        r = session.get(url, headers=HEADERS, cookies=COOKIES, timeout=30)
        if r.status_code == 200 and "<html" in r.text.lower():
            try:
                rows = parse_month(r.text, year, month)
                return rows
            except Exception:
                if attempt == 2:
                    raise
        time.sleep(sleep_sec)
    return []

def main():
    ap = argparse.ArgumentParser(description="Scrape ForexFactory economic calendar (monthly) to CSV.")
    ap.add_argument("--out_dir", required=True, help="Output directory (will be created).")
    ap.add_argument("--start_year", type=int, default=2023)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    yearly_dir = os.path.join(args.out_dir, "yearly")
    os.makedirs(yearly_dir, exist_ok=True)

    # Build list of (year, month)
    ym = []
    for y in range(args.start_year, args.end_year + 1):
        for m in range(1, 13):
            # If current year is the last and it's the future month, still fetch (it will return none gracefully)
            ym.append((y, m))
    # Trim future months if end year is current year and month has not happened — optional
    # (we keep simple)

    total = len(ym)
    all_rows: List[Dict] = []

    session = requests.Session()

    pbar = tqdm(total=100, desc="📊 Progress", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}%")
    completed_months = 0

    for (y, m) in ym:
        # Skip months out of today if wanted (optional)
        try:
            rows = fetch_month(y, m, session, args.sleep)
        except Exception as e:
            rows = []
        all_rows.extend(rows)

        completed_months += 1
        # Update a single-line 0–100 based on months completed
        pct = int(round(completed_months * 100.0 / total))
        pbar.n = pct
        pbar.refresh()

    pbar.n = 100
    pbar.refresh()
    pbar.close()

    # Make DataFrame
    df = pd.DataFrame(all_rows, columns=[
        "date", "time", "currency", "impact", "event",
        "actual", "forecast", "previous", "impact_level"
    ])

    # Clean up / normalize
    # Combine to datetime (UTC textual, we keep as naive)
    def parse_datetime(row):
        try:
            if row["time"] and re.match(r"^\d{1,2}:\d{2}", str(row["time"])):
                return pd.to_datetime(f"{row['date']} {row['time']}", errors="coerce")
            else:
                return pd.to_datetime(row["date"], errors="coerce")
        except Exception:
            return pd.NaT

    if not df.empty:
        df["datetime"] = df.apply(parse_datetime, axis=1)
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month

        # Drop rows not in the requested window or invalid dates
        mask = (df["year"] >= args.start_year) & (df["year"] <= args.end_year)
        before = len(df)
        df = df[mask].copy()
        dropped = before - len(df)

        # Sort
        df.sort_values(["datetime", "currency", "event"], inplace=True)

    # Save full CSV
    full_path = os.path.join(args.out_dir, f"forexfactory_{args.start_year}_{args.end_year}_full.csv")
    df.to_csv(full_path, index=False, encoding="utf-8-sig")

    # Save per-year
    if not df.empty and "year" in df.columns:
        for y, df_y in df.groupby("year"):
            outy = os.path.join(yearly_dir, f"forexfactory_{y}.csv")
            df_y.to_csv(outy, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved full CSV: {full_path}")
    if not df.empty and "year" in df.columns:
        print(f"📁 Yearly files in: {yearly_dir}")
        print(f"📊 Rows: {len(df):,} (dropped outside range/invalid: {dropped if 'dropped' in locals() else 0:,})")
    else:
        print("⚠️ No rows parsed. The site layout may have changed, or requests were blocked. Try increasing --sleep or running again.")
        

if __name__ == "__main__":
    main()
