#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Forex Factory Economic Calendar (2007–2025)
➡ Saves per-year CSVs
➡ Single-line 0–100% progress bar
"""

import os, csv, re, time, random, argparse
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

BASE = "https://www.forexfactory.com/calendar"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"}
COLS = ["date","time","currency","impact","event","actual","forecast","previous","source_url"]

# -------- Utilities --------
def _text(el):
    return " ".join(el.get_text(" ", strip=True).split()) if el else ""

def month_url(dt: datetime) -> str:
    return f"{BASE}?month={dt.strftime('%Y-%m-01')}"

def fetch_html(session, url, sleep, retries):
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and "<html" in r.text.lower():
                return r.text
        except Exception:
            pass
        time.sleep(sleep * (i + 1))
    return None

def parse_calendar(html, url):
    soup = BeautifulSoup(html, "html.parser")
    rows, cur_date = [], ""
    for tr in soup.find_all("tr", class_=re.compile("calendar__row")):
        cls = " ".join(tr.get("class", []))
        if "calendar__row--day" in cls:
            cur_date = _text(tr)
            continue
        t = _text(tr.find(class_="calendar__time"))
        c = _text(tr.find(class_="calendar__currency"))
        imp_el = tr.find(class_="calendar__impact")
        impact = ""
        if imp_el:
            m = re.search(r"impact--(high|medium|low)", " ".join(imp_el.get("class", [])))
            impact = m.group(1).capitalize() if m else _text(imp_el)
        ev = _text(tr.find(class_="calendar__event"))
        act = _text(tr.find(class_="calendar__actual"))
        fc = _text(tr.find(class_="calendar__forecast"))
        pv = _text(tr.find(class_="calendar__previous"))
        if cur_date and (ev or c):
            rows.append({
                "date": cur_date, "time": t, "currency": c, "impact": impact,
                "event": ev, "actual": act, "forecast": fc, "previous": pv, "source_url": url
            })
    return rows

# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="Download ForexFactory calendar 2007–2025")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start_year", type=int, default=2007)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--max_retries", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(args.start_year, 1, 1)
    end = datetime(args.end_year, 12, 31)
    total_months = 0
    cur = start
    while cur <= end:
        total_months += 1
        cur += relativedelta(months=+1)

    session = requests.Session()
    pbar = tqdm(total=100, ncols=80, bar_format="📊 {l_bar}{bar}| {n:.0f}%")

    cur = start
    done_months = 0
    year_data = []
    current_year = args.start_year

    while cur <= end:
        y, m = cur.year, cur.month
        url = month_url(cur)
        html = fetch_html(session, url, args.sleep, args.max_retries)
        if html:
            rows = parse_calendar(html, url)
            year_data.extend(rows)

        done_months += 1
        percent = (done_months / total_months) * 100
        pbar.n = percent
        pbar.refresh()

        if (m == 12) or (cur + relativedelta(months=+1)).year != y:
            # save yearly file
            out_path = out_dir / f"{y}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLS)
                w.writeheader()
                w.writerows(year_data)
            year_data = []
        time.sleep(args.sleep)
        cur += relativedelta(months=+1)

    pbar.n = 100
    pbar.refresh()
    pbar.close()
    print(f"\n✅ Done! Files saved in {out_dir.resolve()}")

if __name__ == "__main__":
    main()
