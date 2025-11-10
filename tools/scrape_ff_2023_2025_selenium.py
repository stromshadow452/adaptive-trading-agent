#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Scrape ForexFactory calendar (month-by-month) with undetected-chromedriver (2023–2025).

Usage (PowerShell):
  python tools\scrape_ff_2023_2025_selenium.py --out_dir "data\ff_test" --start_year 2023 --end_year 2023 --sleep 2.5
  # add --headless to run without a visible browser

What it does:
- Opens https://www.forexfactory.com/calendar?month=<Mon.YYYY>
- Dismisses typical cookie banners if found
- Scrolls to bottom to trigger lazy loading
- Extracts (time, currency, impact, event, actual, forecast, previous, date)
- Saves monthly CSV + one full CSV
- Handles Cloudflare “verifying you are human” with undetected-chromedriver
"""

import re
import time
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

COLS = ["date", "time", "currency", "impact", "event",
        "actual", "forecast", "previous", "year", "month",
        "datetime_local", "site_timezone"]


# ---------------------- Chrome setup ----------------------

PROFILE_DIR = Path("data/.ff_uc_profile")  # store cookies between runs

def chrome(headless: bool):
    options = uc.ChromeOptions()
    options.add_argument(f"--user-data-dir={PROFILE_DIR.resolve()}")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1600,2200")
    options.add_argument("--lang=en-US,en")
    if headless:
        options.add_argument("--headless=new")

    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(60)
    return driver


# ---------------------- Cloudflare handling ----------------------

def is_cf_challenge(driver) -> bool:
    """Detect Cloudflare challenge page."""
    try:
        title = driver.title.lower()
        if "just a moment" in title or "checking your browser" in title:
            return True
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        if "verifying you are human" in body or "review the security" in body:
            return True
    except Exception:
        pass
    return False

def wait_cf_or_ready(driver, max_wait=30):
    start = time.time()
    while time.time() - start < max_wait:
        if not is_cf_challenge(driver):
            return True
        time.sleep(1.5)
    return False


# ---------------------- Site helpers ----------------------

def build_url(year: int, month: int) -> str:
    return f"https://www.forexfactory.com/calendar?month={MONTHS[month-1]}.{year}"

def click_if_present(driver, by, value, timeout=5):
    try:
        el = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((by, value)))
        el.click()
        return True
    except Exception:
        return False

def handle_cookies_and_modals(driver):
    selectors = [
        (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
        (By.XPATH, "//button[contains(., 'Accept All')]"),
        (By.XPATH, "//button[contains(., 'Accept')]"),
    ]
    for by, sel in selectors:
        if click_if_present(driver, by, sel):
            break

def smooth_scroll_until_stable(driver, pause: float = 0.25, max_rounds: int = 20):
    """Scroll down gradually until height stops increasing."""
    last_h = 0
    same = 0
    for _ in range(max_rounds):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight/4);")
        time.sleep(pause)
        h = driver.execute_script("return document.body.scrollHeight;")
        if h == last_h:
            same += 1
            if same >= 2:
                break
        else:
            same = 0
        last_h = h


# ---------------------- Data extraction ----------------------

def get_site_timezone_text(driver) -> Optional[str]:
    try:
        el = driver.find_element(By.XPATH, "//*[contains(text(),'Time:') or contains(text(),'Times are shown')]")
        return el.text.strip()
    except Exception:
        return None

def extract_rows(driver, site_tz: Optional[str]) -> List[Dict]:
    rows = []
    selectors = [
        "[data-event-id]",
        "[data-entity-id]",
        "table.calendar__table tbody tr",
        "table tbody tr"
    ]

    def find_text(el, css_list):
        for sel in css_list:
            try:
                txt = el.find_element(By.CSS_SELECTOR, sel).text.strip()
                if txt:
                    return txt
            except Exception:
                continue
        return None

    def grab(el):
        return dict(
            time=find_text(el, [".time", "td.time"]),
            currency=find_text(el, [".currency", "td.currency"]),
            impact=find_text(el, [".impact", "td.impact"]),
            event=find_text(el, [".event", "td.event"]),
            actual=find_text(el, [".actual", "td.actual"]),
            forecast=find_text(el, [".forecast", "td.forecast"]),
            previous=find_text(el, [".previous", "td.previous"]),
            date=None,
            site_timezone=site_tz
        )

    for sel in selectors:
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
        except Exception:
            elems = []
        if len(elems) > 20:
            for el in elems:
                data = grab(el)
                if any(data.get(k) for k in ("event", "currency", "time")):
                    rows.append(data)
            if rows:
                break
    return rows


# ---------------------- Output helpers ----------------------

def _s(val):
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val).strip()

def month_to_csv(rows: List[Dict], year: int, month: int, out_dir: Path) -> Path:
    df = pd.DataFrame(rows)
    df["year"] = year
    df["month"] = month
    default_date = f"01 {MONTHS[month-1].title()} {year}"
    if "date" not in df.columns:
        df["date"] = default_date
    else:
        df["date"] = df["date"].fillna(default_date)

    def parse_dt(row):
        t = _s(row.get("time"))
        d = _s(row.get("date"))
        s = f"{d} {t}" if re.match(r"^\d{1,2}:\d{2}", t) else d
        for fmt in ("%d %b %Y %H:%M", "%d %b %Y",
                    "%d %B %Y %H:%M", "%d %B %Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return pd.NaT

    df["datetime_local"] = df.apply(parse_dt, axis=1)

    keep_mask = df[["event", "currency", "time"]].notna().any(axis=1)
    df = df[keep_mask].copy()

    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLS]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"forexfactory_{year}_{month:02d}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    return out_file

def save_debug(driver, out_dir: Path, year: int, month: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"DEBUG_{year}_{month:02d}.html"
    png_path = out_dir / f"DEBUG_{year}_{month:02d}.png"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    driver.save_screenshot(str(png_path))


# ---------------------- Main ----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start_year", type=int, default=2023)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=2.5)
    ap.add_argument("--headless", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Output: {out_dir}")
    print(f"📅 Range: {args.start_year}–{args.end_year}")
    print(f"⏳ Delay per page: {args.sleep}s\n")

    driver = chrome(headless=args.headless)
    all_rows = []

    try:
        months = [(y, m) for y in range(args.start_year, args.end_year + 1) for m in range(1, 13)]
        today = date.today()

        for (year, month) in tqdm(months, desc="📊 Progress", ncols=100):
            if (year, month) > (today.year, today.month):
                continue

            url = build_url(year, month)
            for attempt in range(3):
                try:
                    driver.get(url)
                except Exception:
                    time.sleep(2)
                    continue

                if is_cf_challenge(driver):
                    if wait_cf_or_ready(driver, max_wait=30):
                        break
                    time.sleep(5 + attempt * 5)
                    continue
                break

            if is_cf_challenge(driver):
                print(f"⚠️  Skipping {year}-{month:02d}, Cloudflare check failed.")
                save_debug(driver, out_dir, year, month)
                continue

            handle_cookies_and_modals(driver)
            time.sleep(args.sleep)
            smooth_scroll_until_stable(driver, pause=0.25, max_rounds=20)
            time.sleep(args.sleep)

            site_tz = get_site_timezone_text(driver)
            rows = extract_rows(driver, site_tz)
            if not rows:
                save_debug(driver, out_dir, year, month)
            out_file = month_to_csv(rows, year, month, out_dir)
            all_rows.extend(rows)
            time.sleep(0.5)

    finally:
        driver.quit()

    df_all = pd.DataFrame(all_rows)
    for c in COLS:
        if c not in df_all.columns:
            df_all[c] = pd.NA
    df_all = df_all[COLS]
    out_all = out_dir / f"forexfactory_{args.start_year}_{args.end_year}_full.csv"
    df_all.to_csv(out_all, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved all CSVs in {out_dir}")
    print(f"✅ Full file: {out_all}")


if __name__ == "__main__":
    main()
