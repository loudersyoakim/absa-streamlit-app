"""
inference.py -- pipeline ABSA (ACD -> ASC) memakai IndoBERT/mBERT hasil
fine-tuning, plus parsing input dan scraping ulasan Tokopedia.
"""
from __future__ import annotations

import os
import json
import re
import time
from datetime import datetime
from typing import Callable, Optional, Dict, List, Any, Union
from urllib.parse import urlparse, unquote

import torch
import requests
from bs4 import BeautifulSoup
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import ASPECTS

MODEL_DIR = "model"
MODEL_PATHS = {
    "IndoBERT": {
        "ACD": os.path.join(MODEL_DIR, "indobert_acd", "best_model"),
        "ASC": os.path.join(MODEL_DIR, "indobert_asc", "best_model"),
    },
    "mBERT": {
        "ACD": os.path.join(MODEL_DIR, "mbert_acd", "best_model"),
        "ASC": os.path.join(MODEL_DIR, "mbert_asc", "best_model"),
    },
}
MODEL_KEY_ALIASES = {"IndoBERT": "indobert", "mBERT": "mbert"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ProgressFn = Optional[Callable[[str], None]]
DEFAULT_THRESHOLD = 0.5

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}


class InvalidTokopediaURLError(Exception):
    pass


class ReviewsNotFoundError(Exception):
    pass


def _noop(_msg: str) -> None:
    return None


@st.cache_resource(show_spinner=False)
def load_label_maps() -> dict:
    path = os.path.join(MODEL_DIR, "label_maps.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models(model_choice: str):
    paths = MODEL_PATHS[model_choice]

    tokenizer_acd = AutoTokenizer.from_pretrained(paths["ACD"])
    model_acd = AutoModelForSequenceClassification.from_pretrained(paths["ACD"]).to(DEVICE)
    model_acd.eval()

    tokenizer_asc = AutoTokenizer.from_pretrained(paths["ASC"])
    model_asc = AutoModelForSequenceClassification.from_pretrained(paths["ASC"]).to(DEVICE)
    model_asc.eval()

    return tokenizer_acd, model_acd, tokenizer_asc, model_asc


@st.cache_resource(show_spinner=False)
def load_acd_threshold(model_choice: str) -> float:
    path = os.path.join(MODEL_DIR, "acd_optimal_thresholds.json")
    key = MODEL_KEY_ALIASES.get(model_choice, model_choice.lower())
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data[key]["threshold"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return DEFAULT_THRESHOLD


def get_model_info(model_type: str) -> Dict[str, Any]:
    maps = load_label_maps()
    return {
        "type": model_type,
        "acd_model": "Aspect Category Detection",
        "asc_model": "Aspect-based Sentiment Classification",
        "aspects": [ASPECTS.get(code, code) for code in maps.get("aspect_labels", [])],
    }


def _empty_result_shell(model_choice: str, source_preview: str) -> dict:
    return {
        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
        "model": model_choice,
        "source_preview": source_preview[:160],
        "aspects": {},
        "totals": {"Positif": 0, "Negatif": 0, "Netral": 0},
        "most_positive_aspect": "Tidak ada",
        "most_negative_aspect": "Tidak ada",
        "n_reviews_processed": 0,
    }


def _analyze_single(text: str, tok_acd, mod_acd, tok_asc, mod_asc, maps: dict, threshold: float) -> dict:
    aspect_labels = maps["aspect_labels"]
    sentiment_labels = maps["sentiment_labels"]
    id2sent = {i: lbl for i, lbl in enumerate(sentiment_labels)}

    inputs_acd = tok_acd(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits_acd = mod_acd(**inputs_acd).logits
        probs_acd = torch.sigmoid(logits_acd).squeeze(0).cpu().tolist()

    if isinstance(probs_acd, float):
        probs_acd = [probs_acd]

    detected = [aspect_labels[i] for i, p in enumerate(probs_acd) if p >= threshold and i < len(aspect_labels)]

    per_aspect_sentiment = {}
    for aspect in detected:
        asc_input = f"{aspect} {text}"
        inputs_asc = tok_asc(asc_input, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            logits_asc = mod_asc(**inputs_asc).logits
            pred_idx = int(torch.argmax(logits_asc, dim=1).item())
        per_aspect_sentiment[aspect] = id2sent.get(pred_idx, "Netral")

    return per_aspect_sentiment


def analyze_texts_local(
    texts: list[str],
    model_choice: str,
    threshold: float,
    progress: ProgressFn = None,
    source_label: str = "",
) -> dict:
    progress = progress or _noop
    texts = [t.strip() for t in texts if t and t.strip()]

    if not texts:
        return _empty_result_shell(model_choice, source_label)

    source_preview = source_label or texts[0]

    progress(f"Memuat model {model_choice}")
    tok_acd, mod_acd, tok_asc, mod_asc = load_models(model_choice)
    maps = load_label_maps()

    aspects_result: dict[str, dict] = {}
    totals = {"Positif": 0, "Negatif": 0, "Netral": 0}

    progress("Model sedang mendeteksi aspek yang dibahas dalam ulasan")
    for text in texts:
        per_aspect_sentiment = _analyze_single(text, tok_acd, mod_acd, tok_asc, mod_asc, maps, threshold)

        for aspect, sent_label in per_aspect_sentiment.items():
            bucket = aspects_result.setdefault(
                aspect,
                {
                    "name": ASPECTS.get(aspect, aspect),
                    "counts": {"Positif": 0, "Negatif": 0, "Netral": 0},
                    "total": 0,
                    "reviews": {"Positif": [], "Negatif": [], "Netral": []},
                },
            )
            bucket["counts"][sent_label] += 1
            bucket["total"] += 1
            if len(bucket["reviews"][sent_label]) < 30:
                bucket["reviews"][sent_label].append(text[:220])
            totals[sent_label] += 1

    progress("Mengklasifikasikan sentimen untuk setiap aspek yang terdeteksi")

    def _ratio(code, key):
        c = aspects_result[code]["counts"]
        tot = aspects_result[code]["total"] or 1
        return c[key] / tot

    if aspects_result:
        eligible = list(aspects_result.keys())
        most_positive = max(eligible, key=lambda a: (_ratio(a, "Positif"), aspects_result[a]["counts"]["Positif"]))
        most_negative = max(eligible, key=lambda a: (_ratio(a, "Negatif"), aspects_result[a]["counts"]["Negatif"]))
    else:
        most_positive = most_negative = "Tidak ada"

    progress("Menyusun hasil analisis")

    return {
        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
        "model": model_choice,
        "source_preview": source_preview[:160],
        "aspects": aspects_result,
        "totals": totals,
        "most_positive_aspect": most_positive,
        "most_negative_aspect": most_negative,
        "n_reviews_processed": len(texts),
    }


def analyze_text_local(text: str, model_choice: str, threshold: float, progress: ProgressFn = None) -> dict:
    return analyze_texts_local([text], model_choice, threshold, progress=progress, source_label=text)


def parse_text_input(text: str) -> List[str]:
    return [text.strip()]


def parse_file_input(files: List) -> List[str]:
    reviews: List[str] = []
    for file in files:
        file_type = file.name.split(".")[-1].lower()
        try:
            if file_type == "csv":
                import pandas as pd
                df = pd.read_csv(file)
                col = df.columns[0]
                reviews.extend(df[col].astype(str).tolist())
            elif file_type == "txt":
                content = file.read().decode("utf-8")
                reviews.extend([line.strip() for line in content.split("\n") if line.strip()])
            elif file_type == "xlsx":
                import pandas as pd
                df = pd.read_excel(file)
                col = df.columns[0]
                reviews.extend(df[col].astype(str).tolist())
        except Exception as e:
            print(f"Gagal membaca berkas {file.name}: {e}")
            continue
    return reviews


def _is_valid_tokopedia_product_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if "tokopedia.com" not in parsed.netloc:
        return False
    path_parts = [p for p in unquote(parsed.path).split("/") if p]
    return len(path_parts) >= 2


def _build_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1366,1000")
    options.add_argument(f"user-agent={REQUEST_HEADERS['User-Agent']}")

    try:
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    except Exception as e:
        raise InvalidTokopediaURLError("URL yang diinput tidak valid.") from e


def _extract_reviews_from_page_source(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    review_section = soup.find("section", id="review-feed")
    if not review_section:
        return []

    reviews = []
    for article in review_section.find_all("article", class_=lambda c: c and "css-" in c):
        span = article.find("span", {"data-testid": "lblItemUlasan"})
        if span:
            text = span.get_text(strip=True)
            if len(text) >= 5:
                reviews.append(text)
    return reviews


def scrape_tokopedia_reviews(url: str) -> tuple[List[str], str]:
    """
    Scraping ulasan sungguhan pakai headless Chrome, karena Tokopedia
    memuat ulasan lewat JavaScript setelah tab "Ulasan" diklik dan
    halamannya di-scroll -- bukan ada di HTML mentah. Dibuat "keras":
    tiap langkah yang rawan gagal (buka halaman, tutup popup, klik tab,
    klik halaman berikutnya) di-retry beberapa kali sebelum menyerah,
    dan paginasi terus jalan sampai benar-benar tidak ada ulasan baru
    lagi atau tombol halaman berikutnya sudah tidak aktif.

    Mengembalikan (daftar_ulasan, nama_produk). Nama produk diambil dari
    <h1 data-testid="lblPDPDetailProductName"> yang sudah ada di HTML
    awal (bukan hasil lazy-load), jadi bisa diambil segera setelah
    halaman terbuka, sebelum urusan klik tab ulasan.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException

    if not _is_valid_tokopedia_product_url(url):
        raise InvalidTokopediaURLError("URL yang diinput tidak valid.")

    driver = _build_driver()
    all_reviews: List[str] = []
    product_title = ""

    try:
        opened = False
        for attempt in range(3):
            try:
                driver.set_page_load_timeout(30)
                driver.get(url)
                opened = True
                break
            except WebDriverException:
                continue
        if not opened:
            raise InvalidTokopediaURLError("URL yang diinput tidak valid.")

        try:
            title_el = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="lblPDPDetailProductName"]'))
            )
            product_title = title_el.text.strip()
        except TimeoutException:
            product_title = ""

        try:
            close_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "div.css-11hzwo5 button"))
            )
            driver.execute_script("arguments[0].click();", close_button)
            time.sleep(1)
        except TimeoutException:
            pass

        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(1)

        review_tab_clicked = False
        for attempt in range(3):
            try:
                review_tab = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Ulasan')]"))
                )
                driver.execute_script("arguments[0].click();", review_tab)
                review_tab_clicked = True
                break
            except TimeoutException:
                driver.execute_script("window.scrollBy(0, 300);")
                time.sleep(1)

        if not review_tab_clicked:
            raise ReviewsNotFoundError("Mohon maaf, ulasan tidak berhasil didapatkan.")

        time.sleep(3)

        page_count = 1
        seen_reviews = set()
        max_pages = 20
        consecutive_empty_pages = 0

        while page_count <= max_pages:
            try:
                more_buttons = driver.find_elements(
                    By.XPATH, "//span[text()='Selengkapnya'] | //button[text()='Selengkapnya']"
                )
                for btn in more_buttons:
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.1)
                    except Exception:
                        pass
                if more_buttons:
                    time.sleep(1)
            except Exception:
                pass

            page_reviews = _extract_reviews_from_page_source(driver.page_source)

            new_found = 0
            for review in page_reviews:
                if review not in seen_reviews:
                    seen_reviews.add(review)
                    all_reviews.append(review)
                    new_found += 1

            if new_found == 0:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 2:
                    break
            else:
                consecutive_empty_pages = 0

            next_clicked = False
            for attempt in range(2):
                try:
                    next_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Laman berikutnya']"))
                    )
                    driver.execute_script("arguments[0].click();", next_button)
                    next_clicked = True
                    time.sleep(2)
                    break
                except TimeoutException:
                    time.sleep(1)
                    continue

            if not next_clicked:
                break

            page_count += 1

    finally:
        driver.quit()

    if not all_reviews:
        raise ReviewsNotFoundError("Mohon maaf, ulasan tidak berhasil didapatkan.")

    return all_reviews, product_title


def preprocess_text(text: str) -> str:
    return text.strip()


def _aspect_summary(aspect_code: str, bucket: dict) -> Dict[str, Any]:
    counts = bucket["counts"]
    total = bucket["total"] or 0
    return {
        "aspect": bucket.get("name", aspect_code),
        "positive": counts["Positif"],
        "neutral": counts["Netral"],
        "negative": counts["Negatif"],
        "total": total,
    }


def _calculate_top_discussed(aspects_result: Dict[str, dict]) -> Dict[str, Any]:
    if not aspects_result:
        return {"aspect": "Tidak ada data", "percentage": 0, "positive": 0, "neutral": 0, "negative": 0, "total": 0}

    grand_total = sum(b["total"] for b in aspects_result.values()) or 1
    top_code = max(aspects_result, key=lambda a: aspects_result[a]["total"])
    summary = _aspect_summary(top_code, aspects_result[top_code])
    summary["percentage"] = (summary["total"] / grand_total) * 100
    return summary


def _calculate_top_complained(aspects_result: Dict[str, dict]) -> Dict[str, Any]:
    eligible = {a: b for a, b in aspects_result.items() if b["counts"]["Negatif"] > 0}
    if not eligible:
        return {"aspect": "Tidak ada data", "percentage": 0, "positive": 0, "neutral": 0, "negative": 0, "total": 0}

    def _ratio(code):
        b = eligible[code]
        return b["counts"]["Negatif"] / (b["total"] or 1)

    top_code = max(eligible, key=lambda a: (eligible[a]["counts"]["Negatif"], _ratio(a)))
    summary = _aspect_summary(top_code, eligible[top_code])
    summary["percentage"] = _ratio(top_code) * 100
    return summary


def _calculate_overall_sentiment(totals: Dict[str, int]) -> Dict[str, int]:
    return {
        "Positif": totals.get("Positif", 0),
        "Netral": totals.get("Netral", 0),
        "Negatif": totals.get("Negatif", 0),
    }


def _calculate_aspect_distribution(aspects_result: Dict[str, dict]) -> Dict[str, int]:
    return {bucket.get("name", code): bucket["total"] for code, bucket in aspects_result.items()}


def _structure_detailed_results(aspects_result: Dict[str, dict]) -> List[Dict[str, Any]]:
    detailed = []
    for code, bucket in aspects_result.items():
        detailed.append({
            "aspect": bucket.get("name", code),
            "positive_count": bucket["counts"]["Positif"],
            "neutral_count": bucket["counts"]["Netral"],
            "negative_count": bucket["counts"]["Negatif"],
            "positive_reviews": bucket["reviews"]["Positif"],
            "neutral_reviews": bucket["reviews"]["Netral"],
            "negative_reviews": bucket["reviews"]["Negatif"],
        })
    return detailed


def _first_n_words(text: str, n: int = 5) -> str:
    words = text.strip().split()
    if not words:
        return ""
    snippet = " ".join(words[:n])
    return snippet + ("..." if len(words) > n else "")


def run_inference(
    input_data: Union[str, List],
    model_type: str = "IndoBERT",
    mode: str = "Teks",
    threshold: Optional[float] = None,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    progress = progress or _noop

    progress("Menyiapkan data ulasan")
    if mode == "Teks":
        reviews = parse_text_input(input_data)
        source_label = input_data
        judul_ulasan = _first_n_words(input_data)
    elif mode == "Upload File":
        reviews = parse_file_input(input_data)
        source_label = f"{len(input_data)} berkas diunggah" if input_data else ""
        judul_ulasan = _first_n_words(reviews[0]) if reviews else "Tidak ada ulasan"
    elif mode == "URL Tokopedia":
        reviews, product_title = scrape_tokopedia_reviews(input_data)
        source_label = input_data
        judul_ulasan = product_title or input_data
    else:
        raise ValueError(f"Mode input tidak dikenal: {mode}")

    if not reviews:
        raise ReviewsNotFoundError("Mohon maaf, ulasan tidak berhasil didapatkan.")

    progress("Membaca dan membersihkan teks ulasan")
    reviews = [preprocess_text(r) for r in reviews if r and r.strip()]

    if threshold is None:
        threshold = load_acd_threshold(model_type)

    hasil = analyze_texts_local(
        reviews,
        model_choice=model_type,
        threshold=threshold,
        progress=progress,
        source_label=source_label,
    )

    progress("Menghitung statistik akhir")
    aspects_result = hasil["aspects"]
    hasil["mode"] = mode
    hasil["judul_ulasan"] = judul_ulasan
    hasil["top_discussed"] = _calculate_top_discussed(aspects_result)
    hasil["top_complained"] = _calculate_top_complained(aspects_result)
    hasil["overall_sentiment_distribution"] = _calculate_overall_sentiment(hasil["totals"])
    hasil["aspect_distribution"] = _calculate_aspect_distribution(aspects_result)
    hasil["aspects_detailed"] = _structure_detailed_results(aspects_result)

    return hasil