"""
utils.py -- data acuan kategori aspek (Tabel 3.2) dan fungsi bantu umum
yang dipakai bersama oleh inference.py dan app.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

MODEL_DIR = "model"

ASPECTS: Dict[str, str] = {
    "A1": "Kinerja Visual",
    "A2": "Keandalan & Cacat Produk",
    "A3": "Desain, Kualitas Fisik & Fitur",
    "B1": "Komunikasi & Dukungan Penjual",
    "B2": "Layanan Purna Jual",
    "B3": "Akurasi Informasi Produk",
    "C1": "Pengemasan & Pengiriman",
    "C2": "Harga & Nilai",
    "D1": "Umum",
}

ASPECT_DESC: Dict[str, str] = {
    "A1": "Kualitas panel saat berfungsi normal: resolusi, akurasi warna, refresh rate, ghosting, kecerahan.",
    "A2": "Cacat bawaan (dead pixel, backlight bleed), kestabilan jangka panjang, fungsi elektronik.",
    "A3": "Build quality, ergonomi (stand/bracket), konektivitas port, fitur tambahan, aksesoris.",
    "B1": "Kecepatan respons, keramahan, dan profesionalisme penjual saat dihubungi.",
    "B2": "Proses klaim garansi, retur, dan penanganan komplain pasca-pembelian.",
    "B3": "Kesesuaian deskripsi/foto produk pada etalase dengan unit fisik yang diterima.",
    "C1": "Standar packing, keamanan saat pengiriman, dan ketepatan waktu.",
    "C2": "Persepsi harga dibanding fitur (value for money), kewajaran harga, promo.",
    "D1": "Ungkapan umum yang tidak merujuk objek spesifik (salam, terima kasih).",
}

SENTIMENTS = ["Positif", "Negatif", "Netral"]


def _load_optimal_thresholds() -> dict:
    path = os.path.join(MODEL_DIR, "acd_optimal_thresholds.json")
    fallback = {
        "indobert": {"threshold": 0.45, "f1_macro_at_threshold": None},
        "mbert": {"threshold": 0.30, "f1_macro_at_threshold": None},
    }
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback


_THRESH = _load_optimal_thresholds()

MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "IndoBERT": {
        "f1_macro_acd": _THRESH.get("indobert", {}).get("f1_macro_at_threshold") or 0.8778,
        "f1_macro_asc": 0.9074,
        "default_threshold": _THRESH.get("indobert", {}).get("threshold", 0.45),
    },
    "mBERT": {
        "f1_macro_acd": _THRESH.get("mbert", {}).get("f1_macro_at_threshold") or 0.8699,
        "f1_macro_asc": 0.8956,
        "default_threshold": _THRESH.get("mbert", {}).get("threshold", 0.30),
    },
}


def model_weights_available(model_choice: str) -> bool:
    """Inferensi sekarang dijalankan lewat HuggingFace Inference API (bukan
    model lokal), jadi ketersediaannya baru benar-benar diperiksa saat
    inference._hf_infer() dipanggil (butuh HF_TOKEN valid & repo bisa
    diakses). Fungsi ini dipertahankan untuk kompatibilitas pemanggil lama
    dan selalu True di sini."""
    return True


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    if len(text) > max_length:
        return text[: max_length - len(suffix)] + suffix
    return text


def format_number(num: int, decimal_places: int = 0) -> str:
    return f"{num:,.{decimal_places}f}".replace(",", ".")


def format_percentage(value: float, decimal_places: int = 1) -> str:
    return f"{value:.{decimal_places}f}%"


def sanitize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentiment_color(sentiment: str) -> str:
    colors = {
        "positif": "#10B981", "positive": "#10B981",
        "netral": "#6B7280", "neutral": "#6B7280",
        "negatif": "#EF4444", "negative": "#EF4444",
    }
    return colors.get(sentiment.lower(), "#6B7280")


def build_report_md(result: dict, title: str = "Laporan Analisis ABSA") -> str:
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Model digunakan:** {result['model']}")
    lines.append(f"- **Waktu analisis:** {result['timestamp']}")
    lines.append(f"- **Sumber:** {result.get('source_preview', '-')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Ringkasan Umum")
    lines.append("")
    total = result["totals"]
    grand_total = sum(total.values()) or 1
    lines.append("| Sentimen | Jumlah | Persentase |")
    lines.append("|---|---|---|")
    for s in SENTIMENTS:
        pct = 100 * total[s] / grand_total
        lines.append(f"| {s} | {total[s]} | {pct:.1f}% |")
    lines.append("")
    lines.append("**Distribusi Kategori Aspek:**")
    lines.append("")
    lines.append("| Kode | Aspek | Jumlah Ulasan |")
    lines.append("|---|---|---|")
    for code, data in result["aspects"].items():
        lines.append(f"| {code} | {data['name']} | {data['total']} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Sebaran Sentimen per Aspek")
    lines.append("")
    lines.append("| Aspek | Positif | Negatif | Netral | Total |")
    lines.append("|---|---|---|---|---|")
    for code, data in result["aspects"].items():
        c = data["counts"]
        lines.append(f"| {data['name']} | {c['Positif']} | {c['Negatif']} | {c['Netral']} | {data['total']} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Detail Ulasan per Aspek")
    lines.append("")
    for code, data in result["aspects"].items():
        lines.append(f"### {data['name']} ({code})")
        lines.append("")
        for s in SENTIMENTS:
            revs = data["reviews"].get(s, [])
            if not revs:
                continue
            lines.append(f"**{s}:**")
            for r in revs:
                lines.append(f"- {r}")
            lines.append("")
    lines.append(f"*Laporan dibuat otomatis pada {result['timestamp']}.*")

    return "\n".join(lines)