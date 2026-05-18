import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from preprocessing.normalize_description import (
    _remove_emojis,
    _remove_hashtags,
    _remove_phone_numbers,
    clean_description,
)


def test_clean_description_from_dataset():
    csv_path = ROOT / "data" / "episodes.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    original = row["description"]
    cleaned = clean_description(original)

    assert cleaned == cleaned.strip()
    assert cleaned == cleaned.lower()
    assert "#" not in cleaned
    assert "+34" not in cleaned
    assert "636" not in cleaned
    assert "14" not in cleaned
    assert "20" not in cleaned
    assert "whatsapp" in cleaned


def test_remove_emojis():
    text = "Esto es divertido 😄🔥"
    assert _remove_emojis(text) == "Esto es divertido "


def test_remove_hashtags():
    text = "Este es un #ejemplo de texto con #hashtag"
    assert _remove_hashtags(text) == "Este es un  de texto con "


def test_remove_phone_numbers_from_description():
    text = "Mándanos un audio por whatsapp al (+34) 636 75 14 20"
    cleaned = _remove_phone_numbers(text)
    assert "+34" not in cleaned
    assert "636" not in cleaned
    assert "whatsapp" in cleaned
