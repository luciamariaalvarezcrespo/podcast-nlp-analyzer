
"""Fine-tunea roberta-base para clasificar episodios de podcast en los temas
definidos en zero_shot.py.

Los datos de entrenamiento son las pseudo-etiquetas generadas por el
zero-shot classifier. Solo usamos los ejemplos donde el modelo zero-shot
tenía alta confianza (score > MIN_CONFIDENCE).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Modelo base: RoBERTa de OpenAI/Facebook
BASE_MODEL = "roberta-base"

# Solo usamos ejemplos donde el zero-shot tenía esta confianza mínima.
MIN_CONFIDENCE = 0.50

# Guardamos el modelo fine-tuneado
MODEL_OUTPUT_DIR = Path("models/topic_classifier")

# Hiperparámetros de entrenamiento
# Warning: es un dataset pequeño (cuidado con el overfitting!!)
TRAINING_ARGS = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 2e-5,        # estándar para fine-tuning de transformers
    "weight_decay": 0.01,         # regularización para evitar overfitting
    "warmup_ratio": 0.1,          # 10% del entrenamiento de calentamiento
    "eval_strategy": "epoch",     # evalúa al final de cada epoch
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "logging_steps": 10,
    "seed": 42,
}

MAX_TOKEN_LENGTH = 500  # Las descripciones no suelen ser más largas


# ---------------------------------------------------------------------------
# Paso 1 — Preparar los datos
# ---------------------------------------------------------------------------

def prepare_data(
    df_labeled: pd.DataFrame,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    """
    Filtra ejemplos de alta confianza y construye los mapeos label/id.

    El modelo necesita etiquetas numéricas internamente (0, 1, 2...).
    Guardamos los mapeos para poder interpretar las predicciones después.
    """
    # Filtrar baja confianza y los "sin clasificar"
    df_filtered = df_labeled[
        (df_labeled["topic_score"] >= min_confidence)
        & (df_labeled["predicted_topic"] != "sin clasificar")
        & (df_labeled["clean_description"].str.len() > 20)
    ].copy()

    n_before = len(df_labeled)
    n_after = len(df_filtered)
    log.info(
        f"Ejemplos: {n_before} total → {n_after} con confianza ≥ {min_confidence} "
        f"({n_before - n_after} descartados)"
    )

    if n_after < 10:
        raise ValueError(
            f"Solo quedan {n_after} ejemplos después de filtrar. "
            f"Prueba a bajar MIN_CONFIDENCE (ahora: {min_confidence}) "
            f"o a minar más episodios con EPISODES_PER_SHOW = 50."
        )

    # Mostrar distribución de clases
    class_dist = df_filtered["predicted_topic"].value_counts()
    log.info(f"Distribución de clases:\n{class_dist.to_string()}")

    # Construir mapeos label/id
    labels = sorted(df_filtered["predicted_topic"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    df_filtered["label"] = df_filtered["predicted_topic"].map(label2id)

    return df_filtered, label2id, id2label


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en entrenamiento y validación.
    """
    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    log.info(f"Split: {len(df_train)} train / {len(df_val)} val")
    return df_train, df_val


# ---------------------------------------------------------------------------
# Paso 2 — Dataset compatible con HuggingFace Trainer
# ---------------------------------------------------------------------------

class PodcastDataset:
    """
    Convierte nuestro DataFrame en el formato que espera
    el Trainer de HuggingFace: un objeto con __len__ y __getitem__.

    HuggingFace Trainer espera que __getitem__ devuelva un dict con
    'input_ids', 'attention_mask' y 'labels'.
    El tokenizador genera los dos primeros y luego yo añado 'labels'.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_TOKEN_LENGTH):
        self.labels = df["label"].tolist()

        # Tokenizamos todo de una vez
        # truncation=True: corta si el texto es más largo que max_length
        # padding="max_length": rellena con [PAD] si es más corto
        self.encodings = tokenizer(
            df["clean_description"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",  # devuelve tensores de PyTorch
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Devuelve un ejemplo individual como dict de tensores
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Paso 3 — Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(
    df_labeled: pd.DataFrame,
    output_dir: Path = MODEL_OUTPUT_DIR,
    min_confidence: float = MIN_CONFIDENCE,
) -> Path:
    """
    Pipeline completo de fine-tuning: prepara datos, tokeniza, entrena y guarda.
    """
    # Preparar datos
    df_filtered, label2id, id2label = prepare_data(df_labeled, min_confidence)
    df_train, df_val = split_data(df_filtered)

    num_labels = len(label2id)
    log.info(f"Clases: {num_labels} | Train: {len(df_train)} | Val: {len(df_val)}")

    # Cargar tokenizador y modelo base
    log.info(f"Cargando {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # AutoModelForSequenceClassification añade automáticamente la capa de
    # clasificación encima de RoBERTa. num_labels dice cuántas clases hay.
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Crear datasets
    train_dataset = PodcastDataset(df_train, tokenizer)
    val_dataset = PodcastDataset(df_val, tokenizer)

    # Configurar entrenamiento
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **TRAINING_ARGS,
    )

    # Entrenar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    log.info("Iniciando entrenamiento...")
    trainer.train()
    log.info("Entrenamiento completado")

    # Guardar modelo + tokenizador + mapeos de etiquetas
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Guardar los mapeos para poder interpretar predicciones
    with open(output_dir / "label_mappings.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, ensure_ascii=False, indent=2)

    log.info(f"Modelo guardado en {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Inferencia con el modelo fine-tuneado
# ---------------------------------------------------------------------------

def load_finetuned_classifier(model_dir: Path = MODEL_OUTPUT_DIR):
    """
    Carga el modelo fine-tuneado para hacer predicciones.

    Devuelve un pipeline de HuggingFace listo para usar.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo en {model_dir}. "
            f"Ejecuta fine_tune() primero."
        )

    return pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )


def predict_topic(classifier, text: str) -> tuple[str, float]:
    """
    Predice el tema de un texto con el modelo fine-tuneado.
    """
    if not text or len(text.strip()) < 20:
        return "sin clasificar", 0.0

    result = classifier(text)[0]
    return result["label"], round(result["score"], 4)


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    labeled_path = Path("data/processed/episodes_labeled.csv")
    if not labeled_path.exists():
        print(f"No se encuentra {labeled_path}")
        print("Ejecuta primero el zero-shot:")
        print("  python src/analysis/zero_shot.py")
        sys.exit(1)

    df = pd.read_csv(labeled_path)
    log.info(f"Cargados {len(df)} episodios etiquetados")

    model_path = fine_tune(df)

    # Probar el modelo recién entrenado
    log.info("Probando el modelo fine-tuneado...")
    clf = load_finetuned_classifier(model_path)

    test_texts = [
        "Entrevistamos a una abogada especializada en violencia de género y órdenes de alejamiento.",
        "Hablamos sobre la baja de maternidad, la crianza y cómo conciliar con el trabajo.",
        "Analizamos la brecha salarial entre hombres y mujeres en el sector tecnológico.",
    ]

    print("\n--- Predicciones del modelo fine-tuneado ---")
    for text in test_texts:
        label, score = predict_topic(clf, text)
        print(f"\n  Texto: {text[:70]}...")
        print(f"  → {label}  (score: {score})")