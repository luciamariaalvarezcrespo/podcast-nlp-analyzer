"""
Clasifica episodios de podcast usando zero-shot classification.
No necesita datos etiquetados — el modelo ya entiende español.

Cómo funciona
    Para cada episodio, el modelo se pregunta internamente:
    "¿Este texto implica que el tema es X?"
    Lo hace para cada etiqueta y devuelve la que más puntuación tenga.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm # Para mostrar una barra de progreso durante la clasificación
from transformers import pipeline

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Etiquetas — los temas que queremos detectar
# ---------------------------------------------------------------------------
# Estas etiquetas las definimos nosotras basándonos en los podcasts del dataset.
# Cuanto más descriptiva sea la etiqueta, mejor funciona el zero-shot.
# "maternidad" funciona peor que "maternidad y conciliación laboral".
# Estas etiquetas fueron elegidas manualmente por expertos en sociología

TOPIC_LABELS = [
    "violencia de género y justicia",
    "maternidad, crianza y conciliación laboral",
    "cuerpo, salud y sexualidad femenina",
    "política, activismo y movimiento feminista",
    "economía, trabajo y brecha salarial",
    "identidad de género y derechos LGTBIQ+",
    "cultura, medios de comunicación y representación",
    "humor, entretenimiento y cultura popular",
]

# Umbral mínimo de confianza para asignar una etiqueta.
# Si el modelo no está seguro (score < umbral), la etiqueta es "sin clasificar".
# 0.35 es conservador — preferimos pocos errores a muchas etiquetas incorrectas.
# Este umbral fue escogido tras revisar ejemplos en la literatura y a base de prueba y error
CONFIDENCE_THRESHOLD = 0.20

# Modelo que usamos: XLM-RoBERTa entrenado en XNLI (31 idiomas, incluido español)
# Es grande (~1.1GB) pero muy preciso. Solo se descarga una vez y queda en caché.
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_classifier():
    """
    Carga el pipeline de zero-shot classification.
    """
    log.info(f"Cargando modelo {MODEL_NAME}...")
    log.info("(La primera vez tarda ~2 min descargando. Después es instantáneo.)")

    classifier = pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME,
    )

    log.info("Modelo cargado")
    return classifier


# ---------------------------------------------------------------------------
# Clasificación de un episodio
# ---------------------------------------------------------------------------

def classify_one(
    classifier,
    text: str,
    labels: list[str] = TOPIC_LABELS,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Clasifica un texto y devuelve la etiqueta ganadora con su score.

    Args:
        classifier: Pipeline de transformers (resultado de load_classifier).
        text:       Descripción limpia del episodio.
        labels:     Lista de etiquetas candidatas.
        threshold:  Score mínimo para asignar etiqueta.

    Returns:
        Dict con:
            - predicted_topic:  etiqueta ganadora (o "sin clasificar")
            - topic_score:      confianza del modelo (0.0 a 1.0)
            - all_scores:       dict con score de cada etiqueta (para análisis)

    Example:
        >>> result = classify_one(clf, "Hablamos sobre la baja de maternidad")
        >>> result["predicted_topic"]
        'maternidad, crianza y conciliación laboral'
        >>> result["topic_score"] > 0.5
        True
    """
    if not text or len(text.strip()) < 20:
        # Texto demasiado corto para clasificar con fiabilidad
        return {
            "predicted_topic": "sin clasificar",
            "topic_score": 0.0,
            "all_scores": {},
        }

    # El modelo devuelve las etiquetas ordenadas de mayor a menor score
    result = classifier(text, candidate_labels=labels)

    best_label = result["labels"][0]
    best_score = result["scores"][0]

    # Si el modelo no está seguro, preferimos "sin clasificar" a un error
    if best_score < threshold:
        best_label = "sin clasificar"

    return {
        "predicted_topic": best_label,
        "topic_score": round(best_score, 4),
        "all_scores": dict(zip(result["labels"], [round(s, 4) for s in result["scores"]])),
    }


# ---------------------------------------------------------------------------
# Clasificación del DataFrame completo
# ---------------------------------------------------------------------------

def classify_episodes(
    df: pd.DataFrame,
    text_col: str = "clean_description",
    labels: list[str] = TOPIC_LABELS,
    threshold: float = CONFIDENCE_THRESHOLD,
    save_path: str | None = "data/processed/episodes_labeled.csv",
) -> pd.DataFrame:
    """
    Clasifica todos los episodios de un DataFrame.

    Añade tres columnas al DataFrame:
        - predicted_topic:  tema principal detectado
        - topic_score:      confianza del modelo (0-1)
        - all_scores:       scores de todas las etiquetas (JSON string)

    Args:
        df:         DataFrame con episodios. Debe tener la columna `text_col`.
        text_col:   Columna con el texto limpio (salida del preprocessing).
        labels:     Lista de etiquetas candidatas.
        threshold:  Score mínimo para asignar etiqueta.
        save_path:  Si se especifica, guarda el resultado en CSV.

    Returns:
        DataFrame con las columnas de clasificación añadidas.
    """
    classifier = load_classifier()

    results = []
    start = time.time()

    # tqdm muestra una barra de progreso porque es mucho texto
    for text in tqdm(df[text_col].fillna(""), desc="Clasificando episodios"):
        results.append(classify_one(classifier, text, labels, threshold))

    elapsed = round(time.time() - start, 1)
    log.info(f"Clasificación completada en {elapsed}s ({len(df)} episodios)")

    # Añadir resultados al DataFrame original
    df_out = df.copy()
    df_out["predicted_topic"] = [r["predicted_topic"] for r in results]
    df_out["topic_score"] = [r["topic_score"] for r in results]
    df_out["all_scores"] = [str(r["all_scores"]) for r in results]

    # Guardar
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
        log.info(f"Guardado en {save_path}")

    return df_out


def print_topic_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen de los temas detectados en consola."""
    if "predicted_topic" not in df.columns:
        print("El DataFrame no tiene columna 'predicted_topic'. Ejecuta classify_episodes primero.")
        return

    print("\n" + "=" * 65)
    print(f"{'TEMA DETECTADO':<45} {'EPISODIOS':>8}  {'SCORE MEDIO':>10}")
    print("=" * 65)

    summary = (
        df.groupby("predicted_topic")
        .agg(count=("predicted_topic", "size"), avg_score=("topic_score", "mean"))
        .sort_values("count", ascending=False)
    )

    for topic, row in summary.iterrows():
        print(f"{str(topic):<45} {row['count']:>8}  {row['avg_score']:>10.3f}")

    print("=" * 65)
    print(f"Total: {len(df)} episodios, {df['predicted_topic'].nunique()} temas\n")


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Busca el CSV limpio generado por el preprocessing
    clean_path = Path("data/episodes.csv")
    if not clean_path.exists():
        print(f"No se encuentra {clean_path}")
        sys.exit(1)

    df = pd.read_csv(clean_path)
    log.info(f"Cargados {len(df)} episodios desde {clean_path}")

    # Aplicar limpieza de descripciones si no existe la columna
    if "clean_description" not in df.columns:
        from preprocessing.normalize_description import clean_description
        log.info("Limpiando descripciones...")
        df["clean_description"] = df["description"].apply(clean_description)

    df_labeled = classify_episodes(df)
    print_topic_summary(df_labeled)

    # Mostramos algunos ejemplos para ver que funciona bien
    print("\n--- Ejemplos de clasificación ---")
    sample = df_labeled[df_labeled["predicted_topic"] != "sin clasificar"].sample(
        min(5, len(df_labeled)), random_state=42
    )
    for _, row in sample.iterrows():
        print(f"\n  Podcast: {row.get('show_name', 'N/A')}")
        print(f"  Episodio: {str(row.get('episode_name', ''))[:60]}...")
        print(f"  → Tema: {row['predicted_topic']}  (score: {row['topic_score']})")