"""
Descubre temas en episodios de podcast usando BERTopic.

A diferencia del clasificador zero-shot, aquí NO definimos
los temas de antemano. BERTopic los descubre solo mirando qué episodios
hablan de cosas parecidas.

Pipeline interno:
    texto 1. embeddings (sentence-transformers)
          2. reducción de dimensiones (UMAP)
          3. clustering (HDBSCAN)
          4. nombres de temas (c-TF-IDF)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Modelo de embeddings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Número mínimo de episodios para que un grupo sea considerado un tema
MIN_CLUSTER_SIZE = 5

# Número mínimo de veces que una palabra debe aparecer para incluirse
# en el vocabulario. Filtra palabras rarísimas que no aportan.
MIN_WORD_FREQUENCY = 2

# Palabras vacías en español que no aportan significado temático.
# BERTopic tiene su propio sistema, pero añadir estas mejora los nombres de temas.
SPANISH_STOP_WORDS = [
    "hablamos", "episodio", "programa", "semana", "hoy", "esta", "este",
    "podcast", "capítulo", "también", "así", "muy", "más", "solo",
    "aquí", "nos", "les", "las", "los", "una", "uno", "del", "que",
    "con", "por", "para", "como", "pero", "cuando", "tiene", "hay",
]

OUTPUT_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/figures")


# ---------------------------------------------------------------------------
# Preparación del texto
# ---------------------------------------------------------------------------

def prepare_texts(
    df: pd.DataFrame,
    text_col: str = "clean_description",
    min_length: int = 30,
) -> tuple[list[str], pd.DataFrame]:
    """
    Extrae y filtra los textos del DataFrame para topic modeling.

    BERTopic necesita textos con suficiente contenido para encontrar
    similitudes. Textos muy cortos (títulos, descripciones vacías)
    generan ruido en los clusters.
    """
    # Rellenar NaN con cadena vacía y filtrar por longitud
    df_valid = df.copy()
    df_valid[text_col] = df_valid[text_col].fillna("")
    df_valid = df_valid[df_valid[text_col].str.len() >= min_length].reset_index(drop=True)

    n_dropped = len(df) - len(df_valid)
    if n_dropped > 0:
        log.info(f"Filtrados {n_dropped} episodios con texto muy corto o vacío")

    texts = df_valid[text_col].tolist()
    log.info(f"Textos para topic modeling: {len(texts)}")
    return texts, df_valid


# ---------------------------------------------------------------------------
# Construcción del modelo BERTopic
# ---------------------------------------------------------------------------

def build_topic_model(
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    embedding_model_name: str = EMBEDDING_MODEL,
) -> tuple[BERTopic, SentenceTransformer]:
    """
    Construye el modelo BERTopic con sus componentes configurados.

    Devolvemos también el SentenceTransformer por separado porque
    lo necesitaremos para generar los embeddings antes de entrenar.
    """
    log.info(f"Cargando modelo de embeddings: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    # UMAP: reduce las dimensiones de los embeddings
    # n_components=5 es estándar para BERTopic
    # n_neighbors=15 controla cuánto contexto local vs global considera
    # metric="cosine" funciona mejor que euclidiana para embeddings de texto
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,     # puntos en el mismo cluster deben estar muy juntos
        metric="cosine",
        random_state=42,  # reproducibilidad
    )

    # HDBSCAN: agrupa los episodios por densidad
    # min_cluster_size: mínimo de episodios para formar un tema
    # metric="euclidean": sobre los vectores ya reducidos por UMAP
    # prediction_data=True: necesario para poder predecir nuevos textos después
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",  # más estable según papers
        prediction_data=True,
    )

    # CountVectorizer: tokeniza los textos para c-TF-IDF
    # Eliminamos stopwords en español para que los nombres de temas
    vectorizer_model = CountVectorizer(
        stop_words=SPANISH_STOP_WORDS,
        min_df=MIN_WORD_FREQUENCY,
        ngram_range=(1, 2),   # incluye bigramas: "violencia género", "brecha salarial"
    )

    # KeyBERTInspired: mejora los nombres de temas usando similitud semántica
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,       # palabras clave por tema para inspección
        verbose=True,
    )

    return topic_model, embedding_model


# ---------------------------------------------------------------------------
# Entrenamiento y predicción
# ---------------------------------------------------------------------------

def fit_topics(
    topic_model: BERTopic,
    embedding_model: SentenceTransformer,
    texts: list[str],
) -> tuple[list[int], list[float]]:
    """
    Genera embeddings y entrena BERTopic sobre los textos.
    """
    log.info("Generando embeddings... (esto tarda 1-2 minutos la primera vez)")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    log.info("Entrenando BERTopic...")
    topics, probs = topic_model.fit_transform(texts, embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_noise = topics.count(-1)
    log.info(f"Temas descubiertos: {n_topics} | Episodios sin tema (ruido): {n_noise}")

    return topics, probs


# ---------------------------------------------------------------------------
# Añadir resultados al DataFrame
# ---------------------------------------------------------------------------

def add_topics_to_dataframe(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
    probs: list[float],
) -> pd.DataFrame:
    """
    Añade los temas descubiertos al DataFrame de episodios.
    """
    # Construir mapeo legible
    # topic_model.get_topic_info() devuelve un DataFrame con columnas
    topic_info = topic_model.get_topic_info()
    id_to_label = dict(zip(topic_info["Topic"], topic_info["Name"]))

    df_out = df.copy()
    df_out["bertopic_id"] = topics
    df_out["bertopic_label"] = [id_to_label.get(t, "sin_tema") for t in topics]
    df_out["bertopic_prob"] = [round(float(p), 4) for p in probs]

    return df_out


# ---------------------------------------------------------------------------
# Resumen en consola
# ---------------------------------------------------------------------------

def print_topic_summary(topic_model: BERTopic) -> None:
    """
    Imprime los temas descubiertos con sus palabras clave y número de episodios.
    """
    info = topic_model.get_topic_info()

    print("\n" + "=" * 75)
    print(f"{'ID':>4}  {'EPISODIOS':>9}  PALABRAS CLAVE DEL TEMA")
    print("=" * 75)

    for _, row in info.iterrows():
        topic_id = row["Topic"]
        count = row["Count"]
        name = row["Name"]

        if topic_id == -1:
            print(f"\n{'—' * 75}")
            print(f"  -1  {count:>9}  [sin tema — episodios que no encajan en ningún cluster]")
            continue

        # Mostrar también las 5 palabras más representativas del tema
        top_words = topic_model.get_topic(topic_id)
        words_str = ", ".join([w for w, _ in top_words[:5]])
        print(f"{topic_id:>4}  {count:>9}  {words_str}")

    print("=" * 75)
    n_topics = len(info) - 1  # excluimos el -1
    print(f"Total: {n_topics} temas descubiertos\n")


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

def save_visualizations(
    topic_model: BERTopic,
    texts: list[str],
    embeddings,
    output_dir: Path = REPORTS_DIR,
) -> None:
    """
    Genera y guarda visualizaciones interactivas de los temas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generando visualizaciones...")

    # Barchart: palabras clave por tema
    fig_bar = topic_model.visualize_barchart(top_n_topics=10, n_words=7)
    fig_bar.write_html(str(output_dir / "topics_overview.html"))
    log.info("  → topics_overview.html")

    # Mapa 2D de episodios (requiere reducir embeddings a 2D para visualizar)
    # Usamos UMAP de 2 componentes solo para visualización
    from umap import UMAP as UMAP2D
    umap_2d = UMAP2D(n_components=2, metric="cosine", random_state=42)
    embeddings_2d = umap_2d.fit_transform(embeddings)

    fig_map = topic_model.visualize_documents(
        texts,
        embeddings=embeddings_2d,
        hide_document_hover=False,  # muestra el texto al pasar el ratón
    )
    fig_map.write_html(str(output_dir / "topics_map.html"))
    log.info("  → topics_map.html")

    # Heatmap de similitud entre temas
    try:
        fig_heat = topic_model.visualize_heatmap()
        fig_heat.write_html(str(output_dir / "topics_heatmap.html"))
        log.info("  → topics_heatmap.html")
    except Exception:
        # El heatmap falla si hay muy pocos temas; no es crítico
        log.warning("  Heatmap no generado (necesita al menos 2 temas)")

    log.info(f"Visualizaciones guardadas en {output_dir}")


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def run_topic_modeling(
    df: pd.DataFrame,
    text_col: str = "clean_description",
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    save_results: bool = True,
) -> tuple[BERTopic, pd.DataFrame]:
    """
    Pipeline completo
    """
    # Preparar textos
    texts, df_valid = prepare_texts(df, text_col)

    if len(texts) < 10:
        raise ValueError(
            f"Solo hay {len(texts)} textos válidos. "
            "Necesitas al menos 10 para topic modeling. "
            "Comprueba que el preprocessing generó texto y que "
            "ejecutaste el miner con EPISODES_PER_SHOW = 50."
        )

    # Construir modelo
    topic_model, embedding_model = build_topic_model(min_cluster_size)

    # Generar embeddings y entrenar
    # Guardamos los embeddings en una variable porque los reutilizamos para visualizaciones
    log.info("Generando embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    log.info("Entrenando BERTopic...")
    topics, probs = topic_model.fit_transform(texts, embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_noise = list(topics).count(-1)
    log.info(f"Temas descubiertos: {n_topics} | Ruido: {n_noise}")

    # Añadir resultados
    df_with_topics = add_topics_to_dataframe(df_valid, topic_model, topics, probs)

    # Guardar
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "episodes_with_topics.csv"
        df_with_topics.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"Resultados guardados en {out_path}")

        save_visualizations(topic_model, texts, embeddings)

    return topic_model, df_with_topics


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    clean_path = Path("data/processed/episodes_labeled.csv")
    if not clean_path.exists():
        print(f"No se encuentra {clean_path}")
        print("Ejecuta primero el preprocessing:")
        print("  python src/preprocessing/clean_text.py")
        sys.exit(1)

    df = pd.read_csv(clean_path)
    log.info(f"Cargados {len(df)} episodios")

    topic_model, df_topics = run_topic_modeling(df)

    print_topic_summary(topic_model)

    # Mostrar algunos ejemplos por tema
    print("\n--- Ejemplos de episodios por tema ---")
    for topic_id in sorted(df_topics["bertopic_id"].unique()):
        if topic_id == -1:
            continue
        sample = df_topics[df_topics["bertopic_id"] == topic_id].head(2)
        label = sample["bertopic_label"].iloc[0]
        print(f"\nTema {topic_id}: {label}")
        for _, row in sample.iterrows():
            print(f"  · {str(row.get('episode_name', ''))[:70]}")

    print("\n Visualizaciones interactivas en reports/figures/")
    print("   Abre topics_map.html en el navegador para ver el mapa de episodios.")