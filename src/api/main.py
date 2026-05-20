"""
API REST para el proyecto podcast-nlp-es.

Expone los modelos NLP (clasificador de temas y topics BERTopic)
como endpoints HTTP. Cualquier aplicación puede consultarlos.

Endpoints:
    GET  /health    estado de la API y modelos cargados
    POST /classify  clasifica un texto en un tema
    GET  /topics    lista los temas descubiertos por BERTopic
    GET  /episodes  lista episodios con filtros opcionales

Cómo lanzar el servidor: uvicorn src.api.main:app --reload

La documentación está en:
    http://127.0.0.1:8000/docs 
    http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Asegura que el directorio raíz del proyecto esté en sys.path cuando se
# ejecuta el módulo directamente con `python src/api/main.py`
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.fine_tune import load_finetuned_classifier, predict_topic
from src.preprocessing.normalize_description import clean_description

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas de datos y modelos
# ---------------------------------------------------------------------------

MODEL_DIR = Path("models/topic_classifier")
EPISODES_PATH = Path("data/processed/episodes_with_topics.csv")


# ---------------------------------------------------------------------------
# Estado global de la aplicación
# ---------------------------------------------------------------------------
# Guardamos el modelo y los datos aquí para no recargarlos en cada request

app_state: dict = {
    "classifier": None,       # modelo fine-tuneado
    "episodes_df": None,      # DataFrame con episodios y sus temas
    "model_loaded": False,
    "data_loaded": False,
}


# ---------------------------------------------------------------------------
# Lifespan: qué cargar al arrancar y qué limpiar al apagar
# ---------------------------------------------------------------------------
# El lifespan es la forma moderna de FastAPI de hacer "cosas al inicio"
# Es lo que se ejecuta antes de que la API empiece a aceptar requests

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelos y datos al arrancar. Se ejecuta una sola vez."""

    log.info("Arrancando podcast-nlp-es API...")

    # Cargar modelo fine-tuneado
    if MODEL_DIR.exists():
        try:
            app_state["classifier"] = load_finetuned_classifier(MODEL_DIR)
            app_state["model_loaded"] = True
            log.info(f"Modelo cargado desde {MODEL_DIR}")
        except Exception as e:
            log.warning(f"No se pudo cargar el modelo: {e}")
            log.warning("El endpoint /classify no estará disponible.")
    else:
        log.warning(
            f"No se encuentra el modelo en {MODEL_DIR}. "
            "Ejecuta primero: python src/analysis/fine_tune.py"
        )

    # Cargar dataset de episodios
    if EPISODES_PATH.exists():
        try:
            app_state["episodes_df"] = pd.read_csv(EPISODES_PATH)
            app_state["data_loaded"] = True
            log.info(f"Dataset cargado: {len(app_state['episodes_df'])} episodios")
        except Exception as e:
            log.warning(f"No se pudo cargar el dataset: {e}")
    else:
        log.warning(
            f"No se encuentra el dataset en {EPISODES_PATH}. "
            "Ejecuta primero: python src/analysis/topics.py"
        )

    yield  # aquí FastAPI empieza a servir requests

    # Código de limpieza al apagar (opcional, pero buena práctica)
    log.info("Apagando API...")
    app_state["classifier"] = None
    app_state["episodes_df"] = None


# ---------------------------------------------------------------------------
# Creación de la app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Podcast NLP ES",
    description=(
        "API para análisis de podcasts feministas españoles. "
        "Clasifica episodios por tema y lista los temas descubiertos "
        "mediante topic modeling con BERTopic."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Modelos Pydantic — definen la forma del JSON de entrada y salida
# ---------------------------------------------------------------------------
# Pydantic valida automáticamente que los datos tengan el tipo correcto
# Si envías un número donde se espera un string, FastAPI devuelve un
# error 422 con un mensaje

class ClassifyRequest(BaseModel):
    """JSON que debe enviar el cliente al llamar a POST /classify."""

    text: str = Field(
        ...,                          # ... significa "campo obligatorio"
        min_length=10,
        max_length=5000,
        description="Texto a clasificar. Mínimo 10 caracteres.",
        examples=["Hablamos sobre la baja de maternidad y la conciliación laboral con hijos pequeños."],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hablamos sobre la baja de maternidad y la conciliación laboral."
            }
        }
    )


class ClassifyResponse(BaseModel):
    """JSON que devuelve la API tras clasificar un texto."""

    topic: str = Field(description="Tema detectado.")
    score: float = Field(description="Confianza del modelo (0.0 a 1.0).")
    text_cleaned: str = Field(description="Texto tras el preprocessing.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic": "maternidad, crianza y conciliación laboral",
                "score": 0.91,
                "text_cleaned": "Hablamos sobre la baja de maternidad y la conciliación laboral.",
            }
        }
    )


class TopicInfo(BaseModel):
    """Información de un tema descubierto por BERTopic"""

    id: int = Field(description="ID numérico del tema (-1 = sin tema).")
    label: str = Field(description="Etiqueta del tema (palabras clave).")
    episode_count: int = Field(description="Número de episodios en este tema.")


class EpisodeInfo(BaseModel):
    """Información de un episodio"""

    episode_name: str
    show_name: str
    bertopic_label: Optional[str] = None
    bertopic_id: Optional[int] = None
    bertopic_prob: Optional[float] = None
    release_date: Optional[str] = None
    duration_min: Optional[float] = None
    spotify_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Estado de la API"""

    status: str
    model_loaded: bool
    data_loaded: bool
    episode_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado de la API",
    tags=["Sistema"],
)
def health() -> HealthResponse:
    """
    Comprueba que la API está funcionando y qué recursos tiene cargados

    Útil para monitorización y para verificar que el modelo y los datos
    se cargaron correctamente al arrancar
    """
    episode_count = (
        len(app_state["episodes_df"])
        if app_state["data_loaded"]
        else None
    )

    return HealthResponse(
        status="ok",
        model_loaded=app_state["model_loaded"],
        data_loaded=app_state["data_loaded"],
        episode_count=episode_count,
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Clasifica un texto en un tema",
    tags=["Análisis"],
)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Recibe un texto (descripción de episodio, párrafo, etc) y devuelve
    el tema al que pertenece según el clasificador fine-tuneado

    El texto se limpia automáticamente antes de clasificar (mismo
    preprocessing que se usó para entrenar el modelo).

    **Ejemplo de uso:**
    ```json
    {"text": "Hablamos sobre la brecha salarial entre hombres y mujeres"}
    ```
    """
    # Verificar que el modelo está disponible
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo clasificador no está cargado. "
                "Asegúrate de haber ejecutado fine_tune.py antes de lanzar la API."
            ),
        )

    # Limpiar el texto (mismo pipeline que en entrenamiento)
    text_clean = clean_description(request.text)

    if not text_clean:
        raise HTTPException(
            status_code=422,
            detail="El texto quedó vacío tras el preprocessing. Envía un texto con más contenido.",
        )

    # Clasificar con el modelo fine-tuneado
    topic, score = predict_topic(app_state["classifier"], text_clean)

    return ClassifyResponse(
        topic=topic,
        score=score,
        text_cleaned=text_clean,
    )


@app.get(
    "/topics",
    response_model=list[TopicInfo],
    summary="Lista los temas descubiertos por BERTopic",
    tags=["Análisis"],
)
def get_topics(include_noise: bool = Query(False, description="Incluir el grupo sin tema (id=-1)")) -> list[TopicInfo]:
    """
    Devuelve todos los temas descubiertos automáticamente por BERTopic,
    ordenados por número de episodios de mayor a menor

    Estos temas son distintos de las categorías del clasificador:
    BERTopic los descubrió sin que nadie le dijera qué buscar
    """
    if not app_state["data_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="El dataset no está cargado. Ejecuta topics.py antes de lanzar la API.",
        )

    df = app_state["episodes_df"]

    # Verificar que el dataset tiene las columnas de BERTopic
    if "bertopic_id" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="El dataset no tiene columnas de BERTopic. Ejecuta topics.py primero.",
        )

    # Agrupar por tema y contar episodios
    topic_counts = (
        df.groupby(["bertopic_id", "bertopic_label"])
        .size()
        .reset_index(name="episode_count")
        .sort_values("episode_count", ascending=False)
    )

    topics = []
    for _, row in topic_counts.iterrows():
        topic_id = int(row["bertopic_id"])
        if topic_id == -1 and not include_noise:
            continue
        topics.append(TopicInfo(
            id=topic_id,
            label=str(row["bertopic_label"]),
            episode_count=int(row["episode_count"]),
        ))

    return topics


@app.get(
    "/episodes",
    response_model=list[EpisodeInfo],
    summary="Lista episodios con filtros opcionales",
    tags=["Datos"],
)
def get_episodes(
    show: Optional[str] = Query(None, description="Filtrar por nombre del podcast (parcial)."),
    topic_id: Optional[int] = Query(None, description="Filtrar por ID de tema BERTopic."),
    topic_label: Optional[str] = Query(None, description="Filtrar por texto en la etiqueta del tema."),
    limit: int = Query(20, ge=1, le=100, description="Máximo de episodios a devolver (1-100)."),
) -> list[EpisodeInfo]:
    """
    Devuelve una lista de episodios con su información y tema asignado

    Los filtros son opcionales y acumulables:
    - `/episodes`                           primeros 20 episodios
    - `/episodes?show=radiojaputa`          episodios de Radiojaputa
    - `/episodes?topic_id=2`                episodios del tema 2
    - `/episodes?topic_label=maternidad`    episodios cuya etiqueta contiene "maternidad"
    - `/episodes?show=malasmadres&limit=5`  5 episodios de Malasmadres
    """
    if not app_state["data_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="El dataset no está cargado.",
        )

    df = app_state["episodes_df"].copy()

    # Aplicar filtros
    if show:
        df = df[df["show_name"].str.contains(show, case=False, na=False)]

    if topic_id is not None:
        if "bertopic_id" not in df.columns:
            raise HTTPException(status_code=500, detail="El dataset no tiene columnas de BERTopic.")
        df = df[df["bertopic_id"] == topic_id]

    if topic_label:
        if "bertopic_label" not in df.columns:
            raise HTTPException(status_code=500, detail="El dataset no tiene columnas de BERTopic.")
        df = df[df["bertopic_label"].str.contains(topic_label, case=False, na=False)]

    if df.empty:
        return []

    # Devolver los primeros "limit" episodios
    df = df.head(limit)

    # Convertir a lista de EpisodeInfo
    # Usamos .get() para columnas opcionales que pueden no existir
    episodes = []
    for _, row in df.iterrows():
        episodes.append(EpisodeInfo(
            episode_name=str(row.get("episode_name", "")),
            show_name=str(row.get("show_name", "")),
            bertopic_label=str(row["bertopic_label"]) if "bertopic_label" in df.columns else None,
            bertopic_id=int(row["bertopic_id"]) if "bertopic_id" in df.columns else None,
            bertopic_prob=float(row["bertopic_prob"]) if "bertopic_prob" in df.columns else None,
            release_date=str(row["release_date"]) if "release_date" in df.columns else None,
            duration_min=float(row["duration_min"]) if "duration_min" in df.columns else None,
            spotify_url=str(row["spotify_url"]) if "spotify_url" in df.columns else None,
        ))

    return episodes