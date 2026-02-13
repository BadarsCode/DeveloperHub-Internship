from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_VECTOR_STORE_DIR = DEFAULT_DATA_DIR / "vector_store"
DEFAULT_STREAMLIT_VECTOR_ROOT = DEFAULT_DATA_DIR / "streamlit_vector_store"
DEFAULT_STREAMLIT_UPLOAD_ROOT = DEFAULT_DATA_DIR / "streamlit_uploads"
