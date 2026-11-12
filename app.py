from src.config import config
from src.database.connection import init_database

def main() -> None:
    """Entrypoint for the Knowledge Base application."""
    init_database(config.DB_PATH)
    try:
        import streamlit as st
        st.set_page_config(page_title="ML Knowledge Base", layout="wide")
        st.title("Intelligent Knowledge Base (ML/DL)")
        st.info("System initialized. Database ready.")
    except ImportError:
        print("Streamlit not installed in this environment. Run inside .venv with requirements installed.")

if __name__ == "__main__":
    main()

