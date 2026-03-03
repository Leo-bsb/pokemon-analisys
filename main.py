from src.etl.extract import run as extract_run
from src.etl.transform import run as transform_run
from src.etl.load import run as load_run
from src.analysis.analysis import run as analysis_run
from src.models.model_comparison import run as model_comparison_run

if __name__ == "__main__":
    print("Iniciando pipeline...")
    extract_run()
    transform_run()
    load_run()
    analysis_run()
    model_comparison_run()

    print("\n" + "="*80)
    print("PIPELINE FINALIZADO COM SUCESSO - EXECUTE 'STREAMLIT RUN APP.PY'")
    print("="*80)