from src.medallion.bronze import BronzeLayer
from src.medallion.silver import SilverLayer
from src.medallion.gold import GoldLayer
import yaml

def run_full_ingestion(config_path: str = "configs/settings.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    print("─── STARTING FULL INGESTION PIPELINE ───")
    
    # 1. Bronze
    bronze = BronzeLayer(
        catalog=cfg['tables']['catalog'],
        schema=cfg['tables']['schema'],
        table_name=cfg['tables']['bronze_table'],
        raw_volume=cfg['paths']['raw_volume']
    )
    bronze.ingest_raw_files()
    
    # 2. Silver
    silver = SilverLayer(
        catalog=cfg['tables']['catalog'],
        schema=cfg['tables']['schema'],
        bronze_table=cfg['tables']['bronze_table'],
        silver_table=cfg['tables']['silver_table'],
        chunk_size=cfg['processing']['chunk_size']
    )
    silver.process_silver()
    
    # 3. Gold
    gold = GoldLayer(
        catalog=cfg['tables']['catalog'],
        schema=cfg['tables']['schema'],
        silver_table=cfg['tables']['silver_table'],
        index_name=cfg['tables']['gold_index'],
        endpoint=cfg['databricks']['vector_search_endpoint']
    )
    gold.sync_index()
    
    print("─── INGESTION PIPELINE COMPLETE ───")

if __name__ == "__main__":
    run_full_ingestion()
