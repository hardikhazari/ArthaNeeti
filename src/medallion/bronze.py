import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name

class BronzeLayer:
    def __init__(self, catalog: str, schema: str, table_name: str, raw_volume: str):
        self.spark = SparkSession.getActiveInstance()
        self.full_table_name = f"{catalog}.{schema}.{table_name}"
        self.raw_volume = raw_volume

    def ingest_raw_files(self):
        """
        Ingest raw binary files from Volume into Bronze Delta Table.
        """
        print(f"[*] Ingesting from {self.raw_volume} to {self.full_table_name}...")
        
        # Binary data ingestion (PDFs etc)
        df = (self.spark.read.format("binaryFile")
              .option("pathGlobFilter", "*.pdf")
              .load(self.raw_volume)
              .withColumn("ingested_at", current_timestamp())
              .withColumn("source_path", input_file_name()))

        # Write to Delta
        (df.write.format("delta")
         .mode("overwrite")  # In demo/hackathon we often overwrite, or append with CDF
         .option("overwriteSchema", "true")
         .saveAsTable(self.full_table_name))

        # Enable Change Data Feed
        self.spark.sql(f"ALTER TABLE {self.full_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        
        print(f"[+] Bronze ingestion complete. Table: {self.full_table_name}")

if __name__ == "__main__":
    # Example usage for testing
    import yaml
    with open("configs/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    bronze = BronzeLayer(
        catalog=cfg['tables']['catalog'],
        schema=cfg['tables']['schema'],
        table_name=cfg['tables']['bronze_table'],
        raw_volume=cfg['paths']['raw_volume']
    )
    bronze.ingest_raw_files()
