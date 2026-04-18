from pyspark.sql import SparkSession
from pyspark.sql.functions import col

class CDFOrchestrator:
    def __init__(self, catalog: str, schema: str, silver_table: str):
        self.spark = SparkSession.getActiveInstance()
        self.silver_fqn = f"{catalog}.{schema}.{silver_table}"

    def get_incremental_changes(self, start_version: int):
        """
        Read incremental changes from Silver table using Change Data Feed.
        """
        print(f"[*] Reading CDF from version {start_version} for {self.silver_fqn}...")
        
        changes_df = (self.spark.read.format("delta")
                      .option("readChangeData", "true")
                      .option("startingVersion", start_version)
                      .table(self.silver_fqn))
        
        # Filter for only inserts and updates
        incremental_df = changes_df.filter(col("_change_type").isin(["insert", "update_postimage"]))
        
        return incremental_df

if __name__ == "__main__":
    import yaml
    with open("configs/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cdf = CDFOrchestrator(
        catalog=cfg['tables']['catalog'],
        schema=cfg['tables']['schema'],
        silver_table=cfg['tables']['silver_table']
    )
    # Example: print last 5 changes if any
    try:
        changes = cdf.get_incremental_changes(start_version=0)
        changes.show(5)
    except Exception as e:
        print(f"[!] No CDF access or version invalid: {e}")
