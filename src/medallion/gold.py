from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession

class GoldLayer:
    def __init__(self, catalog: str, schema: str, silver_table: str, index_name: str, endpoint: str):
        self.spark = SparkSession.getActiveInstance()
        self.silver_fqn = f"{catalog}.{schema}.{silver_table}"
        self.index_fqn = f"{catalog}.{schema}.{index_name}"
        self.endpoint = endpoint
        self.vsc = VectorSearchClient()

    def sync_index(self):
        """
        Synchronize the Silver Delta table with the Databricks Vector Search index.
        """
        print(f"[*] Syncing Vector Index {self.index_fqn} from {self.silver_fqn}...")
        
        try:
            # Check if index exists, else create it
            index = self.vsc.get_index(endpoint_name=self.endpoint, index_name=self.index_fqn)
            index.sync()
            print(f"[+] Index sync triggered for {self.index_fqn}")
        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                print(f"[!] Index {self.index_fqn} does not exist. Creating...")
                # In production, you'd specify primary_key and embedding parameters
                # For now, we assume it's created via the UI as per reproducibility guide
                print("[!] Please ensure index is created in the Databricks UI.")
            else:
                raise e

if __name__ == "__main__":
    import yaml
    with open("configs/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    gold = GoldLayer(
        catalog=cfg['tables']['catalog'],
        schema=cfg['tables']['schema'],
        silver_table=cfg['tables']['silver_table'],
        index_name=cfg['tables']['gold_index'],
        endpoint=cfg['databricks']['vector_search_endpoint']
    )
    gold.sync_index()
