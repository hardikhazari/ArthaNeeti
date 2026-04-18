from databricks_langchain import DatabricksVectorSearch
from pyspark.sql import SparkSession

class SearchEngine:
    def __init__(self, index_name: str, search_columns: list):
        self.vector_store = DatabricksVectorSearch(
            index_name=index_name,
            columns=search_columns
        )

    def retrieve(self, query: str, k: int = 15, query_type: str = "HYBRID", filters: dict = None):
        """
        Hybrid retrieval (Dense + BM25) with optional metadata filtering.
        """
        print(f"[*] Retrieving top-{k} docs for: {repr(query)} (Filters: {filters})...")
        
        search_kwargs = {
            "k": k,
            "query_type": query_type
        }
        
        # Add metadata filters if provided
        if filters:
            search_kwargs["filters"] = filters

        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        return retriever.invoke(query)

if __name__ == "__main__":
    import yaml
    with open("configs/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    engine = SearchEngine(
        index_name=f"{cfg['tables']['catalog']}.{cfg['tables']['schema']}.{cfg['tables']['gold_index']}",
        search_columns=["source_file", "section_id", "heading_chain", "raw_text"]
    )
    # Example filtered retrieve:
    # docs = engine.retrieve("safety rules", filters={"source_file": "fssai-1.pdf"})
