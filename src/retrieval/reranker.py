import json
from databricks.sdk import WorkspaceClient

class MosaicReranker:
    def __init__(self, endpoint: str):
        self.w = WorkspaceClient()
        self.endpoint = endpoint

    def rerank(self, query: str, documents: list, top_n: int = 5):
        """
        Rerank documents using Mosaic AI Model Serving (BGE reranker).
        """
        print(f"[*] Reranking {len(documents)} docs down to {top_n}...")
        
        if not documents:
            return []

        # Prepare payload
        pairs = [[query, d.page_content] for d in documents]
        
        try:
            response = self.w.serving_endpoints.query(
                name=self.endpoint,
                input={"inputs": pairs}
            )
            
            # Extract scores (assuming the model serving response format)
            scores = response.predictions # This depends on the specific model deployment format
            
            # Sort documents by scores
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [d for d, s in scored_docs[:top_n]]
            
        except Exception as e:
            print(f"[!] Reranker failed: {e}. Falling back to original order.")
            return documents[:top_n]

if __name__ == "__main__":
    # Test would require a live endpoint
    pass
