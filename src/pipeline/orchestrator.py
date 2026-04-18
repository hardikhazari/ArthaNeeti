import mlflow
import yaml
import re
from src.llm.router import QueryRouter
from src.llm.generator import ResponseGenerator
from src.retrieval.search_engine import SearchEngine
from src.retrieval.reranker import MosaicReranker
from src.guardrails.validator import HallucinationValidator
from src.llm.prompts import EXPANSION_PROMPT
from langchain_core.prompts import PromptTemplate
from databricks_langchain import ChatDatabricks

class ArthaNeetiOrchestrator:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.router = QueryRouter(endpoint=self.cfg['models']['router'])
        self.generator = ResponseGenerator(endpoint=self.cfg['models']['reasoning'])
        self.retriever = SearchEngine(
            index_name=f"{self.cfg['tables']['catalog']}.{self.cfg['tables']['schema']}.{self.cfg['tables']['gold_index']}",
            search_columns=["source_file", "section_id", "heading_chain", "raw_text"]
        )
        self.reranker = MosaicReranker(endpoint=self.cfg['models']['reranker'])
        self.validator = HallucinationValidator(endpoint=self.cfg['models']['router'])
        
        # Expansion setup
        self.expansion_llm = ChatDatabricks(endpoint=self.cfg['models']['router'])
        self.expansion_prompt = PromptTemplate.from_template(EXPANSION_PROMPT)

    def _expand_query(self, query: str) -> list:
        """
        Generate search variations for the query.
        """
        print("[*] Performing Query Expansion...")
        chain = self.expansion_prompt | self.expansion_llm
        response = chain.invoke({"question": query}).content
        variations = [v.strip() for v in response.split("\n") if v.strip()]
        return [query] + variations[:2] # Original + 2 variations

    def _stitch_xrefs(self, docs: list) -> list:
        """
        Recursive Multi-Hop Stitching: Detect references and pull them into context.
        """
        all_docs = {d.page_content: d for d in docs}
        new_docs_needed = []
        
        # Simple extraction logic: look for text like "Section 12" in metadata or content
        for doc in docs:
            xrefs = doc.metadata.get("xrefs", [])
            # Also try regex for inline references
            inline_xrefs = re.findall(r"(?:Section|Article|Chapter)\s+(\d+)", doc.page_content, re.IGNORECASE)
            new_docs_needed.extend(xrefs + inline_xrefs)
            
        if new_docs_needed:
            print(f"[*] Multi-Hop: Stitching {len(set(new_docs_needed))} additional references...")
            for ref in set(new_docs_needed):
                # Retrieve specifically by section_id metadata
                ref_docs = self.retriever.retrieve(ref, k=1, filters={"section_id": ref})
                for rd in ref_docs:
                    if rd.page_content not in all_docs:
                        all_docs[rd.page_content] = rd
        
        return list(all_docs.values())

    def run(self, query: str) -> dict:
        """
        Main entry point with MLflow tracking, Expansion, and Stitching.
        """
        mlflow.set_experiment("/Shared/ArthaNeeti_Evaluation")
        
        with mlflow.start_run(run_name=f"Query: {query[:30]}"):
            mlflow.log_param("query", query)
            
            # Step 1: Route
            category = self.router.route_query(query)
            mlflow.log_param("category", category)
            
            if category == "SIMPLE":
                answer = self.generator.generate_simple_answer(query)
                mlflow.log_metric("is_rag", 0)
                return {"answer": answer, "category": "SIMPLE", "docs": []}
            
            mlflow.log_metric("is_rag", 1)
            
            # Step 2: Expand
            search_queries = self._expand_query(query)
            mlflow.log_param("expansion_count", len(search_queries))
            
            # Step 3: Retrieve (Deduplicated)
            all_retrieved = {}
            for sq in search_queries:
                batch = self.retriever.retrieve(sq, k=10)
                for d in batch:
                    all_retrieved[d.page_content] = d
            
            # Step 4: Multi-Hop Stitching
            consolidated_docs = self._stitch_xrefs(list(all_retrieved.values()))
            mlflow.log_metric("stitched_doc_count", len(consolidated_docs))
            
            # Step 5: Rerank
            reranked_docs = self.reranker.rerank(query, consolidated_docs, top_n=5)
            
            # Step 6: Generate
            context = "\n\n".join([d.page_content for d in reranked_docs])
            answer = self.generator.generate_grounded_answer(query, context)
            
            # Step 7: Validate
            validation = self.validator.validate_response(query, context, answer)
            mlflow.log_metric("is_grounded", 1 if validation["is_grounded"] else 0)
            
            print(f"[+] Final Analysis Complete. Grounded: {validation['is_grounded']}")
            
            return {
                "answer": answer,
                "category": "COMPLEX",
                "docs": reranked_docs,
                "is_grounded": validation["is_grounded"],
                "validator_feedback": validation["feedback"]
            }

if __name__ == "__main__":
    orchestrator = ArthaNeetiOrchestrator()
    # result = orchestrator.run("What are the FSSAI rules for cloud kitchens?")
