from src.llm.router import QueryRouter
from src.llm.generator import ResponseGenerator
from src.retrieval.search_engine import SearchEngine
from src.retrieval.reranker import MosaicReranker
from src.guardrails.validator import HallucinationValidator
import yaml

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

    def run(self, query: str) -> dict:
        """
        Main entry point for query processing.
        Decides between SIMPLE and COMPLEX paths.
        """
        # Step 1: Route
        category = self.router.route_query(query)
        
        if category == "SIMPLE":
            answer = self.generator.generate_simple_answer(query)
            return {
                "answer": answer,
                "category": "SIMPLE",
                "docs": []
            }
        
        # Step 2: Retrieve
        docs = self.retriever.retrieve(query)
        
        # Step 3: Rerank
        reranked_docs = self.reranker.rerank(query, docs)
        
        # Step 4: Generate
        context = "\n\n".join([d.page_content for d in reranked_docs])
        answer = self.generator.generate_grounded_answer(query, context)
        
        # Step 5: Validate
        validation = self.validator.validate_response(query, context, answer)
        
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
    # print(result['answer'])
