from databricks_langchain import ChatDatabricks
from langchain_core.prompts import PromptTemplate
from src.llm.prompts import ROUTER_PROMPT

class QueryRouter:
    def __init__(self, endpoint: str):
        self.llm = ChatDatabricks(endpoint=endpoint)
        self.prompt = PromptTemplate.from_template(ROUTER_PROMPT)

    def route_query(self, question: str) -> str:
        """
        Classifies query as SIMPLE or COMPLEX.
        """
        print(f"[*] Routing query: {repr(question)}...")
        chain = self.prompt | self.llm
        response = chain.invoke({"question": question})
        
        category = response.content.strip().upper()
        if "COMPLEX" in category:
            return "COMPLEX"
        return "SIMPLE"

if __name__ == "__main__":
    # Test
    router = QueryRouter(endpoint="databricks-meta-llama-3-1-8b-instruct")
    print(router.route_query("What are the FSSAI rules?"))
    print(router.route_query("Hello there!"))
