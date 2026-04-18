from databricks_langchain import ChatDatabricks
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.prompts import RAG_PROMPT

class ResponseGenerator:
    def __init__(self, endpoint: str):
        self.llm = ChatDatabricks(endpoint=endpoint)
        self.rag_prompt = PromptTemplate.from_template(RAG_PROMPT)

    def generate_simple_answer(self, question: str) -> str:
        """
        Direct LLM response for simple queries.
        """
        print("[*] Generating simple direct answer...")
        return self.llm.invoke(question).content

    def generate_grounded_answer(self, question: str, context: str) -> str:
        """
        Grounded generation using RAG context.
        """
        print("[*] Generating grounded RAG answer...")
        chain = self.rag_prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

if __name__ == "__main__":
    import yaml
    with open("configs/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    gen = ResponseGenerator(endpoint=cfg['models']['reasoning'])
    print(gen.generate_simple_answer("How are you?"))
