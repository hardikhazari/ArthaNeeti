from databricks_langchain import ChatDatabricks
from langchain_core.prompts import PromptTemplate
from src.llm.prompts import VALIDATOR_PROMPT

class HallucinationValidator:
    def __init__(self, endpoint: str):
        self.llm = ChatDatabricks(endpoint=endpoint)
        self.prompt = PromptTemplate.from_template(VALIDATOR_PROMPT)

    def validate_response(self, question: str, context: str, response: str) -> dict:
        """
        Validates if the generated response is grounded in the context.
        """
        print("[*] Validating response groundedness...")
        chain = self.prompt | self.llm
        result = chain.invoke({
            "context": context,
            "response": response
        }).content
        
        is_grounded = "YES" in result.upper()
        
        return {
            "is_grounded": is_grounded,
            "feedback": result
        }

if __name__ == "__main__":
    pass
