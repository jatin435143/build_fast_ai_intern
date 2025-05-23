import os
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from typing import Optional, List
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import ClassVar

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))  

# this class helps gemini to get used with langchain

class GeminiLLM(LLM):
    model_name: ClassVar[str] = "models/gemini-1.5-flash"
    temperature: ClassVar[float] = 0.5

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = prompt[:5000]
        model = genai.GenerativeModel(model_name=self.model_name)
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "gemini"


CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, say you don't know. Do not make up an answer.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please."""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"  # when needed can load model from here
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# RetrievalQA chain with Gemini
llm = GeminiLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Query and response
user_query = input("Write Query Here: ")[:1000]  
response = qa_chain.invoke({"query": user_query})

print("RESULT:", response["result"])
print("SOURCE_DOCS:",response["source_documents"])

