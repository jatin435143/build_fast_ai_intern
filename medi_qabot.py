import streamlit as st
import os

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_huggingface import HuggingFaceEndpoint   # type: ignore
from langchain_core.prompts import PromptTemplate   
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from typing import Optional, List
from typing import ClassVar

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY"))

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


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):  # modify the general prompt fro each ques 
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt



def main():
    st.title("Ask chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})

        CUSTOM_PROMPT_TEMPLATE = """ Use the pieces of information provided in the context to answer user's question.
                                     If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                                     Dont provide anything out of the given context

                                     Context: {context}
                                     Question: {question}

                                     Start the answer directly. No small talk please.  """
        
        HUGGING_FACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        llm = GeminiLLM()


        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store!")

            qa_chain=RetrievalQA.from_chain_type(
                                  llm=llm,
                                  chain_type="stuff",
                                  retriever=vectorstore.as_retriever(search_kwargs={'k':3}), # similar kitne doc return
                                  return_source_documents=True,  # meta data is sored for each chunk like page nuber we need that also to return
                                  chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                                  )
            
            response=qa_chain.invoke({'query':prompt})
            result=response['result']
            source_docs=response['source_documents']
            result_to_show=result
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'aasistant' , 'content':result_to_show})


        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()