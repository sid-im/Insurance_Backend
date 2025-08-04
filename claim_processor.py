import os
import json
import logging
import os
from dotenv import load_dotenv
from groq import Groq 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
load_dotenv()
# Get the API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Prompt Template
CLAIM_DECISION_PROMPT = PromptTemplate(
    input_variables=["query", "clauses"],
    template="""
You are an insurance claims decision assistant. Given a claim query and relevant policy clauses, output a JSON object like:
{{
  "approval": true | false,
  "reasons": "...reasoning based on clauses..."
}}

Query: {query}
Relevant Policy Clauses:
{clauses}

Respond only with the JSON and brief justification.
""",
)

class ClaimProcessor:
    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = self._load_vectorstore()
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
        self.chain = LLMChain(llm=self.llm, prompt=CLAIM_DECISION_PROMPT)

    def _load_vectorstore(self):
        loader = PyPDFLoader(self.policy_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = splitter.split_documents(pages)
        return FAISS.from_documents(docs, self.embeddings)

    def process_claim(self, query: str) -> dict:
        context_docs = self.vectorstore.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in context_docs])
        response = self.chain.run(query=query, clauses=context)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "approval": None,
                "reasons": "Failed to parse LLM output",
                "raw": response
            }
        return result
