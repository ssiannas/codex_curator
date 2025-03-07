import os
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodexCurator:
    template = """
    <s> [INST]
    You are an assistant for worldbuilding tasks. Use the following context elements to answer questions regarding worldbuilding, session planning, campaign management, creative writing, NPC and character creation, and more. If you do not know the answer, simply say so. Keep your answers relatively brief and concise, with mostly two to three paragraphs. We can expand on those if needed. [/INST] </s>
    
    [INST] Question: {question}
    Context: {context}
    Answer: [/INST]
    """

    def __init__(self, llm_model: str = "deepseek-r1:32b", embedding_model: str = "mxbai-embed-large"):
        self.model_name = llm_model
        self.model = ChatOllama(model=self.model_name)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            template=self.template
        )
        self.vector_store = None
        self.retriever = None
        self.load_lib()

    def load_document(self, pdf_file_path: str):
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        return chunks

    def ingest(self, chunks):
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info(
            "Ingestion completed. Document embeddings stored successfully.")

    def load_and_ingest(self, single_file_path: str = None):
        chunks = self.load_document(single_file_path)
        self.ingest(chunks)
        
    def load_lib(self, pdf_file_path: str = "knowledge_base"):
        files = [f for f in os.listdir(pdf_file_path) if f.endswith(".pdf")]
        for file in files:
            print(f"Loading {file}")
            chunks = self.load_and_ingest(f"{pdf_file_path}/{file}")
        print("Knowledge base loaded.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError(
                "No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None


if __name__ == "__main__":
    cc = CodexCurator()
    cc.load_lib()
    print(cc.ask("What is the meaning of life?"))