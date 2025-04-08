import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from huggingface_hub import login


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(pages)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vector_store(docs, embeddings):
    if os.path.exists("pdf_index"):
        return FAISS.load_local(
            folder_path="pdf_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True  
        )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("pdf_index")
    return db

def ask_question(db, question, use_llm=False):
    if use_llm:
        llm = HuggingFaceHub(
            # repo_id="google/flan-t5-small",
            # model_kwargs={"temperature": 0.1, "max_length": 512}
            repo_id="google/flan-t5-base",  # 220M params (3x larger than small)
            model_kwargs={"temperature":0, "max_length":1024}
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )
        return qa.invoke(question)['result']
    else:
        docs = db.similarity_search(question, k=3)
        return "\n\n".join([f"📄 Page {doc.metadata['page']+1}:\n{doc.page_content[:500]}..." for doc in docs])

if __name__ == "__main__":
    # Set your Hugging Face token (free at: https://huggingface.co/settings/tokens)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "my-api-key"
    
    pdf_path = "data/test_paper.pdf"  # Change this to your PDF path
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        exit()
    
    print("⏳ Loading and processing PDF...")
    docs = load_pdf(pdf_path)
    embeddings = get_embeddings()
    db = get_vector_store(docs, embeddings)
    
    print("✅ Ready! Ask questions about the document (type 'quit' to exit)")
    while True:
        question = input("\n❓ Question: ").strip()
        if question.lower() in ('quit', 'exit'):
            break
            
        use_llm = input("Use LLM for detailed answer? (y/n): ").lower() == 'y'
        answer = ask_question(db, question, use_llm=use_llm)
        print(f"\n🔍 Answer:\n{answer}")