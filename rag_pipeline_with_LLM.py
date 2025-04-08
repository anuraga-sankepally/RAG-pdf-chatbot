import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline 
from transformers.pipelines import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from huggingface_hub import login


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300
    )
    return text_splitter.split_documents(pages)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vector_store(docs, embeddings):
    if os.path.exists("pdf_index"):
        return FAISS.load_local(
            "pdf_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("pdf_index")
    return db

def get_llm():
    # Correct pipeline for FLAN-T5
    flan_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_token=512,
        do_sample = True,
        temperature=0.3
    )
    return HuggingFacePipeline(pipeline=flan_pipeline)

def ask_question(db, question, use_llm=False):
    if use_llm:
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",  # Changed back to stuff for simplicity
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True  # Added to see sources
        )
        result = qa.invoke({"query": question})
        return f"{result['result']}\n\nSources:\n" + "\n".join(
            f"📄 Page {doc.metadata['page']+1}: {doc.page_content[:200]}..." 
            for doc in result['source_documents']
        )
    else:
        docs = db.similarity_search(question, k=3)
        return "\n\n".join([f"📄 Page {doc.metadata['page']+1}:\n{doc.page_content[:500]}..." for doc in docs])

if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your key
    
    pdf_path = "data/test_paper_2.pdf"
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