import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "movie_knowledge_interstellar.pdf"

DB_DIRECTORY = "./movie_chroma_db"

# Change this if ollama list shows a different model
MODEL_NAME = "gemma3:4b"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Movie Search",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        🎬 AI Movie Search Application
    </h1>

    <p style="text-align:center;">
        PDF RAG + ChromaDB + Ollama AI
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

@st.cache_resource
def create_vector_database():

    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(
            f"{PDF_FILE} not found."
        )

    # Load PDF
    loader = PyPDFLoader(PDF_FILE)

    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    # Embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIRECTORY,
        collection_name="movie_knowledge"
    )

    return vectorstore


# ============================================================
# RAG SEARCH
# ============================================================

def search_rag(query):

    vectorstore = create_vector_database()

    documents = vectorstore.similarity_search(
        query,
        k=5
    )

    if not documents:
        return "No information found."

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    return context


# ============================================================
# AI FUNCTION
# ============================================================

def generate_ai_answer(
    movie_name,
    context
):

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.4
    )

    prompt = f"""
You are an AI Movie Assistant.

The user searched for:

{movie_name}

Use the following information retrieved from
the movie knowledge PDF.

========================
RAG CONTEXT
========================

{context}

========================
TASK
========================

Give a clear and student-friendly answer.

Include:

🎬 Movie Overview

👨 Hero / Lead Actor

👩 Heroine / Lead Actress

🎭 Main Cast

🎥 Crew

📅 Release Information

⭐ Rating if available

📝 Story Summary

🎭 Genre

💡 Interesting Information

🍿 Recommendation

Important:

- Use only the information available in the RAG context.
- Do not invent facts.
- If some information is not available,
  say "Information not available in the PDF."
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎬 AI Movie Search")

    st.write("This application uses:")

    st.write("📄 Movie PDF")

    st.write("✂️ Text Splitting")

    st.write("🧠 Ollama Embeddings")

    st.write("🗄️ ChromaDB")

    st.write("📚 RAG")

    st.write("🤖 Ollama AI")

    st.divider()

    st.subheader("🔄 Pipeline")

    st.write(
        """
        Movie Search
        ↓
        PDF
        ↓
        Text Splitting
        ↓
        Embeddings
        ↓
        ChromaDB
        ↓
        RAG
        ↓
        Ollama
        ↓
        AI Answer
        """
    )


# ============================================================
# MOVIE SEARCH
# ============================================================

movie_name = st.text_input(
    "🔎 Enter Movie Name",
    placeholder="Example: Interstellar"
)


if st.button(
    "🎬 Search Movie",
    use_container_width=True
):

    if movie_name.strip() == "":

        st.warning(
            "Please enter a movie name."
        )

    else:

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        try:

            with st.spinner(
                "📚 Searching movie knowledge base..."
            ):

                context = search_rag(
                    movie_name
                )

            # ------------------------------------------------
            # SHOW RETRIEVED INFORMATION
            # ------------------------------------------------

            st.success(
                "✅ Movie information retrieved!"
            )

            st.divider()

            st.subheader(
                "📚 Retrieved Information"
            )

            with st.expander(
                "View RAG Context"
            ):

                st.write(
                    context
                )

            # ------------------------------------------------
            # AI
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Movie Analysis"
            )

            with st.spinner(
                "🤖 Generating AI response..."
            ):

                answer = generate_ai_answer(
                    movie_name,
                    context
                )

            st.markdown(
                answer
            )

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔄 Project Pipeline"
            )

            st.code(
                """
USER
  ↓
Movie Name
  ↓
Movie PDF
  ↓
PyPDFLoader
  ↓
Text Splitting
  ↓
Ollama Embeddings
  ↓
ChromaDB
  ↓
RAG Retrieval
  ↓
Relevant Movie Information
  ↓
Ollama LLM
  ↓
AI Summary
  ↓
Recommendation
                """,
                language="text"
            )

        except Exception as e:

            st.error(
                f"❌ Error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎬 AI Movie Search | PDF RAG + ChromaDB + Ollama"
)