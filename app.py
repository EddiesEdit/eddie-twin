
from pathlib import Path

import streamlit as st

from eddie_twin.factory import create_application


st.set_page_config(
    page_title="Eddie Twin",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Eddie Twin")
st.caption(
    "Your personal AI digital twin — ask questions grounded "
    "in your documents."
)


@st.cache_resource
def get_application():
    """Create one shared application instance across reruns."""
    return create_application()


try:
    app = get_application()
except Exception:
    st.error(
        "Eddie Twin could not start. Check your application "
        "configuration and deployment logs."
    )
    st.stop()


# ----------------------------
# Session state
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Sidebar: document ingestion
# ----------------------------

with st.sidebar:
    st.header("Knowledge base")
    st.write("Add a PDF to your digital twin's knowledge base.")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="PDF contents will be processed and stored in Qdrant.",
    )

    document_type = st.selectbox(
        "Document type",
        options=["cv", "project", "certificate", "research", "other"],
    )

    if st.button(
        "Ingest PDF",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        temporary_path = None

        try:
            # Streamlit's upload is held in memory. Write it to a
            # temporary file so the existing PDF ingestion pipeline
            # can process it.
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as temporary_file:
                temporary_file.write(uploaded_file.getvalue())
                temporary_path = Path(temporary_file.name)

            with st.spinner("Processing and indexing your PDF..."):
                result = app.ingestion_pipeline.ingest_file(
                    file_path=temporary_path,
                    document_type=document_type,
                )

            st.success(
                f"Processed {result['chunks_processed']} chunks "
                f"and stored {result['vectors_stored']} vectors."
            )

        except Exception:
            st.error(
                "PDF ingestion failed. Check that the file is valid "
                "and review the application logs."
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


# ----------------------------
# Chat history
# ----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    filename = source.get("source", "Unknown source")
                    page = source.get("page_number", "Unknown")
                    st.write(f"- **{filename}** — page {page}")


# ----------------------------
# Ask Eddie Twin
# ----------------------------

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your knowledge base..."):
            try:
                result = app.query_pipeline.query(question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            filename = source.get(
                                "source", "Unknown source"
                            )
                            page = source.get(
                                "page_number", "Unknown"
                            )
                            st.write(f"- **{filename}** — page {page}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except ValueError as exc:
                st.warning(str(exc))
            except Exception:
                st.error(
                    "I couldn't complete that request. "
                    "Please try again or check the application logs."
                )
