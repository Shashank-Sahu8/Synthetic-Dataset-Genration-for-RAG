"""
Streamlit frontend for the Synthetic Dataset Generation SaaS platform.

Pages
─────
  1. 🚀 Create Project      – register a new project, receive your API key
  2. 📊 View Dataset         – monitor batch progress + inspect QA entries
  3. 🔌 SDK Integration      – ready-to-paste snippets for your RAG pipeline

Documents are ingested ONLY via the Python SDK — not through this UI.

Run with:
    streamlit run frontend/app.py
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st

# Make project root importable when running from the frontend/ directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdk.client import SyntheticDatasetClient, SDKError, create_project

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synthetic Dataset Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – API server URL (shared across all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/static/img/langgraph.png",
    width=140,
)
st.sidebar.title("Synthetic Dataset Generator")
st.sidebar.markdown("*Powered by LangGraph + RAGAS*")
st.sidebar.divider()

api_base = st.sidebar.text_input(
    "🔌 Backend URL",
    value="http://localhost:8000",
    help="URL of the FastAPI backend",
)

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigation",
    ["🚀 Create Project", "� View Dataset", "🔌 SDK Integration"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: persistent API key stored in session_state
# ─────────────────────────────────────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "project_id" not in st.session_state:
    st.session_state["project_id"] = ""
if "project_name" not in st.session_state:
    st.session_state["project_name"] = ""


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Create Project
# ─────────────────────────────────────────────────────────────────────────────
if page == "🚀 Create Project":
    st.title("🚀 Create a New Project")
    st.markdown(
        """
        Every project gets a unique **API Key**.  
        You will use this key with the **Python SDK** to upload documents and fetch your dataset.
        """
    )

    with st.form("create_project_form"):
        project_name = st.text_input(
            "Project Name",
            placeholder="e.g. My RAG Benchmark – Banking Docs",
            max_chars=255,
        )
        submitted = st.form_submit_button("Create Project", type="primary")

    if submitted:
        if not project_name.strip():
            st.error("Project name cannot be empty.")
        else:
            with st.spinner("Creating project..."):
                try:
                    result = create_project(name=project_name.strip(), base_url=api_base)
                    st.session_state["api_key"] = result["api_key"]
                    st.session_state["project_id"] = result["project_id"]
                    st.session_state["project_name"] = result["name"]
                    st.success("✅ Project created successfully!")
                except SDKError as e:
                    st.error(f"Failed to create project: {e}")

    # Always show the stored project info (survives page switches)
    if st.session_state["api_key"]:
        st.divider()
        col1, col2 = st.columns([1, 2])
        col1.metric("Project Name", st.session_state["project_name"])
        col1.metric("Project ID", st.session_state["project_id"][:8] + "…")

        col2.markdown("### 🔑 Your API Key")
        col2.code(st.session_state["api_key"], language="text")
        col2.warning(
            "⚠️ Save this key securely. It will not be shown again after you navigate away."
        )

        st.info(
            "➡️ **Next step:** Go to **� SDK Integration** to see how to pass documents "
            "from your RAG pipeline, or switch to **📊 View Dataset** to monitor progress."
        )

    elif not submitted:
        # Show if the user manually pastes a key from a previous session
        st.divider()
        st.subheader("Already have a project? Restore your session")
        with st.form("restore_form"):
            restored_key = st.text_input("Paste your existing API Key here", type="password")
            restore_btn = st.form_submit_button("Restore")
        if restore_btn and restored_key.strip():
            # Validate the key by hitting the health endpoint
            try:
                sdk = SyntheticDatasetClient(
                    api_key=restored_key.strip(), base_url=api_base
                )
                # A lightweight check: fetch dataset (empty is fine, 401 means bad key)
                sdk.get_dataset()
                st.session_state["api_key"] = restored_key.strip()
                st.success("✅ API Key validated. Session restored.")
                st.rerun()
            except SDKError as e:
                st.error(f"Could not validate key: {e}")





# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – View Dataset
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 View Dataset":
    st.title("📊 View Dataset")

    if not st.session_state["api_key"]:
        st.warning("⚠️ No API key found. Please **Create a Project** first.")
        st.stop()

    sdk = SyntheticDatasetClient(
        api_key=st.session_state["api_key"], base_url=api_base
    )

    # ── Batch Status ──────────────────────────────────────────────────────────
    st.subheader("Phase-1 Batch Formation Status")

    default_doc_id = st.session_state.get("last_doc_id", "")
    doc_id_check = st.text_input(
        "doc_id to check",
        value=default_doc_id,
        placeholder="e.g. DOC-A",
    )

    col_refresh, col_auto = st.columns([1, 3])
    refresh_clicked = col_refresh.button("🔄 Refresh")
    auto_refresh = col_auto.checkbox("Auto-refresh every 10 s")

    if doc_id_check and (refresh_clicked or auto_refresh):
        try:
            status = sdk.get_batch_status(doc_id_check)
            total = status.get("total_batches", 0)
            st.metric("Batches created", total)

            if total > 0:
                batch_rows = status["batches"]
                for b in batch_rows:
                    ctx_icon = "✅" if b["has_context"] else "⏳"
                    st.markdown(
                        f"{ctx_icon} **Batch {b['batch_index']}**  —  id: `{b['batch_id'][:8]}…`  "
                        f"created: {b['created_at']}"
                    )
            else:
                st.info("No batches yet. The pipeline may still be starting.")
        except SDKError as e:
            st.error(f"Could not fetch status: {e}")

        if auto_refresh:
            time.sleep(10)
            st.rerun()

    st.divider()

    # ── QA Dataset ───────────────────────────────────────────────────────────
    st.subheader("Generated QA Dataset")

    show_faulty = st.checkbox("Include faulty entries (accuracy < 80%)", value=False)

    if st.button("📥 Fetch Dataset"):
        try:
            with st.spinner("Fetching…"):
                data = sdk.get_dataset(include_faulty=show_faulty)
            total = data.get("total", 0)
            entries = data.get("entries", [])

            st.metric("Total QA pairs", total)

            if total == 0:
                st.info(
                    "No entries yet. The pipeline is either still running (Phase-1 "
                    "generates batches/contexts; QA generation is Phase-2)."
                )
            else:
                for i, entry in enumerate(entries, 1):
                    with st.expander(f"#{i}  {entry['question'][:80]}…"):
                        st.markdown(f"**Question:** {entry['question']}")
                        st.markdown(f"**Answer:** {entry['answer']}")
                        st.markdown(f"**Source pages:** {entry['source_page_numbers']}")
                        st.markdown(f"**Accuracy:** {entry.get('overall_accuracy')}")
                        st.markdown(f"**Faulty:** {'Yes ❌' if entry['is_faulty'] else 'No ✅'}")
                        with st.expander("Source context"):
                            st.text(entry["source_context"])

                # Download button
                st.download_button(
                    label="⬇️ Download as JSON",
                    data=json.dumps(entries, indent=2, default=str),
                    file_name="synthetic_dataset.json",
                    mime="application/json",
                )
        except SDKError as e:
            st.error(f"Failed to fetch dataset: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – SDK Integration Guide
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔌 SDK Integration":
    st.title("🔌 SDK Integration Guide")

    api_key_val = st.session_state["api_key"] or "sdk-YOUR_API_KEY_HERE"

    st.info(
        "**Documents are ingested only via the Python SDK** — not through this UI."
    )

    # ── Install ───────────────────────────────────────────────────────────────
    st.subheader("1. Install")
    st.code("pip install requests", language="bash")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("2. Upload pages")
    st.markdown(
        "Pass a plain list of page dicts — each with `page_no`, `doc_id`, and `text`."
    )
    st.code(
        f"""import json
from sdk import SyntheticDatasetClient

sdk = SyntheticDatasetClient(
    api_key="{api_key_val}",
    base_url="{api_base}",
)

with open("Dummy-Data/docAData.json") as f:
    pages = json.load(f)
    # pages = [{{"page_no": 1, "doc_id": "DOC-A", "text": "..."}}, ...]

sdk.upload(pages)
""",
        language="python",
    )

    st.divider()

    # ── Poll + fetch ──────────────────────────────────────────────────────────
    st.subheader("3. Poll status & fetch the dataset")
    st.code(
        f"""import time

# Poll until pipeline completes
while True:
    status = sdk.get_batch_status(doc_id="DOC-A")
    print(f"Batches created: {{status['total_batches']}}")
    if status["total_batches"] > 0 and all(b["has_context"] for b in status["batches"]):
        break
    time.sleep(15)

# Fetch validated QA pairs
dataset = sdk.get_dataset()
print(f"Total QA pairs: {{dataset['total']}}")

for entry in dataset["entries"]:
    print(entry["question"])
    print(entry["answer"])
    print(f"Pages: {{entry['source_page_numbers']}}  Accuracy: {{entry['overall_accuracy']}}")
    print()
""",
        language="python",
    )
