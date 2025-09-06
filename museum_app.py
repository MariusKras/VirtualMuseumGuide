import os, json, logging, sys
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from streamlit_clickable_images import clickable_images
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from src.utils import (
    load_llm_config,
    system_prompt_route,
    system_prompt_answer,
    explain_text,
    load_gallery,
    load_item_json,
    short_meta_from_item,
    retrieve,
    build_timeline_text,
    greet_on_select,
    ask_item_or_search,
    answer_from_search,
    to_data_url,
    save_conversation_to_gcs,
)

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
LLM_CONFIG_URL = os.getenv(
    "LLM_CONFIG_URL",
    "https://storage.googleapis.com/museum-guide-config/llm_config.json",
)
LOG_BUCKET = os.getenv("LOG_BUCKET", "museum-guide-config")

with open("lang.json", "r", encoding="utf-8") as f:
    LANG = json.load(f)


def handle_exception(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        raise exc_value
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


logger = logging.getLogger("museum_app")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    fh_info = logging.FileHandler("app.log", encoding="utf-8")
    fh_info.setLevel(logging.INFO)
    fh_err = logging.FileHandler("errors.log", encoding="utf-8")
    fh_err.setLevel(logging.ERROR)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh_info.setFormatter(fmt)
    fh_err.setFormatter(fmt)
    logger.addHandler(fh_info)
    logger.addHandler(fh_err)

sys.excepthook = handle_exception

ASSETS_DIR = Path("assets/processed")
DATA_DIR = Path("data")
MAX_QUESTIONS = 20

st.set_page_config(page_title="Muziejaus pokalbių prototipas", layout="wide")


@st.cache_data()
def load_config(url: str) -> dict:
    return load_llm_config(url)


@st.cache_resource
def get_clients():
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OPENAI_API_KEY
    )
    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)
    return chat, embeddings, index


@st.cache_data
def gallery_cache():
    return load_gallery(ASSETS_DIR, DATA_DIR, logger)


LLM_CONFIG = load_config(LLM_CONFIG_URL)
FUNCTIONS = LLM_CONFIG["functions"]

chat, embeddings, index = get_clients()
GALLERY = gallery_cache()

if "view" not in st.session_state:
    st.session_state.view = "gallery"
if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None
if "selected_item_title" not in st.session_state:
    st.session_state.selected_item_title = None
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "history" not in st.session_state:
    st.session_state.history = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "lang" not in st.session_state:
    st.session_state.lang = "lt"

spacer, col_en, col_lt = st.columns([0.84, 0.08, 0.08])
with col_en:
    if st.button("EN", use_container_width=True, key="btn_en"):
        st.session_state.lang = "en"
        st.rerun()
with col_lt:
    if st.button("LT", use_container_width=True, key="btn_lt"):
        st.session_state.lang = "lt"
        st.rerun()

current_lang = LANG[st.session_state.lang]

with st.sidebar:
    if st.session_state.view == "chat":
        if st.button(current_lang["back"]):
            st.session_state.view = "gallery"
            st.rerun()
        if st.session_state.history:
            json_str = json.dumps(
                st.session_state.history, ensure_ascii=False, indent=2
            )
            txt_lines = []
            for m in st.session_state.history:
                role = (
                    "User"
                    if (m["role"] == "user" and st.session_state.lang == "en")
                    else (
                        "Vartotojas"
                        if m["role"] == "user"
                        else ("Guide" if st.session_state.lang == "en" else "Gidas")
                    )
                )
                txt_lines.append(f"{role}: {m['content']}")
            txt_str = "\n\n".join(txt_lines)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                current_lang["download_json"],
                data=json_str.encode("utf-8"),
                file_name=f"conversation_{ts}.json"
                if st.session_state.lang == "en"
                else f"pokalbis_{ts}.json",
                mime="application/json",
            )
            st.download_button(
                current_lang["download_txt"],
                data=txt_str.encode("utf-8"),
                file_name=f"conversation_{ts}.txt"
                if st.session_state.lang == "en"
                else f"pokalbis_{ts}.txt",
                mime="text/plain",
            )


def render_gallery():
    st.title(current_lang["title"])
    st.markdown(current_lang["intro"])
    rows = [GALLERY[i : i + 3] for i in range(0, len(GALLERY), 3)]
    clicked_id = None
    for row in rows:
        cols = st.columns(3, gap="small")
        for col, item in zip(cols, row):
            title_ui = (
                item["title_en"] if st.session_state.lang == "en" else item["title_lt"]
            )
            with col:
                idx = clickable_images(
                    [to_data_url(item["image"])],
                    div_style={
                        "display": "grid",
                        "justify-items": "center",
                        "align-items": "center",
                        "padding": "0",
                    },
                    img_style={
                        "width": "70%",
                        "height": "auto",
                        "cursor": "pointer",
                        "margin": "0 auto",
                        "display": "block",
                    },
                    key=f"click-{item['id']}",
                )
                st.markdown(
                    f"<div style='width:70%;margin:6px auto 0;text-align:center;font-size:1rem;font-weight:400'>{title_ui}</div>",
                    unsafe_allow_html=True,
                )
                if idx is not None and idx == 0:
                    clicked_id = item["id"]
    if clicked_id:
        item = next(x for x in GALLERY if x["id"] == clicked_id)
        title_ui = (
            item["title_en"] if st.session_state.lang == "en" else item["title_lt"]
        )
        st.session_state.selected_item_id = item["id"]
        st.session_state.selected_image = item["image"]
        logger.info(f'Selected item: "{title_ui}"')
        st.session_state.view = "chat"
        st.session_state.history = [
            {
                "role": "assistant",
                "content": greet_on_select(st.session_state.lang, title_ui),
            }
        ]
        st.rerun()


def render_chat():
    item_js = (
        load_item_json(DATA_DIR, st.session_state.selected_item_id, logger)
        if st.session_state.selected_item_id
        else {}
    )
    title_lt = item_js.get("pavadinimas") or (st.session_state.selected_item_id or "")
    title_en = item_js.get("pavadinimas_en") or title_lt
    selected_title_ui = title_en if st.session_state.lang == "en" else title_lt
    exclude_title_lt = title_lt

    st.markdown(f"### {selected_title_ui}")
    if st.session_state.selected_image:
        st.image(st.session_state.selected_image, use_container_width=False, width=300)

    st.divider()

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q_input = st.chat_input(current_lang["ask"])

    if not q_input:
        return

    user_question = q_input.strip()
    if not user_question:
        st.warning(current_lang["empty_warning"])
        logger.info("Rejected user message: empty")
        return
    if len(user_question.split()) > 200:
        st.warning(current_lang["long_warning"])
        logger.info(f"Rejected user message: length {len(user_question.split())} > 200")
        return
    if st.session_state.question_count >= MAX_QUESTIONS:
        st.warning(current_lang["limit_warning"])
        logger.info(
            f"Rate limit reached: {st.session_state.question_count} ≥ {MAX_QUESTIONS}"
        )
        return

    st.session_state.question_count += 1
    st.session_state.history.append({"role": "user", "content": user_question})
    logger.info(f"User: {user_question}")

    with st.chat_message("user"):
        st.markdown(user_question)

    used_search = False
    retrieved = []
    reply = ""
    tokens_used = 0
    usd_cost = 0.0

    with st.chat_message("assistant"):
        with st.spinner(current_lang["spinner"]):
            with get_openai_callback() as cb:
                action, content = ask_item_or_search(
                    chat=chat,
                    config=LLM_CONFIG,
                    functions=FUNCTIONS,
                    item_json=item_js,
                    history=st.session_state.history,
                    question=user_question,
                    lang=st.session_state.lang,
                    data_dir=DATA_DIR,
                    logger=logger,
                )

                if action == "search":
                    used_search = True
                    meta_str = short_meta_from_item(item_js)
                    ctx_label = "Context" if st.session_state.lang == "en" else "Kontekstas"
                    search_query = f"{user_question}\n{ctx_label}: {meta_str}" if meta_str else user_question
                    retrieved = retrieve(
                        embeddings=embeddings,
                        index=index,
                        namespace=PINECONE_NAMESPACE,
                        query=search_query,
                        top_k=5,
                        exclude_title=exclude_title_lt,
                        logger=logger,
                    )
                    reply = answer_from_search(
                        chat=chat,
                        config=LLM_CONFIG,
                        selected_title=selected_title_ui,
                        question=user_question,
                        chunks=retrieved,
                        item_json=item_js,
                        lang=st.session_state.lang,
                        logger=logger,
                    )
                elif action == "answer":
                    reply = content
                else:
                    reply = content

                tokens_used = cb.total_tokens
                usd_cost = cb.total_cost or 0.0

        st.markdown(reply)
        st.caption(f"{current_lang['tokens']}: {tokens_used} • {current_lang['cost']}: ${usd_cost:.4f}")

    st.session_state.history.append({"role": "assistant", "content": reply})
    logger.info(f"Assistant: {reply}")
    logger.info(f"Usage: tokens={tokens_used}, cost=${usd_cost:.6f}")

    try:
        obj_path = save_conversation_to_gcs(st.session_state.history, LOG_BUCKET)
        logger.info(f"Saved conversation to gs://{LOG_BUCKET}/{obj_path}")
    except Exception as e:
        logger.error(f"Failed to save conversation to GCS: {e}")


if st.session_state.view == "gallery":
    render_gallery()
else:
    render_chat()
