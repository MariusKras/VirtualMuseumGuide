"""Utility functions for the museum guide app, including config loading,
prompt handling, exhibit data helpers, retrieval, and answer generation."""

import json, base64
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from google.cloud import storage
from datetime import datetime


def load_llm_config(url: str) -> dict:
    """Fetch and parse the remote LLM config JSON from the given URL."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def system_prompt_route(config: dict, lang: str) -> str:
    """Return the routing system prompt for the given language."""
    return (
        config["system_prompts"]["route_en"]
        if lang == "en"
        else config["system_prompts"]["route_lt"]
    )


def system_prompt_answer(config: dict, lang: str) -> str:
    """Return the answering system prompt for the given language."""
    return (
        config["system_prompts"]["answer_en"]
        if lang == "en"
        else config["system_prompts"]["answer_lt"]
    )


def explain_text(config: dict, lang: str) -> str:
    """Return the explainer text for the given language."""
    return (
        config["explain_text"]["en"] if lang == "en" else config["explain_text"]["lt"]
    )


def load_gallery(assets_dir: Path, data_dir: Path, logger) -> list[dict]:
    """Load gallery items by pairing processed images with their JSON metadata."""
    items = []
    for img in sorted(assets_dir.glob("*.jpg")):
        base = img.stem
        try:
            with open(data_dir / f"{base}.json", "r", encoding="utf-8") as f:
                js = json.load(f)
            if not isinstance(js, dict):
                raise TypeError("JSON root is not a dict")
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as e:
            logger.error(f'Gallery JSON load failed for "{base}.json": {e}')
            continue
        title_lt = js.get("pavadinimas") or base
        title_en = js.get("pavadinimas_en") or title_lt
        items.append(
            {"id": base, "title_lt": title_lt, "title_en": title_en, "image": str(img)}
        )
    return items


def load_item_json(data_dir: Path, item_id: str, logger) -> dict:
    """Load a single exhibit JSON by item id with error handling."""
    try:
        with open(data_dir / f"{item_id}.json", "r", encoding="utf-8") as f:
            js = json.load(f)
        if not isinstance(js, dict):
            raise TypeError("JSON root is not a dict")
        return js
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as e:
        logger.error(f'Item JSON load failed for "{item_id}.json": {e}')
        return {}


def short_meta_from_item(js: dict) -> str:
    """Build a short metadata string from key exhibit fields."""
    fields = {
        "pavadinimas": js.get("pavadinimas"),
        "technika": js.get("technika"),
        "laikotarpis": js.get("laikotarpis"),
        "medžiaga": js.get("medžiaga"),
    }
    parts = [f"{k}: {v}" for k, v in fields.items() if v]
    return "; ".join(parts)


def retrieve(
    embeddings,
    index,
    namespace: str,
    query: str,
    top_k: int = 5,
    exclude_title: str | None = None,
    logger=None,
) -> list[dict]:
    """Query Pinecone with an embedded search vector and return matched snippets."""
    vec = embeddings.embed_query(query)
    kwargs = dict(
        namespace=namespace,
        vector=vec,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )
    if exclude_title:
        kwargs["filter"] = {"pavadinimas": {"$ne": exclude_title}}
    try:
        res = index.query(**kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Pinecone query failed: {e}")
        return []
    out = []
    for m in res.matches or []:
        md = m.metadata or {}
        item_name = md.get("pavadinimas")
        chunk_text = md.get("chunk_text") or ""
        out.append(
            {"id": m.id, "score": m.score, "item": item_name, "text": chunk_text}
        )
    return out


def build_timeline_text(data_dir: Path, lang: str, logger) -> str:
    """Construct an unsorted timeline text from all exhibit JSON files."""
    header = "Timeline (unsorted):" if lang == "en" else "Laiko juosta (nesurikiuota):"
    missing = "Period not specified" if lang == "en" else "Nenurodytas laikotarpis"
    lines = []
    for jf in sorted(data_dir.glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                js = json.load(f)
            if not isinstance(js, dict):
                raise TypeError("JSON root is not a dict")
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as e:
            logger.error(f'Timeline JSON load failed for "{jf.name}": {e}')
            continue
        title = js.get("pavadinimas", jf.stem)
        period = js.get("laikotarpis") or missing
        lines.append(f"• {title} — {period}\n")
    return header + "\n" + "\n".join(lines)


def greet_on_select(lang: str, title: str) -> str:
    """Return the greeting shown when an exhibit is selected."""
    if lang == "en":
        return (
            f'Hello! You selected: "{title}". '
            "You can ask about this exhibit. If your question involves other exhibits, I can look across the available collection, "
            "but the conversation will stay focused on your selected item. If you want to know how to use the guide, just ask "
            "“How do I use this guide?”"
        )
    return (
        f'Sveiki! Pasirinkote: "{title}". '
        "Galite klausti apie šį eksponatą. Jei klausimas apima ir kitus eksponatus, galiu paieškoti visoje turimoje kolekcijoje, "
        "tačiau pokalbis liks sutelktas į jūsų pasirinktą eksponatą. Jei norite sužinoti, kaip naudotis gidu, "
        "tiesiog paklauskite „Kaip naudotis šiuo gidu?“"
    )


def ask_item_or_search(
    chat,
    config: dict,
    functions: list[dict],
    item_json: dict,
    history: list[dict],
    question: str,
    lang: str,
    data_dir: Path,
    logger,
):
    """Decide to answer directly or call a function; return (action, content)."""
    system = system_prompt_route(config, lang)
    trimmed, user_count, assistant_count = [], 0, 0
    for msg in reversed(history):
        if msg["role"] == "user" and user_count < 4:
            trimmed.append(msg)
            user_count += 1
        elif msg["role"] == "assistant" and assistant_count < 4:
            trimmed.append(msg)
            assistant_count += 1
        if user_count >= 4 and assistant_count >= 4:
            break
    trimmed = list(reversed(trimmed))
    lc_msgs = [SystemMessage(content=system)]
    for m in trimmed:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_msgs.append(AIMessage(content=m["content"]))
    item_blob = json.dumps(item_json, ensure_ascii=False)
    prefix = (
        "Selected exhibit (JSON):" if lang == "en" else "Pasirinktas eksponatas (JSON):"
    )
    q_label = "Question:" if lang == "en" else "Klausimas:"
    lc_msgs.append(
        HumanMessage(content=f"{prefix}\n{item_blob}\n\n{q_label}\n{question}")
    )
    chat_fc = chat.bind(functions=functions, function_call="auto")
    try:
        response = chat_fc.invoke(lc_msgs)
    except Exception as e:
        logger.error(f"LLM call failed in ask_item_or_search: {e}")
        return "answer", (
            "Sorry, I can’t answer right now. Please try again later."
            if lang == "en"
            else "Atsiprašau, šiuo metu negaliu atsakyti. Pabandykite dar kartą vėliau."
        )
    fc = response.additional_kwargs.get("function_call")
    if fc:
        name = fc.get("name")
        if name == "search_documents":
            return "search", ""
        elif name == "explain_app":
            return "answer", explain_text(config, lang)
        elif name == "timeline":
            return "answer", build_timeline_text(data_dir, lang, logger)
    content = (response.content or "").strip()
    return None, content


def answer_from_search(
    chat,
    config: dict,
    selected_title: str,
    question: str,
    chunks: list[dict],
    item_json: dict,
    lang: str,
    logger,
):
    """Generate the final answer using retrieved snippets and the answer prompt."""
    system = system_prompt_answer(config, lang)
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"Source {i}: {c['item']}\nExcerpt:\n{c['text'][:1200]}"
            if lang == "en"
            else f"Šaltinis {i}: {c['item']}\nIštrauka:\n{c['text'][:1200]}"
        )
    ground = (
        "\n\n---\n".join(blocks)
        if blocks
        else (
            "(No relevant snippets found.)"
            if lang == "en"
            else "(Nerasta aktualių ištraukų.)"
        )
    )
    item_blob = json.dumps(item_json, ensure_ascii=False)
    head1 = (
        "Initially selected exhibit"
        if lang == "en"
        else "Pradinis vartotojo pasirinktas eksponatas"
    )
    head2 = (
        "Selected exhibit description (JSON)"
        if lang == "en"
        else "Pasirinkto eksponato aprašas (JSON)"
    )
    head3 = "Question" if lang == "en" else "Klausimas"
    head4 = "Retrieved snippets" if lang == "en" else "Rastos ištraukos"
    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"{head1}: {selected_title}\n\n"
                f"{head2}:\n{item_blob}\n\n"
                f"{head3}:\n{question}\n\n"
                f"{head4}:\n{ground}"
            )
        ),
    ]
    try:
        response = chat.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed in answer_from_search: {e}")
        return (
            "Sorry, I can’t provide an answer right now."
            if lang == "en"
            else "Atsiprašau, šiuo metu negaliu pateikti atsakymo."
        )


def to_data_url(p: str) -> str:
    """Return a data URL for a local JPEG file path."""
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def save_conversation_to_gcs(conversation: list[dict], bucket_name: str, prefix: str = "conversations") -> str:
    """Persist the conversation JSON to a Cloud Storage bucket and return the object path."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_path = f"{prefix}/conversation_{ts}.json"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(conversation, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    return blob_path
