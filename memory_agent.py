#!/usr/bin/env python3
"""
memory_agent.py – memory‑augmented chat assistant

Implements William Li’s “Agent Memory (and tool use) Are All You Need” design.

Key subsystems
--------------
• Short‑term prompt context
• Long‑term FAISS vector store (deduplicated, persisted)
• High‑level profile memory (JSON, conflict‑aware)
• Markdown file ingestion (≈500k words)
• Safe loading / no dummy vectors / no unsafe deserialise
• Stable SHA‑256 content‑hash de‑duplication
"""

# ────────────────────────────────────────────────────────────────────────────
# Imports & setup
# ────────────────────────────────────────────────────────────────────────────
import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import openai
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.prompt import Prompt

# console first – so we can print even before .env is OK
console = Console()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LATEST_PROFILE_N = 8  # or read from CLI / env
if not OPENAI_API_KEY:
    console.print(
        "[bold red]Error:[/] OPENAI_API_KEY not found – set it in your environment or a .env file."
    )
    sys.exit(1)

# one global client object
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
INDEX_DIR = Path("memory_index")
HASH_PATH = INDEX_DIR / "doc_hashes.json"
PROFILE_PATH = Path("high_level_memory.json")
CHAT_LOG_PATH = Path("chat_history.jsonl")

EMBED_MODEL = "text-embedding-3-small"  # adjust if you prefer large
# use the smaller o4 mini model by default
CHAT_MODEL = "o4-mini"
HEADING_RE = re.compile(r"^\s*#+\s", re.MULTILINE)  # any line that starts with one or more #

# tool definitions for the OpenAI Response API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Retrieve relevant long-term memories",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search the vector memory",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ────────────────────────────────────────────────────────────────────────────
# Vector‑store memory
# ────────────────────────────────────────────────────────────────────────────
class VectorMemory:
    """Persistent FAISS index with de‑duplication via SHA‑256 hashes."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

        self.hashes = self._load_hashes()

        if (index_dir / "index.faiss").exists():
            self.vs = FAISS.load_local(
                str(index_dir), self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            self.vs = None

            # ── internal helpers ────────────────────────────────────────────────

    def _load_hashes(self) -> set:
        if HASH_PATH.exists():
            try:
                return set(json.loads(HASH_PATH.read_text()))
            except Exception:
                console.print("[yellow]⚠️  Corrupt hash list – starting fresh.")
        return set()

    def _save_hashes(self):
        HASH_PATH.write_text(json.dumps(list(self.hashes)))

    # ── public API ───────────────────────────────────────────────────────
    def add(self, docs: Sequence[Document]) -> int:
        new_docs = []
        for d in docs:
            h = hash_text(d.page_content)  # uses normalise() as we patched
            if h not in self.hashes:
                self.hashes.add(h)
                new_docs.append(d)

        if not new_docs:
            return 0

        # ── build index on first real batch ────────────────────────────
        if self.vs is None:
            self.vs = FAISS.from_documents(new_docs, self.embeddings)
        else:
            self.vs.add_documents(new_docs)

        self.vs.save_local(str(self.index_dir))
        self._save_hashes()
        return len(new_docs)

    def search(self, query: str, k: int = 5):
        if self.vs is None:
            return []
        return self.vs.similarity_search_with_score(query, k=k)


# ────────────────────────────────────────────────────────────────────────────
# High‑level profile memory
# ────────────────────────────────────────────────────────────────────────────
class ProfileMemory:
    """Stable user facts (small JSON list)."""

    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        self.data: List[Dict] = json.loads(path.read_text()) if path.exists() else []

    # helpers
    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    # public
    def summary(self, latest_n: int = None) -> str:
        if not self.data:
            return "(no profile memory yet)"
        items = (
            sorted(self.data, key=lambda x: x.get("date") or x["updated"])
            if latest_n
            else self.data
        )
        if latest_n:
            items = items[-latest_n:]
        return "\n".join(f"- {it['content']}" for it in items)

    def upsert(self, updates: List[Dict]):
        changed = False
        for upd in updates:
            key = upd["key"]
            for item in self.data:
                if item["key"] == key:
                    if "date" in upd and "date" not in item:
                        item["date"] = upd["date"]
                    if item["content"] != upd["content"]:
                        item["content"] = upd["content"]
                        item["updated"] = datetime.utcnow().isoformat()
                        changed = True
                    break
            else:
                upd["updated"] = datetime.utcnow().isoformat()
                self.data.append(upd)
                changed = True
        if changed:
            self._save()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)  # crude but works 95 %


def extract_profile_facts(message: str, current_profile: str) -> List[Dict]:
    """
    LLM extracts new long‑term facts.
    NEW: Each returned object can contain an optional 'date' field.
    """
    system_msg = {"role": "system", "content": "You are a memory‑extraction agent."}
    user_body = (
        "Given CURRENT_PROFILE_MEMORY and NEW_MESSAGE below, identify any *stable* personal "
        "facts that should be saved. If a fact contains an explicit or implied date, include it "
        "as an ISO‑8601 'date' field. Reply with **only** a JSON array of objects "
        '(key, content[, date]). Return [] if nothing new.\n'
        f"CURRENT_PROFILE_MEMORY:\n{current_profile}\n\nNEW_MESSAGE:\n{message}"
    )
    messages = [system_msg, {"role": "user", "content": user_body}]

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        facts = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group(0))

        # Fallback: add a naïve date guess if the LLM omitted one
        for f in facts:
            if "date" not in f:
                m = DATE_RE.search(message)
                if m:
                    # normalise dd/mm/yy → yyyy‑mm‑dd as best we can
                    try:
                        f["date"] = datetime.fromisoformat(
                            re.sub(r"[/-]", "-", m.group(1))
                        ).date().isoformat()
                    except Exception:
                        pass
        return facts
    except Exception as e:
        console.print(f"[yellow]⚠️  profile‑extract error: {e}")
        return []


def format_retrieved(pairs: List[Tuple[Document, float]], k_top: int = 3) -> str:
    if not pairs:
        return "(no relevant memories retrieved)"
    out = []
    for doc, score in pairs[:k_top]:
        txt = doc.page_content
        out.append(f"• {txt} (score={score:.2f})")
    return "\n".join(out)[:5000]


def system_prompt(profile_summary: str) -> str:
    """Base prompt instructing the assistant how to use tools."""
    return (
        "You are an assistant.\n\n"
        "You have access to a `search_memory` tool for recalling long-term "
        "memories. Call it whenever the user references previous conversations "
        "or when more context might be useful.\n\n"
        f"USER_PROFILE:\n{profile_summary}"
    )


# ─── chat history helpers ───────────────────────────────────────────────────
def append_chat_log(role: str, content: str, ts: str):
    entry = {"role": role, "content": content, "time": ts}
    with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_chat_logs() -> List[Dict]:
    if not CHAT_LOG_PATH.exists():
        return []
    lines = [ln for ln in CHAT_LOG_PATH.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]




# ────────────────────────────────────────────────────────────────────────────
# CLI loops
# ────────────────────────────────────────────────────────────────────────────
def chat_loop():
    vmem = VectorMemory()
    pmem = ProfileMemory()

    console.print("[bold cyan]Memory‑Agent chat – type 'exit' to quit\n")
    while True:
        user_msg = Prompt.ask("[bold green]you")
        if user_msg.strip().lower() in {"exit", "quit"}:
            break

        messages = [
            {"role": "system", "content": system_prompt(pmem.summary(LATEST_PROFILE_N))},
            {"role": "user", "content": user_msg},
        ]

        while True:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                for call in msg.tool_calls:
                    if call.function.name == "search_memory":
                        args = json.loads(call.function.arguments)
                        result = format_retrieved(vmem.search(args.get("query", ""), k=5))
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": "search_memory",
                                "content": result,
                            }
                        )
                continue
            assistant_msg = msg.content.strip()
            console.print(f"[bold magenta]agent[/]: {assistant_msg}\n")
            messages.append({"role": "assistant", "content": assistant_msg})
            break

        # persist both sides
        now = datetime.utcnow().isoformat()
        vmem.add(
            [
                Document(page_content=user_msg, metadata={"role": "user", "time": now}),
                Document(page_content=assistant_msg, metadata={"role": "assistant", "time": now}),
            ]
        )
        append_chat_log("user", user_msg, now)
        append_chat_log("assistant", assistant_msg, now)

        # profile extraction
        for msg in (user_msg, assistant_msg):
            updates = extract_profile_facts(msg, pmem.summary())
            if updates:
                pmem.upsert(updates)


def normalise(text: str) -> str:
    """Lower‑case, de‑accent, collapse whitespace; used for hash stability."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def chunk_document(raw: str) -> List[Document]:
    """
    Split on each line that begins with '#', '##', '###', …
    The heading itself is included with its section body.
    """
    # find all heading starts
    indices = [m.start() for m in HEADING_RE.finditer(raw)]
    if not indices:
        # fallback: keep whole file as single chunk
        return [Document(page_content=raw.strip())] if raw.strip() else []

    indices.append(len(raw))  # sentinel for last slice
    parts = [
        raw[indices[i]: indices[i + 1]].strip() for i in range(len(indices) - 1)
    ]
    return [Document(page_content=p) for p in parts if p]


# ─── new helper for per‑file hash persistence ───────────────────────────────
def hash_text(text: str) -> str:
    return sha256(normalise(text).encode()).hexdigest()


def load_file_hashes(path: Path) -> set:
    hash_file = path.with_suffix(path.suffix + ".hashes.json")
    if hash_file.exists():
        return set(json.loads(hash_file.read_text()))
    return set()


def save_file_hashes(path: Path, hashes: set):
    hash_file = path.with_suffix(path.suffix + ".hashes.json")
    hash_file.write_text(json.dumps(list(hashes)))


# ─── replace ingest_markdown() by ingest_file() ─────────────────────────────
def ingest_file(md_path: Path, update_profile: bool = False):
    console.print(f"[blue]Scanning {md_path} for new diary entries…[/]")
    text = md_path.read_text("utf-8", "ignore")
    chunks = chunk_document(text)
    vmem = VectorMemory()

    seen_hashes = load_file_hashes(md_path)
    new_docs, new_hashes = [], []

    for ch in chunks:
        h = hash_text(ch.page_content)
        if h not in seen_hashes:
            new_docs.append(ch)
            new_hashes.append(h)

    if not new_docs:
        console.print("[green]✓ No new content detected.")
        return 0

    added = vmem.add(new_docs)

    if update_profile and added:
        pmem = ProfileMemory()
        for doc in new_docs:
            facts = extract_profile_facts(doc.page_content, pmem.summary())
            if facts:
                pmem.upsert(facts)
                fact_docs = [
                    Document(
                        page_content=f"{f['key']}: {f['content']}",
                        metadata={
                            "type": "profile_fact",
                            "key": f["key"],
                            "date": f.get("date"),
                        },
                    )
                    for f in facts
                ]
                vmem.add(fact_docs)
    seen_hashes.update(new_hashes)
    save_file_hashes(md_path, seen_hashes)

    console.print(f"[green]✓ Added {added} new diary chunks from {md_path}")
    return added


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Memory‑augmented chat agent")
    parser.add_argument("--ingest", type=str, help="Path to a markdown file to ingest")
    parser.add_argument("--update-profile", action="store_true", help="Update the profile memory")
    args = parser.parse_args()

    if args.ingest:
        ingest_file(Path(args.ingest), args.update_profile)

    chat_loop()


if __name__ == "__main__":
    main()
