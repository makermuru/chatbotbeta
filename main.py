import ast
import operator
import os
import re
import sqlite3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


DB_KO = "qa_ko.db"
DB_EN = "qa_en.db"


# The free Render instance has 512 MiB of memory. Use multilingual-e5-small
# (80MB) with optional ONNX optimization for ~30% additional memory savings.
model = None
util = None
ENABLE_EMBEDDINGS = os.getenv("ENABLE_EMBEDDINGS", "false").lower() == "true"
USE_ONNX = os.getenv("USE_ONNX", "true").lower() == "true"

if ENABLE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer, util

        model_name = "jhgan/ko-sroberta-multitask"
        if USE_ONNX:
            try:
                from optimum.onnxruntime import ORTModelForSentenceTransformers
                from transformers import AutoTokenizer

                model = ORTModelForSentenceTransformers.from_pretrained(
                    model_name,
                    local_files_only=False,
                )
            except Exception:
                model = SentenceTransformer(
                    model_name,
                    local_files_only=False,
                )
        else:
            model = SentenceTransformer(
                model_name,
                local_files_only=False,
            )
    except Exception as e:
        model = None
        print(f"Embedding model failed to load: {e}")


# Embeddings are expensive to calculate. Keep them per open database and
# refresh them only when the user teaches the chatbot a new answer.
_search_cache = {}


def init_db(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_qa_question ON qa(question)"
    )

    conn.commit()
    return conn


conn_ko = init_db(DB_KO)
conn_en = init_db(DB_EN)


def _cache_key(conn):
    return conn.execute("PRAGMA database_list").fetchone()[2]


def _normalise(text):
    """Make superficial punctuation/case differences irrelevant to retrieval."""
    text = text.casefold().strip()
    return re.sub(r"[^\w가-힣]+", " ", text).strip()


def add_qa(conn, question, answer):
    question, answer = question.strip(), answer.strip()

    if not question or not answer:
        return False

    cur = conn.cursor()

    # Avoid silently accumulating exact duplicate knowledge.
    cur.execute(
        "SELECT 1 FROM qa WHERE question = ? COLLATE NOCASE LIMIT 1",
        (question,),
    )

    if cur.fetchone():
        return False

    cur.execute(
        "INSERT INTO qa (question, answer) VALUES (?, ?)",
        (question, answer),
    )

    conn.commit()

    _search_cache.pop(_cache_key(conn), None)

    return True


def _question_tokens(text):
    return set(
        token
        for token in _normalise(text).split()
        if len(token) > 1
    )


def find_answer(conn, user_question, threshold=64):
    """Return the best answer using semantic similarity plus exact key words."""

    if len(_normalise(user_question)) < 2:
        return None, 0

    key = _cache_key(conn)
    cached = _search_cache.get(key)

    if cached is None:
        rows = conn.execute(
            "SELECT question, answer FROM qa ORDER BY id"
        ).fetchall()

        if not rows:
            return None, 0

        questions = [row[0] for row in rows]

        embeddings = (
            model.encode(
                questions,
                convert_to_tensor=True,
            )
            if model
            else None
        )

        cached = (
            questions,
            [row[1] for row in rows],
            embeddings,
        )

        _search_cache[key] = cached

    questions, answers, db_embeddings = cached

    if model:
        query_embedding = model.encode(
            user_question,
            convert_to_tensor=True,
        )

        semantic_scores = util.cos_sim(
            query_embedding,
            db_embeddings,
        )[0]
    else:
        semantic_scores = [0.0] * len(questions)

    query_tokens = _question_tokens(user_question)

    combined_scores = []

    for question, semantic_score in zip(
        questions,
        semantic_scores,
    ):
        candidate_tokens = _question_tokens(question)

        overlap = len(
            query_tokens & candidate_tokens
        ) / max(
            1,
            len(query_tokens | candidate_tokens),
        )

        # Semantic matching handles paraphrases; lexical overlap protects
        # short factual questions that share important words.
        combined_scores.append(
            (
                float(semantic_score) * 0.82
                + overlap * 0.18
            )
            if model
            else overlap
        )

    best_idx = max(
        range(len(combined_scores)),
        key=combined_scores.__getitem__,
    )

    best_score = combined_scores[best_idx] * 100

    return (
        (answers[best_idx], best_score)
        if best_score >= threshold
        else (None, best_score)
    )


def detect_lang(text):
    return "ko" if re.search(r"[가-힣]", text) else "en"


WORD_TO_OP = {
    "더하기": "+",
    "더해서": "+",
    "더해": "+",
    "플러스": "+",
    "빼기": "-",
    "빼서": "-",
    "빼면": "-",
    "빼": "-",
    "마이너스": "-",
    "곱하기": "*",
    "곱해서": "*",
    "곱하면": "*",
    "곱해": "*",
    "나누기": "/",
    "나눠서": "/",
    "나누면": "/",
    "나눠": "/",
    "plus": "+",
    "add": "+",
    "added to": "+",
    "minus": "-",
    "subtract": "-",
    "subtracted from": "-",
    "multiplied by": "*",
    "times": "*",
    "multiply": "*",
    "divided by": "/",
    "divide": "/",
}


def normalize_expression(text):
    result = text.casefold()

    for word in sorted(
        WORD_TO_OP,
        key=len,
        reverse=True,
    ):
        result = result.replace(
            word,
            f" {WORD_TO_OP[word]} ",
        )

    result = re.sub(
        r"(?<=\d)\s*[x×]\s*(?=\d)",
        " * ",
        result,
    )

    result = re.sub(
        r"(?<=\d)\s*÷\s*(?=\d)",
        " / ",
        result,
    )

    return result


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(node):
    if isinstance(
        node,
        ast.Constant,
    ) and isinstance(
        node.value,
        (int, float),
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](
            safe_eval(node.left),
            safe_eval(node.right),
        )

    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](
            safe_eval(node.operand),
        )

    raise ValueError("Unsupported expression")


def solve_math(expression):
    try:
        return safe_eval(
            ast.parse(
                expression,
                mode="eval",
            ).body
        )
    except (
        ArithmeticError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        return None


MATH_EXTRACT_PATTERN = re.compile(
    r"-?\d+(?:\.\d+)?"
    r"(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+"
)


def extract_math_expression(text):
    match = MATH_EXTRACT_PATTERN.search(
        normalize_expression(text)
    )

    return match.group().strip() if match else None


SUSPICIOUS_RE = re.compile(
    r"\b(?:DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|"
    r"UPDATE\s+.+\bSET|UNION\s+SELECT)\b|"
    r"--\s*$|;\s*--|"
    r"['\"]\s*OR\s*['\"]?1\s*=\s*['\"]?1",
    re.IGNORECASE,
)


def is_suspicious(text):
    return bool(SUSPICIOUS_RE.search(text))


def get_response(
    conn_ko,
    conn_en,
    user_input,
    pending_question=None,
):
    if is_suspicious(user_input):
        return "Nice try.", None

    lang = detect_lang(user_input)
    conn = conn_ko if lang == "ko" else conn_en

    if pending_question:
        added = add_qa(
            conn,
            pending_question,
            user_input,
        )

        return (
            (
                "배웠어요!"
                if lang == "ko"
                else "Learned!"
            ),
            None,
        ) if added else (
            (
                "이미 알고 있어요."
                if lang == "ko"
                else "I already know that."
            ),
            None,
        )

    math_expr = extract_math_expression(user_input)

    if math_expr:
        result = solve_math(math_expr)

        if result is not None:
            return f"{math_expr} = {result}", None

    if ":" in user_input:
        question, answer = user_input.split(":", 1)

        added = add_qa(
            conn,
            question,
            answer,
        )

        return (
            "저장 완료했어요."
            if added
            else "같은 질문이 이미 있어요."
        ), None

    answer, score = find_answer(
        conn,
        user_input,
    )

    if answer:
        return answer, None

    if lang == "ko":
        return (
            f"아직 정확한 답을 모르겠어요 "
            f"(가장 가까운 질문: {score:.1f}%). "
            f"답을 알려주시면 배울게요!",
            user_input,
        )

    return (
        f"I do not know that yet "
        f"(best match: {score:.1f}%). "
        f"Teach me the answer!",
        user_input,
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Pseudo-AI API",
    description="SQLite + semantic search based QA chatbot API",
    version="1.0.0",
)

allowed_origins = [
    value.strip()
    for value in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str


class TeachRequest(BaseModel):
    question: str
    answer: str


@app.get("/")
def root():
    return {
        "name": "Pseudo-AI API",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": model is not None,
        "search_mode": "semantic" if model else "keyword",
        "model_name": "multilingual-e5-small" if model else None,
        "onnx_optimized": USE_ONNX if model else None,
    }


@app.post("/ask")
def ask(request: AskRequest):
    answer, pending_question = get_response(
        conn_ko,
        conn_en,
        request.question,
    )

    return {
        "answer": answer,
        "pending_question": pending_question,
    }


@app.post("/teach")
def teach(request: TeachRequest):
    lang = detect_lang(request.question)

    conn = (
        conn_ko
        if lang == "ko"
        else conn_en
    )

    added = add_qa(
        conn,
        request.question,
        request.answer,
    )

    return {
        "success": added,
        "message": (
            "배웠어요!"
            if added
            else "이미 알고 있어요."
        ),
    }


@app.get("/ask")
def ask_get(question: str):
    answer, pending_question = get_response(
        conn_ko,
        conn_en,
        question,
    )

    return {
        "answer": answer,
        "pending_question": pending_question,
    }
