import json, os, pickle, re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

REQ, IDX = "requests.json", "tfidf_index.pkl"
SUBJECTS = ["численные методы", "компьютерные сети", "программирование на python", "физика"]
GIGACHAT_CREDENTIALS = os.environ["GIGACHAT_CREDENTIALS"]


def norm_subject(s: str) -> str:
    s = (s or "").lower()
    if "python" in s: return "программирование на python"
    if "физ" in s: return "физика"
    if "сеть" in s or "network" in s: return "компьютерные сети"
    for x in SUBJECTS:
        if x in s: return x
    return "численные методы"


def fetch_text(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "study-agent"}, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript"]): t.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def chunks(t: str, n=900, ov=180):
    i = 0
    while i < len(t):
        yield t[i:i+n]
        i += n - ov


def build_index():
    items = json.load(open(REQ, "r", encoding="utf-8"))
    texts, meta = [], []
    for it in items:
        page = fetch_text(it["url"])
        subj = norm_subject(it.get("subject", ""))
        dt = it.get("date", "")
        for ch in chunks(page):
            texts.append(ch)
            meta.append({"url": it["url"], "subject": subj, "date": dt})
    vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    pickle.dump((vec, X, texts, meta), open(IDX, "wb"))
    print("Индекс построен")


def load_index():
    return pickle.load(open(IDX, "rb"))


def retrieve(topic: str, subject: str | None, k: int):
    vec, X, texts, meta = load_index()
    sims = cosine_similarity(vec.transform([topic]), X).ravel()

    if subject:
        s = norm_subject(subject)
        idxs = [i for i in range(len(texts)) if meta[i]["subject"] == s]
    else:
        idxs = list(range(len(texts)))

    idxs.sort(key=lambda i: sims[i], reverse=True)
    return [(texts[i], meta[i], float(sims[i])) for i in idxs[:k]]


def detect_subject(topic: str, probe_k: int = 12) -> str:
    top = retrieve(topic, subject=None, k=probe_k)
    scores = {s: 0.0 for s in SUBJECTS}
    for _, m, sc in top:
        scores[m["subject"]] += sc
    return max(scores.items(), key=lambda x: x[1])[0]


SYS = (
    "Ты — помощник для подготовки к экзамену. Используй ТОЛЬКО контекст.\n"
    "Структура:\n"
    "1) Суть темы (2–4 предложения)\n"
    "2) 6–10 ключевых пунктов\n"
    "3) Мини-конспект (структурировано)\n"
    "4) 5 типичных экзаменационных вопросов + короткие ответы\n"
    "Если контекста не хватает — скажи, чего не хватает.\n"
)


def answer(topic: str):
    subj = detect_subject(topic)
    top = retrieve(topic, subject=subj, k=8)

    ctx = "\n\n".join(
        f"Источник: {m['url']} | предмет: {m['subject']} | дата: {m['date']} | score: {sc:.3f}\n{t}"
        for t, m, sc in top
    )
    sources = sorted({m["url"] for _, m, _ in top})

    llm = GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=True, temperature=0.2)
    resp = llm.invoke([SystemMessage(content=SYS), HumanMessage(content=f"ТЕМА: {topic}\n\nКОНТЕКСТ:\n{ctx}")])

    print(f"\nОпределённый предмет: {subj}\n")
    print(resp.content.strip())
    print("\nИсточники:\n" + "\n".join(f"- {u}" for u in sources))


def main():
    if not os.path.exists(IDX):
        build_index()

    topic = input("Тема: ").strip()
    answer(topic)


if __name__ == "__main__":
    main()