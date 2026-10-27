import json
import os
import re
from datetime import date
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

import telebot

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage


BOT_TOKEN = os.environ["BOT_TOKEN"]
GIGACHAT_CREDENTIALS = os.environ["GIGACHAT_CREDENTIALS"]

OUTPUT_FILE = "requests.json"
SUBJECTS = [
    "численные методы",
    "компьютерные сети",
    "программирование на python",
    "физика",
]


class RequestItem(BaseModel):
    date: str
    subject: str
    url: str


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_text(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "study-agent"}, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean(soup.get_text(" ", strip=True))[:2000]


def normalize_subject(s: str) -> str:
    s = (s or "").lower()
    for subj in SUBJECTS:
        if subj in s:
            return subj
    if "python" in s:
        return "программирование на python"
    if "физ" in s:
        return "физика"
    if "сеть" in s or "network" in s:
        return "компьютерные сети"
    return "численные методы"


def read_all() -> List[Dict[str, Any]]:
    if not os.path.exists(OUTPUT_FILE):
        return []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def write_item(item: RequestItem) -> None:
    data = read_all()
    data.append(item.model_dump())
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_report(subject: str, d1: str, d2: str) -> Dict[str, List[Dict[str, Any]]]:
    data = read_all()
    result: Dict[str, List[Dict[str, Any]]] = {}
    for it in data:
        if not (d1 <= it["date"] <= d2):
            continue
        s = normalize_subject(it.get("subject", ""))
        if subject.lower() not in ("все", "all") and s != normalize_subject(subject):
            continue
        result.setdefault(s, []).append(it)
    return result


SYSTEM_PROMPT = f"""
Ты определяешь, для какого учебного предмета полезен материал.

Предметы:
{chr(10).join("- " + s for s in SUBJECTS)}

Верни СТРОГО JSON:
{{"subject": "один из предметов"}}
""".strip()


llm = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    verify_ssl_certs=True,
    temperature=0.0,
)

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "Отправь ссылку — я сохраню учебный материал.\n"
        "Отчёт:\nОТЧЕТ <предмет|все> <YYYY-MM-DD> <YYYY-MM-DD>\n"
        "Пример:\nОТЧЕТ все 2026-01-01 2026-01-31"
    )


@bot.message_handler(content_types=["text"])
def handle_message(message):
    text = message.text.strip()

    if text.startswith("ОТЧЕТ"):
        _, subject, d1, d2 = text.split(maxsplit=3)
        report = build_report(subject, d1, d2)
        bot.send_message(
            message.chat.id,
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        return

    url = text
    page_text = fetch_text(url)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=page_text),
    ])
    subject = normalize_subject(json.loads(response.content)["subject"])

    item = RequestItem(
        date=date.today().isoformat(),
        subject=subject,
        url=url,
    )
    write_item(item)

    bot.send_message(
        message.chat.id,
        json.dumps(item.model_dump(), ensure_ascii=False, indent=2)
    )


def main():
    bot.infinity_polling()


if __name__ == "__main__":
    main()

