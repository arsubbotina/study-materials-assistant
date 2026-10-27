import json
import os
import re
from datetime import date
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage


OUTPUT_FILE = "requests.json"

SUBJECTS = [
    "численные методы",
    "компьютерные сети",
    "программирование на python",
    "физика",
]

GIGACHAT_CREDENTIALS = os.environ["GIGACHAT_CREDENTIALS"]


class RequestItem(BaseModel):
    date: str
    subject: str
    url: str


def get_page_text(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "student-agent"})
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)[:2000]


def normalize_subject(s: str) -> str:
    s = s.lower()
    for subj in SUBJECTS:
        if subj in s:
            return subj
    if "python" in s:
        return "программирование на python"
    if "физ" in s:
        return "физика"
    if "сеть" in s:
        return "компьютерные сети"
    return "численные методы"


def read_all() -> List[Dict]:
    if not os.path.exists(OUTPUT_FILE):
        return []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def write_item(item: RequestItem):
    data = read_all()
    data.append(item.model_dump())
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_report(subject: str, date_from: str, date_to: str) -> Dict[str, List[Dict]]:
    data = read_all()
    result: Dict[str, List[Dict]] = {}

    for it in data:
        if not (date_from <= it["date"] <= date_to):
            continue

        subj = normalize_subject(it["subject"])

        if subject != "все" and subj != normalize_subject(subject):
            continue

        result.setdefault(subj, []).append(it)

    return result


SYSTEM_PROMPT = f"""
Ты определяешь, к какому учебному предмету относится материал.

Предметы:
{chr(10).join("- " + s for s in SUBJECTS)}

Верни СТРОГО JSON:
{{
  "subject": "один из предметов"
}}
""".strip()


def main():
    llm = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=True,
        temperature=0.0,
    )

    print("Вставь URL для добавления материала")
    print("Или команду отчёта:")
    print("ОТЧЕТ <предмет|все> <YYYY-MM-DD> <YYYY-MM-DD>")
    print("q — выход\n")

    while True:
        user_input = input("> ").strip()

        if user_input == "q":
            break

        if user_input.startswith("ОТЧЕТ"):
            _, subject, date_from, date_to = user_input.split(maxsplit=3)
            report = build_report(subject, date_from, date_to)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print()
            continue

        url = user_input
        text = get_page_text(url)
        today = date.today().isoformat()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]

        response = llm.invoke(messages)
        data = json.loads(response.content)

        item = RequestItem(
            date=today,
            subject=normalize_subject(data["subject"]),
            url=url,
        )

        write_item(item)
        print(json.dumps(item.model_dump(), ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()