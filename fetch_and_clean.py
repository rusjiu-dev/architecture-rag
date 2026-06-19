"""
Скрипт: fetch_and_clean.py
Назначение: скачивание страниц Star Wars fandom, очистка HTML, извлечение текста.
Требования: pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import time

BASE_URL = "https://starwars.fandom.com/wiki/"
OUTPUT_DIR = "raw_docs/"

# Список страниц для скачивания
PAGES = [
    "Darth_Vader", "Luke_Skywalker", "Leia_Organa", "Han_Solo",
    "Obi-Wan_Kenobi", "Emperor_Palpatine", "Yoda", "Chewbacca",
    "R2-D2", "C-3PO", "Boba_Fett", "Mace_Windu",
    "Tatooine", "Coruscant", "Hoth", "Endor_(planet)",
    "Dagobah", "Alderaan", "Naboo", "Kamino",
    "Death_Star", "Lightsaber", "Millennium_Falcon", "TIE_Fighter",
    "X-wing_starfighter", "Star_Destroyer", "AT-AT",
    "Jedi_Order", "Sith", "Galactic_Empire", "Rebel_Alliance",
    "Wookiee", "Ewok", "Clone_trooper", "Stormtrooper"
]

HEADERS = {
    "User-Agent": "KnowledgeBaseBuilder/1.0 (educational purposes)"
}

def clean_text(html_content):
    """Извлечение чистого текста из HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Удаление навигации, скриптов, стилей, инфобоксов
    for element in soup.select(
        "nav, script, style, .infobox, .toc, .mw-editsection, "
        ".noprint, .reference, .reflist, .catlinks, footer, "
        ".sidebar, .mbox, .navigation-not-searchable"
    ):
        element.decompose()
    
    # Извлечение основного контента
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if not content:
        content = soup.select_one("#mw-content-text") or soup.body
    
    text = content.get_text(separator="\n", strip=True)
    
    # Очистка лишних пустых строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\[\d+\]', '', text)  # Удаление ссылок на источники
    text = text.strip()
    
    return text

def fetch_page(page_name):
    """Скачивание страницы."""
    url = BASE_URL + page_name
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for page in PAGES:
        safe_name = page.replace("/", "_").replace("(", "").replace(")", "")
        filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")
        
        if os.path.exists(filepath):
            print(f"[SKIP] {safe_name}")
            continue
        
        try:
            print(f"[FETCH] {page}...")
            html = fetch_page(page)
            text = clean_text(html)
            
            # Сохранение только если текст достаточно длинный (>500 символов)
            if len(text) > 500:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[OK] {safe_name}: {len(text)} chars")
            else:
                print(f"[WARN] {safe_name}: too short ({len(text)} chars), skipping")
            
            time.sleep(1)  # Вежливая задержка
            
        except Exception as e:
            print(f"[ERROR] {page}: {e}")

if __name__ == "__main__":
    main()