"""
fetch_and_clean.py (v5)
Исправления v5:
- Добавлена нормализация переносов строк (склеивание разорванных абзацев).
- Расширен список CUTOFF_MARKERS.
- Расширен список JUNK_LINE_PATTERNS.

Требования: pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import time

BASE_URL = "https://starwars.fandom.com/wiki/"
OUTPUT_DIR = "raw_docs/"

PAGES = [
    "Darth_Vader", "Luke_Skywalker", "Leia_Organa", "Han_Solo",
    "Obi-Wan_Kenobi", "Emperor_Palpatine", "Yoda", "Chewbacca",
    "R2-D2", "C-3PO", "Boba_Fett", "Mace_Windu",
    "Tatooine", "Coruscant", "Hoth", "Endor_(planet)",
    "Dagobah", "Alderaan", "Naboo", "Kamino",
    "Death_Star", "Lightsaber", "Millennium_Falcon", "TIE_Fighter",
    "X-wing_starfighter", "Star_Destroyer", "AT-AT",
    "Jedi_Order", "Sith", "Galactic_Empire", "Rebel_Alliance",
    "Wookiee", "Ewok", "Clone_trooper"
]

HEADERS = {
    "User-Agent": "KnowledgeBaseBuilder/1.0 (educational purposes)"
}

# =========================================================================
# МУСОРНЫЕ ПАТТЕРНЫ
# =========================================================================
JUNK_LINE_PATTERNS = [
    r'Flux Wars Helmet Collection',
    r'Star Wars Helmet Collection',
    r'Helmet Collection',
    r'Micro Galaxy Squadron',
    r'LEGO Star Wars',
    r'LEGO\s+',
    r'Databank A-Z',
    r'\bDatabank\b.*\(backup link\)',
    r'StarWars\.com.*backup link',
    r'Highlights of the Saga',
    r'Weapons & Uniforms',
    r'Building the Galaxy:',
    r'^Quiz:',
    r'Quiz: What',
    r'Quiz: Which',
    r'on StarWars\.com\s*$',
    r'on Jazwares\s*$',
    r'\(backup link\)\s*$',
    r'\(backup link\)',
    r'backup link',
    r'\.\.\. \d+ (weeks|days|months) (ago|later)',
    r'^\s*\[Source\]\s*$',
    r'^\s*†\s*$',
    r'^\s*◊\s*$',
    r'^\s*\*+\s*$',
    r'^\s*\[\d+\]\s*$',
    r'\(First identified as [^\)]+\)',
    r'\(Picture only\)',
    r'^\s*&lt;.*&gt;\s*$',
    r'^\s*<.*>\s*$',
    r'^\s*[A-Za-z]\s*$',
]

JUNK_REGEXES = [re.compile(p, re.IGNORECASE) for p in JUNK_LINE_PATTERNS]

# =========================================================================
# МАРКЕРЫ ОТСЕЧЕНИЯ
# =========================================================================
CUTOFF_MARKERS = [
    "Star Wars: Force Arena",
    "Star Wars: Galaxy of Heroes",
    "LEGO Star Wars:",
    "The Star Wars Holiday Special",
    "Star Tours",
    "Disney Infinity",
    "Kinect Star Wars",
    "Angry Birds Star Wars",
    "Star Wars: Commander",
    "Star Wars: Galactic Defense",
    "Star Wars: Battlefront",
    "Star Wars: The Old Republic",
    "Star Wars: Squadrons",
]

# =========================================================================
# ЗАГОЛОВКИ СЕКЦИЙ ДЛЯ УДАЛЕНИЯ
# =========================================================================
SECTION_HEADERS_TO_REMOVE = [
    "References", "Bibliography", "External links",
    "Notes and references", "Notes", "Sources",
    "Footnotes", "Citations", "See also",
    "Appearances", "Further reading", "Media",
]

# =========================================================================
# МЕТКИ ИНФОБОКСОВ
# =========================================================================
INFOBOX_LABELS = [
    r'^Production information$', r'^Usage$', r'^Role\(s\)$',
    r'^Affiliation$', r'^Manufacturer$', r'^Designer$',
    r'^Model$', r'^Type$', r'^Class$', r'^Cost$',
    r'^Crew$', r'^Passengers$', r'^Capacity$',
    r'^Length$', r'^Width$', r'^Height$', r'^Max speed$',
    r'^Engine unit$', r'^Armament$', r'^First appearance$',
    r'^Latest appearance$', r'^Creator$', r'^Owners$',
    r'^Founded$', r'^Founder$', r'^Leader$', r'^Members$',
    r'^Headquarters$', r'^Language$', r'^Currency$',
    r'^Population$', r'^Region$', r'^Sector$', r'^System$',
    r'^Suns$', r'^Moons$', r'^Climate$', r'^Terrain$',
    r'^Fauna$', r'^Flora$', r'^Physical description$',
    r'^Hair color$', r'^Eye color$', r'^Skin color$',
    r'^Gender$', r'^Height$', r'^Mass$', r'^Homeworld$',
    r'^Born$', r'^Died$', r'^Chronological$', r'^Political$',
    r'^Historical information$', r'^Behind the scenes$',
]

INFOBOX_REGEXES = [re.compile(p, re.IGNORECASE) for p in INFOBOX_LABELS]

# =========================================================================
# HTML-СУЩНОСТИ
# =========================================================================
HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>',
    '&quot;': '"', '&#039;': "'", '&nbsp;': ' ',
    '&mdash;': '—', '&ndash;': '–', '&bull;': '•',
    '&copy;': '©', '&reg;': '®', '&trade;': '™',
    '&hellip;': '…', '&lsquo;': ''', '&rsquo;': ''',
    '&ldquo;': '"', '&rdquo;': '"',
}


# =========================================================================
# ФУНКЦИИ ОЧИСТКИ
# =========================================================================

def clean_html_content(html_content):
    """Многоэтапная очистка HTML перед извлечением текста."""
    soup = BeautifulSoup(html_content, "html.parser")

    elements_to_remove = [
        "nav", "script", "style", "noscript",
        "footer", "header", "aside",
        ".infobox", ".infobox-table", ".sidebar",
        ".toc", ".toclimit-2", "#toc",
        ".mw-editsection",
        ".noprint", ".reference", ".reflist",
        ".catlinks", ".mw-jump-link",
        ".mbox", ".navigation-not-searchable",
        ".thumb", ".tright", ".tleft",
        ".gallery", ".image", "figure",
        ".metadata", ".hatnote",
        ".navbox", ".navbox-styles",
        ".shortdescription", ".dablink",
        ".printfooter", ".mw-authority-control",
        ".external", "audio", "video",
        "table.wikitable",
        ".mw-empty-elt",
        "sup.reference",
        ".sistersitebox",
        ".mw-collapsible",
        ".collapsible",
    ]

    for selector in elements_to_remove:
        try:
            for element in soup.select(selector):
                element.decompose()
        except Exception:
            pass

    headings_to_remove = []
    for heading_tag in soup.find_all(["h2", "h3"]):
        heading_text = heading_tag.get_text(strip=True)
        heading_text_clean = re.sub(r'\[edit\]|\[source\]', '', heading_text).strip()
        if heading_text_clean in SECTION_HEADERS_TO_REMOVE:
            headings_to_remove.append(heading_tag)

    for heading in headings_to_remove:
        try:
            heading.decompose()
        except Exception:
            pass

    for hidden in soup.select("[style*='display:none'], [style*='display: none']"):
        hidden.decompose()

    content = soup.select_one("#mw-content-text .mw-parser-output")
    if not content:
        content = soup.select_one("#mw-content-text") or soup.body
    if not content:
        return ""

    for element in content.find_all(string=True):
        parent = element.parent
        if not parent:
            continue
        text = element.strip()
        if re.match(r'^&lt;.*&gt;$', text) or re.match(r'^<.*>$', text):
            element.replace_with("")

    text = content.get_text(separator="\n", strip=True)
    return text


def is_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for regex in JUNK_REGEXES:
        if regex.search(stripped):
            return True
    return False


def is_infobox_label(line: str) -> bool:
    stripped = line.strip()
    for regex in INFOBOX_REGEXES:
        if regex.match(stripped):
            return True
    return False


def is_empty_or_special_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.match(r'^\s*\d+\s*$', stripped):
        return True
    if re.match(r'^\s*[\d\W_]+\s*$', stripped):
        return True
    if re.match(r'^\s*[A-Za-z]\s*$', stripped):
        return True
    return False


def apply_cutoff_markers(text: str) -> str:
    lines = text.split('\n')
    earliest_cutoff = len(lines)
    for i, line in enumerate(lines):
        for marker in CUTOFF_MARKERS:
            if marker.lower() in line.lower():
                if i < earliest_cutoff:
                    earliest_cutoff = i
                break
    if earliest_cutoff < len(lines):
        lines = lines[:earliest_cutoff]
    return '\n'.join(lines)


def normalize_paragraphs(text: str) -> str:
    """
    Склеивание разорванных строк обратно в абзацы.
    Правила:
    - Если строка не начинается с заглавной буквы/цифры/спецсимвола,
      она продолжает предыдущую.
    - Если предыдущая строка не заканчивается на .!?:"»'')]},
      она продолжается следующей.
    - Строки, начинающиеся с # (заголовки), всегда на новой строке.
    - Пустые строки остаются разделителями абзацев.
    """
    lines = text.split('\n')
    merged = []
    buffer = ""

    for line in lines:
        stripped = line.strip()

        # Пустая строка — завершаем абзац
        if not stripped:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append("")
            continue

        # Заголовки — всегда отдельно
        if stripped.startswith('#'):
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(line)
            continue

        # Элементы списков (начинаются с - * • ▪ ▸) — отдельно
        if stripped[0] in '-*•▪▸○●⇒→':
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(line)
            continue

        if not buffer:
            buffer = line
            continue

        # Проверка: предыдущая строка заканчивается знаком конца предложения?
        prev_ends = buffer.rstrip()[-1:] if buffer.rstrip() else ''
        sentence_endings = set('.!?:"»' + "'" + ')]},')

        # Проверка: текущая строка начинается с заглавной буквы?
        curr_starts_capital = stripped[0].isupper() if stripped else False
        curr_starts_quote = stripped[0] in '"\'«“' if stripped else False

        if prev_ends in sentence_endings and (curr_starts_capital or curr_starts_quote):
            # Новое предложение — начинаем новую строку
            merged.append(buffer)
            buffer = line
        else:
            # Продолжение предыдущей строки — склеиваем через пробел
            buffer = buffer + " " + line

    if buffer:
        merged.append(buffer)

    return '\n'.join(merged)


def deep_clean_extracted_text(text: str) -> str:
    """Финальная многоэтапная очистка."""
    if not text:
        return ""

    # Этап A: отсечение по маркерам
    text = apply_cutoff_markers(text)

    # Этап B: фильтрация мусорных строк
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        if is_junk_line(stripped):
            continue
        if is_infobox_label(stripped):
            continue
        if is_empty_or_special_line(stripped):
            continue
        if re.match(r'^https?://', stripped):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Этап C: удаление backup link и стрелок
    text = re.sub(r'^\s*↑[^\n]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'↑[^\n]*\(backup link\)[^\n]*', '', text)
    text = re.sub(r'\n\s*↑.*', '', text)

    # Этап D: удаление URL
    text = re.sub(r'https?://[^\s]+', '', text)

    # Этап E: удаление [edit], [source], [number]
    text = re.sub(r'\[edit\]|\[source\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)

    # Этап F: артефакты Fandom
    text = re.sub(r'\(First identified as [^\)]+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Picture only\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbackup link\b', '', text, flags=re.IGNORECASE)

    # Этап G: HTML-сущности
    for entity, replacement in HTML_ENTITIES.items():
        text = text.replace(entity, replacement)

    # === НОВЫЙ ЭТАП H: нормализация абзацев ===
    text = normalize_paragraphs(text)

    # Этап I: сжатие пустых строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\n+', '', text)
    text = re.sub(r'\n+$', '', text)
    text = text.strip()

    return text


def fetch_page(page_name):
    url = BASE_URL + page_name
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_pages = len(PAGES)
    success_count = 0
    skip_count = 0
    error_count = 0
    short_count = 0

    for idx, page in enumerate(PAGES, 1):
        safe_name = page.replace("/", "_").replace("(", "").replace(")", "")
        filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.md")

        if os.path.exists(filepath):
            print(f"[{idx}/{total_pages}] [SKIP] {safe_name}")
            skip_count += 1
            continue

        try:
            print(f"[{idx}/{total_pages}] [FETCH] {page}...")
            html = fetch_page(page)
            raw_text = clean_html_content(html)
            cleaned_text = deep_clean_extracted_text(raw_text)

            if len(cleaned_text) > 500:
                title = safe_name.replace("_", " ")
                full_content = f"# {title}\n\n{cleaned_text}"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(full_content)
                print(f"         [OK] {safe_name}: {len(cleaned_text)} символов")
                success_count += 1
            else:
                print(f"         [SHORT] {safe_name}: {len(cleaned_text)} символов")
                short_count += 1

            time.sleep(1)

        except Exception as e:
            print(f"         [ERROR] {page}: {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print(f"ГОТОВО: успешно={success_count}, пропущено={skip_count}, "
          f"коротких={short_count}, ошибок={error_count}")


if __name__ == "__main__":
    main()