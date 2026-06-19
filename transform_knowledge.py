"""
Скрипт: transform_knowledge.py
Назначение: замена всех канонических терминов Star Wars на вымышленные,
            сохранение очищенной базы знаний и словаря замен.
Требования: pip install faker
"""

import json
import os
import re
from pathlib import Path
from faker import Faker

fake = Faker()
Faker.seed(42)  # Для воспроизводимости

# ============================================================================
# Словарь замен: исходный термин → вымышленный термин
# ============================================================================
TERMS_MAP = {
    # --- Персонажи ---
    "Darth Vader": "Xarn Velgor",
    "Anakin Skywalker": "Xarn Velgor",  # Альтернативное имя
    "Vader": "Velgor",
    "Luke Skywalker": "Joren Kast",
    "Luke": "Joren",
    "Skywalker": "Kast",
    "Leia Organa": "Mira Venn",
    "Princess Leia": "Warden Mira",
    "Leia": "Mira",
    "Organa": "Venn",
    "Han Solo": "Dax Corbin",
    "Han": "Dax",
    "Solo": "Corbin",
    "Obi-Wan Kenobi": "Zeth Malkor",
    "Obi-Wan": "Zeth",
    "Ben Kenobi": "Zeth Malkor",
    "Kenobi": "Malkor",
    "Emperor Palpatine": "Overlord Draven Nul",
    "Palpatine": "Draven Nul",
    "Emperor": "Overlord",
    "Darth Sidious": "Draven Nul",
    "Yoda": "Vorn",
    "Grand Master Yoda": "Grand Master Vorn",
    "Chewbacca": "Gorath",
    "Chewie": "Gor",
    "R2-D2": "DR-7X",
    "Artoo": "Drex",
    "C-3PO": "LQ-9M",
    "Threepio": "Loquim",
    "Boba Fett": "Kael Ryn",
    "Mace Windu": "Taron Vale",
    "Windu": "Vale",
    
    # --- Планеты ---
    "Tatooine": "Vorath Prime",
    "Coruscant": "Nexara City-world",
    "Coruscant's": "Nexara's",
    "Hoth": "Kryos",
    "Endor": "Veldara",
    "Dagobah": "Myrkon",
    "Alderaan": "Elara",
    "Alderaan's": "Elara's",
    "Naboo": "Aqualis",
    "Kamino": "Cygnar",
    
    # --- Технологии ---
    "Death Star": "Void Core",
    "Death Star's": "Void Core's",
    "lightsaber": "flux blade",
    "lightsabers": "flux blades",
    "Lightsaber": "Flux blade",
    "Lightsabers": "Flux blades",
    "Millennium Falcon": "Shadow Hawk",
    "Falcon": "Shadow Hawk",
    "TIE Fighter": "Razor Drone",
    "TIE Fighters": "Razor Drones",
    "TIE fighter": "Razor drone",
    "TIE fighters": "Razor drones",
    "X-wing": "Starhawk",
    "X-wings": "Starhawks",
    "X-wing starfighter": "Starhawk fighter",
    "Star Destroyer": "Dominion Cruiser",
    "Star Destroyers": "Dominion Cruisers",
    "AT-AT": "Colossus Strider",
    "AT-ATs": "Colossus Striders",
    "AT-AT walker": "Colossus Strider",
    
    # --- Организации ---
    "Jedi Order": "Keepers of the Flux",
    "Jedi": "Keeper",
    "Jedis": "Keepers",
    "the Jedi": "the Keepers",
    "Sith": "Shade Covenant",
    "the Sith": "the Shade Covenant",
    "Sith Lord": "Shade Lord",
    "Sith Lords": "Shade Lords",
    "Galactic Empire": "Solar Dominion",
    "the Empire": "the Dominion",
    "Empire's": "Dominion's",
    "Imperial": "Dominion",
    "Rebel Alliance": "Free Worlds Coalition",
    "Rebels": "Coalition forces",
    "the Rebellion": "the Coalition",
    "Rebel": "Coalition",
    
    # --- Расы ---
    "Wookiee": "Thoran",
    "Wookiees": "Thorans",
    "Ewok": "Moorling",
    "Ewoks": "Moorlings",
    "Clone Troopers": "Gen-Soldiers",
    "Clone trooper": "Gen-Soldier",
    "clone trooper": "gen-soldier",
    "clones": "gen-soldiers",
    "Stormtroopers": "Legion Vanguard",
    "Stormtrooper": "Legion Vanguard",
    "stormtroopers": "legion vanguard",
    "stormtrooper": "legion vanguard",
    
    # --- Концепты ---
    "the Force": "the Synth Flux",
    "The Force": "The Synth Flux",
    "Force": "Flux",
    "Force-sensitive": "Flux-sensitive",
    "Force powers": "Flux abilities",
    "dark side": "deep flux",
    "dark side of the Force": "deep flux of the Synth",
    "light side": "bright flux",
    "light side of the Force": "bright flux of the Synth",
    
    # --- Прочее ---
    "lightspeed": "hyperlight",
    "hyperspace": "voidspace",
    "droids": "automatons",
    "droid": "automaton",
    "galaxy": "stellar realm",
    "Galactic Republic": "Stellar Concordium",
    "Republic": "Concordium",
    "Clone Wars": "Gene Wars",
    "Star Wars": "Flux Wars",
    "Rebel": "Coalition",
    "Imperial Senate": "Dominion Assembly",
    "stormtrooper armor": "legion vanguard armor",
    "blaster": "pulse rifle",
    "blasters": "pulse rifles",
}

def apply_transformations(text, terms_map):
    """
    Применение замен с учётом регистра и границ слов.
    Многоэтапная замена: сначала длинные фразы, затем короткие.
    """
    # Сортировка по длине термина (от длинных к коротким) для избежания пересечений
    sorted_terms = sorted(terms_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, replacement in sorted_terms:
        # Замена с учётом регистра: точное совпадение
        text = re.sub(
            r'\b' + re.escape(original) + r'\b',
            replacement,
            text
        )
    
    return text

def create_knowledge_base(raw_dir="raw_docs/", output_dir="knowledge_base/", terms_map=TERMS_MAP):
    """Создание финальной базы знаний с заменёнными терминами."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    raw_files = list(Path(raw_dir).glob("*.txt"))
    transformed_count = 0
    
    for filepath in raw_files:
        with open(filepath, "r", encoding="utf-8") as f:
            original_text = f.read()
        
        # Применение замен
        transformed_text = apply_transformations(original_text, terms_map)
        
        # Сохранение результата
        output_name = filepath.stem
        output_path = Path(output_dir) / f"{output_name}.md"
        
        with open(output_path, "w", encoding="utf-8") as f:
            # Добавление заголовка
            title = output_name.replace("_", " ")
            f.write(f"# {title}\n\n")
            f.write(transformed_text)
        
        transformed_count += 1
        print(f"[TRANSFORMED] {output_name} ({len(transformed_text)} chars)")
    
    print(f"\nTotal documents transformed: {transformed_count}")
    
    # Сохранение словаря замен
    terms_map_path = Path(output_dir) / "terms_map.json"
    with open(terms_map_path, "w", encoding="utf-8") as f:
        json.dump(terms_map, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] terms_map.json ({len(terms_map)} entries)")

if __name__ == "__main__":
    create_knowledge_base()