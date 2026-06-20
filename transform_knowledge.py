"""
transform_knowledge.py (v5)
Назначение: замена всех канонических терминов Star Wars на вымышленные.
Исправления v5: расширенный словарь (160+ терминов).

Требования: pip install faker
"""

import json
import re
from pathlib import Path

# ============================================================================
# ПОЛНЫЙ СЛОВАРЬ ЗАМЕН (исходное → вымышленное)
# ============================================================================
TERMS_MAP = {
    # --- Персонажи ---
    "Darth Vader": "Xarn Velgor",
    "Anakin Skywalker": "Xarn Velgor",
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

    # Дополнительные персонажи
    "Qui-Gon Jinn": "Toran Sol",
    "Qui-Gon": "Toran",
    "Jinn": "Sol",
    "Galen Walton Erso": "Viktor Strayn",
    "Galen Erso": "Viktor Strayn",
    "Erso": "Strayn",
    "Orson Krennic": "Aldric Voss",
    "Krennic": "Voss",
    "Firmus Piett": "Corbin Drax",
    "Admiral Piett": "Admiral Drax",
    "Piett": "Drax",
    "Jango Fett": "Ryn Kar",
    "Fett": "Ryn",
    "Poe Dameron": "Reeve Janar",
    "Dameron": "Janar",
    "Saw Gerrera": "Tarn Vossk",
    "Gerrera": "Vossk",
    "Shaef Corssin": "Jorik Thane",
    "Corssin": "Thane",
    "Biggs Darklighter": "Cade Loran",
    "Darklighter": "Loran",
    "Calrissian": "Drev Kass",
    "Lando Calrissian": "Drev Kass",
    "Doctor Cylo-V": "Scientist Vex-9",
    "Cylo-V": "Vex-9",
    "Naga Sadow": "Vorlak Syn",
    "Sadow": "Syn",
    "Aphra": "Lyra Venn",
    "Doctor Aphra": "Lyra Venn",
    "Tarkin": "Varnus",
    "Governor Tarkin": "Governor Varnus",
    "Wilhuff Tarkin": "Wilhuff Varnus",

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
    "Scarif": "Tyrak",
    "Geonosis": "Xarnak",
    "Geonosians": "Xarnak Hive",
    "Geonosian": "Xarnak",
    "Jedha": "Zelara",
    "NaJedha": "Zelara Minor",
    "Kijimi": "Voska",
    "Pasaana": "Ardath",
    "Crait": "Solara",
    "Jakku": "Vorath Minor",
    "Yavin 4": "Myrkon 4",
    "Yavin": "Myrkon",

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
    "Profundity": "Dauntless",
    "Devastator": "Obliterator",
    "Executor": "Imperator",
    "Ark Angel": "Void Wraith",
    "kyber crystal": "void crystal",
    "kyber crystals": "void crystals",
    "ootheca": "gen-capsule",
    "Quadex power core": "Nexus power core",
    "sensor dish": "scanner array",
    "J-Type 327 Nubian": "T-327 Starcraft",

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
    "Mandalorian": "Vraxian",
    "Mandalorians": "Vraxians",
    "Massassi": "Vorlak",
    "Bothan": "Zentari",
    "Bothans": "Zentari",

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
    "Rogue One": "Shadow Cell",
    "Battle of Crait": "Battle of Solara",
    "First Order": "Nexus Command",
    "Great Hyperspace War": "First Flux War",
    "Great Shade Covenant Wars": "Shade Covenant Wars",
}


def apply_transformations(text, terms_map):
    """
    Применение замен с учётом регистра и границ слов.
    Сортировка по длине термина для избежания пересечений.
    """
    sorted_terms = sorted(terms_map.items(), key=lambda x: len(x[0]), reverse=True)

    for original, replacement in sorted_terms:
        text = re.sub(
            r'\b' + re.escape(original) + r'\b',
            replacement,
            text
        )

    return text


def create_knowledge_base(raw_dir="raw_docs/", output_dir="knowledge_base/"):
    """Создание финальной базы знаний с заменёнными терминами."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    raw_files = list(Path(raw_dir).glob("*.md"))
    transformed_count = 0

    for filepath in raw_files:
        with open(filepath, "r", encoding="utf-8") as f:
            original_text = f.read()

        transformed_text = apply_transformations(original_text, TERMS_MAP)

        output_name = filepath.stem
        output_path = Path(output_dir) / f"{output_name}.md"

        with open(output_path, "w", encoding="utf-8") as f:
            title = output_name.replace("_", " ")
            f.write(f"# {title}\n\n")
            f.write(transformed_text)

        transformed_count += 1
        print(f"[TRANSFORMED] {output_name} ({len(transformed_text)} chars)")

    print(f"\nTotal documents transformed: {transformed_count}")

    # Сохранение словаря
    terms_map_path = Path(output_dir) / "terms_map.json"
    with open(terms_map_path, "w", encoding="utf-8") as f:
        json.dump(TERMS_MAP, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] terms_map.json ({len(TERMS_MAP)} entries)")


if __name__ == "__main__":
    create_knowledge_base()