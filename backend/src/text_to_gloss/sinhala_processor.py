"""
Sinhala NLP Processor & Dactylology Engine
Part of the 4-Tier Hierarchical Bilingual OOV Architecture

Provides:
- Sinhala script detection and normalization (\u0D80-\u0DFF)
- Morphological stemmer: Suffix-stripping for noun case markers and verb inflection endings
- Stopword filtering for grammatical particles
- Canonical Sinhala-to-Gloss lexicon mapping to SLSL gloss concepts
- Sinhala Dactylology (Manual Alphabet / සිංහල අත්අකුරු හෝඩිය)
- SOV & Time-adverbial fronting grammar parser
"""

import re
from typing import Dict, List, Optional, Set, Tuple


# Sinhala Unicode Block: U+0D80 - U+0DFF
SINHALA_UNICODE_PATTERN = re.compile(r"[\u0d80-\u0dff]")

# Grammatical particles, copulas, and conjunctions omitted in sign language glossing
SINHALA_STOPWORDS: Set[str] = {
    "සහ", "හෝ", "නමුත්", "නම්", "ද", "ගැන", "සඳහා", "වැනි", "මෙන්",
    "නැත", "නොවේ", "ඇත", "වෙත", "මගින්", "විසින්", "පිළිබඳ", "තුළ",
    "එම", "මෙම", "ඒ", "මේ", "ඔහුගේ", "ඇයගේ", "කළා", "කලා", "කරනවා", "කළේ", "කලේ"
}

# Sinhala noun case suffixes (sorted longest first to greedily match composite affixes)
# Suffixes: -වලින් / -වල් / -වල (plural/genitive), -යේදී / -ේදී / -දී (locative), -ගෙන් (ablative), -ගේ (genitive), -කට / -ක / -ට (dative/locative), -ව (accusative), -න් (plural/instrumental), -ක් / -ෙක් (indefinite)
NOUN_CASE_SUFFIXES: List[str] = [
    "වලින්", "වලදී", "යේදී", "ේදී", "ගෙන්", "වල්", "වල", "කදී", "ගේ", "දී", "කට", "ක", "ට", "ව", "න්", "ක්", "ෙක්"
]

# Sinhala verb inflection suffixes (present, past, continuous, infinitive, optative)
# Examples: කනවා -> stem කන, කෑවා -> stem කන (via irregular map), කමින්, කන්න
VERB_INFLECTION_SUFFIXES: List[str] = [
    "නවාද", "න්නද", "න්නෙ", "න්නේ", "නවා", "මින්", "න්න", "ලා", "වා", "වේ", "මු", "මි", "ති"
]

# Irregular past-tense and phonological shifts in Sinhala verbs
IRREGULAR_VERB_MAP: Dict[str, str] = {
    "කෑවා": "කන",
    "කෑව": "කන",
    "කෑවේ": "කන",
    "කෑ": "කන",
    "ගියා": "යන",
    "ගියේ": "යන",
    "ගිහින්": "යන",
    "ආවා": "එන",
    "ආවේ": "එන",
    "ඇවිත්": "එන",
    "බැලුවා": "බලන",
    "බැලුවේ": "බලන",
    "දුන්නා": "දෙන",
    "ගත්තා": "ගන්න",
    "බිව්වා": "බොන",
    "නිදාගත්තා": "නිදාගන්න",
    "හිටියා": "ඉන්න",
    "කළා": "කරන",
    "කලා": "කරන",
}

# Irregular plural / inflected nouns
IRREGULAR_NOUN_MAP: Dict[str, str] = {
    "අම්මාව": "අම්මා",
    "අම්මාට": "අම්මා",
    "අම්මාගේ": "අම්මා",
    "තාත්තාව": "තාත්තා",
    "තාත්තාට": "තාත්තා",
    "තාත්තාගේ": "තාත්තා",
    "රෝහලේදී": "රෝහල",
    "රෝහලට": "රෝහල",
    "රෝහලේ": "රෝහල",
    "පාසලේදී": "පාසල",
    "පාසලට": "පාසල",
    "පාසලේ": "පාසල",
    "ගෙදරදී": "ගෙදර",
    "ගෙදරට": "ගෙදර",
    "ගෙයක්": "ගෙදර",
    "පොතක්": "පොත",
    "මලක්": "මල",
    "ගසක්": "ගස",
    "ළමයෙක්": "ළමයා",
    "ළමය": "ළමයා",
    "මිනිසෙක්": "මිනිසා",
    "මට": "මම",
    "මාව": "මම",
    "ඔයාට": "ඔයා",
    "ඔබට": "ඔබ",
    "අපිට": "අපි",
}

# Core Sinhala-to-Gloss lexicon (Base lemmas mapped to canonical SLSL gloss tokens)
SINHALA_TO_GLOSS: Dict[str, str] = {
    # Pronouns & Possessives
    "මම": "I",
    "මා": "I",
    "මගේ": "MY",
    "අපි": "WE",
    "අපේ": "OUR",
    "ඔයා": "YOU",
    "ඔබ": "YOU",
    "ඔයාගේ": "YOUR",
    "ඔබගේ": "YOUR",
    "ඔයාලා": "YOU",
    "ඔහු": "HE",
    "ඔහුගේ": "HIS",
    "ඇය": "SHE",
    "ඇයගේ": "HER",
    "ඔවුන්": "THEY",
    "ඔවුන්ගේ": "THEIR",
    "එය": "IT",
    # Greetings & Common Social
    "ආයුබෝවන්": "HELLO",
    "ආයුබොවන්": "HELLO",
    "සුබ උදෑසනක්": "HELLO",
    "සුබ දවසක්": "HELLO",
    "ස්තූතියි": "THANK_YOU",
    "බොහොම ස්තූතියි": "THANK_YOU",
    "ස්තුතියි": "THANK_YOU",
    "කරුණාකර": "PLEASE",
    "කරුණාකරලා": "PLEASE",
    "සමාවෙන්න": "SORRY",
    "ඔව්": "YES",
    "එසේය": "YES",
    "නැහැ": "NO",
    "නෑ": "NO",
    "එපා": "NO",
    "හරි": "YES",
    "හොඳයි": "GOOD",

    # Pronouns
    "මම": "I",
    "මා": "I",
    "අපි": "WE",
    "ඔයා": "YOU",
    "ඔබ": "YOU",
    "ඔයාලා": "YOU",
    "ඔහු": "HE",
    "ඇය": "SHE",
    "ඔවුන්": "THEY",
    "එය": "IT",

    # Family & People
    "අම්මා": "MOTHER",
    "මව": "MOTHER",
    "තාත්තා": "FATHER",
    "පියා": "FATHER",
    "අයියා": "BROTHER",
    "මල්ලී": "BROTHER",
    "අක්කා": "SISTER",
    "නංගී": "SISTER",
    "මිතුරා": "FRIEND",
    "යහළුවා": "FRIEND",
    "ළමයා": "CHILD",
    "දරුවා": "CHILD",
    "මිනිසා": "PERSON",
    "පුද්ගලයා": "PERSON",
    "ගැහැණිය": "WOMAN",

    # Verbs (Base Stems)
    "කන": "EAT",
    "ක": "EAT",
    "කෑම": "EAT",
    "බොන": "DRINK",
    "බො": "DRINK",
    "යන": "GO",
    "ය": "GO",
    "එන": "COME",
    "එ": "COME",
    "දකින": "SEE",
    "බලන": "LOOK",
    "බල": "LOOK",
    "දෙන": "GIVE",
    "දෙ": "GIVE",
    "ගන්න": "TAKE",
    "ගන්": "TAKE",
    "උදව්": "HELP",
    "කරන": "DO",
    "හදන": "MAKE",
    "හද": "MAKE",
    "කියන": "TELL",
    "අහන": "LISTEN",
    "නිදාගන්න": "SLEEP",
    "ඉගෙනගන්න": "STUDY",
    "පාඩම්": "STUDY",
    "උගන්වන": "TEACH",
    "දන්න": "I_KNOW",
    "ලියන": "WRITE",
    "ලිය": "WRITE",
    "කියවන": "STUDY",
    "කියව": "STUDY",
    "වැඩ": "WORK",

    # Time Markers
    "අද": "TODAY",
    "ඊයේ": "YESTERDAY",
    "හෙට": "TOMORROW",
    "දැන්": "NOW",
    "උදේ": "MORNING",
    "දවල්": "AFTERNOON",
    "රෑ": "NIGHT",
    "සවස": "EVENING",
    "දවස": "DAY",
    "සතිය": "WEEK",
    "මාසය": "MONTH",
    "අවුරුද්ද": "YEAR",

    # Colors
    "නිල්": "BLUE",
    "රතු": "RED",
    "කහ": "YELLOW",
    "කොළ": "GREEN",
    "සුදු": "WHITE",
    "කළු": "BLACK",
    "තැඹිලි": "ORANGE",

    # Common Nouns & Places
    "ගෙදර": "HOUSE",
    "නිවස": "HOUSE",
    "ගෙය": "HOUSE",
    "පාසල": "SCHOOL",
    "ඉස්කෝලේ": "SCHOOL",
    "ඉස්කෝලය": "SCHOOL",
    "වතුර": "WATER",
    "තේ": "TEA",
    "කිරි": "MILK",
    "පොත": "BOOK",
    "පොත්": "BOOK",
    "පෑන": "PENCIL",
    "පැන්සල": "PENCIL",
    "ගස": "TREE",
    "ගහ": "TREE",
    "ගස්": "TREE",
    "මල": "FLOWER",
    "මල්": "FLOWER",
    "ආහාර": "FOOD",
    "වාහනය": "VEHICLE",
    "බස්": "BUS",
    "බස් එක": "BUS",
    "කාරය": "CAR",
    "කාර්": "CAR",
    "කෝච්චිය": "TRAIN",
    "දුම්රිය": "TRAIN",
    "බයිසිකලය": "BICYCLE",
    "නගරය": "CITY",
    "කාමරය": "ROOM",
    "වෛද්‍යවරයා": "DOCTOR",
    "දොස්තර": "DOCTOR",
    "පොලිසිය": "POLICE",
    "පන්සල": "TEMPLE",
    "පල්ලිය": "CHURCH",
    "බැංකුව": "BANK",

    # Questions
    "මොකක්ද": "WHAT",
    "මොනවාද": "WHAT",
    "කොහෙද": "WHERE",
    "කවුද": "WHO",
    "කවදාද": "WHEN",
    "ඇයි": "WHY",
    "කොහොමද": "HOW",

    # Adjectives
    "හොඳ": "GOOD",
    "නරක": "BAD",
    "ලොකු": "BIG",
    "පොඩි": "SMALL",
    "ලස්සන": "BEAUTIFUL",
    "සතුටු": "HAPPY",
    "දුක": "SAD",
    "වේගවත්": "FAST",
    "අලුත්": "NEW",
    "පරණ": "OLD",
}

# Sinhala Time Fronting Markers
SINHALA_TIME_MARKERS: Set[str] = {
    "TODAY", "YESTERDAY", "TOMORROW", "NOW", "MORNING", "NIGHT", "AFTERNOON", "EVENING"
}

# Sinhala Question Markers
SINHALA_QUESTION_MARKERS: Set[str] = {
    "WHAT", "WHERE", "WHO", "WHEN", "WHY", "HOW"
}


def is_sinhala(text: str) -> bool:
    """Detect if the input string contains Sinhala Unicode characters."""
    return bool(SINHALA_UNICODE_PATTERN.search(text))


def stem_sinhala_word(word: str) -> str:
    """
    Morphological stemmer for Sinhala words.
    Handles irregular forms and strips common noun and verb suffixes.
    """
    word = word.strip()
    if not word:
        return ""

    # 1. Check irregular tables first
    if word in IRREGULAR_VERB_MAP:
        return IRREGULAR_VERB_MAP[word]
    if word in IRREGULAR_NOUN_MAP:
        return IRREGULAR_NOUN_MAP[word]

    # 2. Strip verb inflection suffixes
    for suf in sorted(VERB_INFLECTION_SUFFIXES, key=len, reverse=True):
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            base = word[:-len(suf)]
            # If base ends in pure consonant + hal-kirima, preserve readability
            if base in IRREGULAR_VERB_MAP:
                return IRREGULAR_VERB_MAP[base]
            # Standard Sinhala verb lemma ends in -න (e.g. කියව + න -> කියවන, බල + න -> බලන)
            lemma_candidate = base + "න"
            if lemma_candidate in SINHALA_TO_GLOSS or lemma_candidate in IRREGULAR_VERB_MAP:
                return lemma_candidate
            return base

    # 3. Strip noun case suffixes
    for suf in sorted(NOUN_CASE_SUFFIXES, key=len, reverse=True):
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            base = word[:-len(suf)]
            if base in IRREGULAR_NOUN_MAP:
                return IRREGULAR_NOUN_MAP[base]
            if base in SINHALA_TO_GLOSS:
                return base
            # Check if base + 'ා' matches (e.g. ළමය -> ළමයා)
            if (base + "ා") in SINHALA_TO_GLOSS or (base + "ා") in IRREGULAR_NOUN_MAP:
                return base + "ා"
            return base

    return word


def fingerspell_sinhala_token(word: str) -> str:
    """
    Generates Sinhala Dactylology manual alphabet representation token.
    Formats character-by-character gesture token: e.g. '#ක_ළ_ඹ'.
    """
    chars = [c for c in word if "\u0d80" <= c <= "\u0dff"]
    if not chars:
        return f"#{word.upper()}"
    return f"#{'_'.join(chars)}"
