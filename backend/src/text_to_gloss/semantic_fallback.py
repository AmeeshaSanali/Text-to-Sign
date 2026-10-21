"""
Semantic Fallback Resolver & Compound Decomposer
Part of the 4-Tier Hierarchical Bilingual OOV Architecture

Implements:
- Tier 2: Semantic Subsumption / Near-Synonym & Hypernym Fallback with confidence scoring
- Tier 3: Compositional / Compound Sign Decomposition for complex multi-concept tokens
- Graceful degradation returning structured ResolutionResult metadata
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel


class ResolutionResult(BaseModel):
    """Encapsulates the resolution outcome for an input token."""
    glosses: List[str]
    match_type: str  # "exact", "semantic_approximation", "compound", "fingerspelled"
    confidence: float
    note: Optional[str] = None


# Tier 3: Compositional / Compound Sign Decomposition Registry
# Maps concepts that do not exist as single signs into pairs/tuples of atomic canonical signs
COMPOUND_DECOMPOSITIONS: Dict[str, List[str]] = {
    # Medical & Facilities
    "hospital": ["DOCTOR", "PLACE"],
    "clinic": ["DOCTOR", "PLACE"],
    "රෝහල": ["DOCTOR", "PLACE"],
    "සායනය": ["DOCTOR", "PLACE"],
    "ඉස්පිරිතාලෙ": ["DOCTOR", "PLACE"],

    # Education
    "teacher": ["TEACH", "PERSON"],
    "instructor": ["TEACH", "PERSON"],
    "tutor": ["TEACH", "PERSON"],
    "ගුරුවරයා": ["TEACH", "PERSON"],
    "ගුරුතුමා": ["TEACH", "PERSON"],
    "student": ["LEARN", "PERSON"],
    "pupil": ["LEARN", "PERSON"],
    "ශිෂ්‍යයා": ["LEARN", "PERSON"],
    "library": ["BOOK", "PLACE"],
    "පුස්තකාලය": ["BOOK", "PLACE"],
    "classroom": ["SCHOOL", "ROOM"],
    "පන්ති කාමරය": ["SCHOOL", "ROOM"],

    # Meals
    "breakfast": ["MORNING", "EAT"],
    "උදේ කෑම": ["MORNING", "EAT"],
    "lunch": ["TODAY", "EAT"],
    "දවල් කෑම": ["TODAY", "EAT"],
    "dinner": ["NIGHT", "EAT"],
    "රෑ කෑම": ["NIGHT", "EAT"],

    # Rooms & Domestic
    "bedroom": ["SLEEP", "ROOM"],
    "නිදන කාමරය": ["SLEEP", "ROOM"],
    "dining room": ["EAT", "ROOM"],
    "කෑම කාමරය": ["EAT", "ROOM"],
    "bookstore": ["BOOK", "STORE"],
    "පොත් සාප්පුව": ["BOOK", "STORE"],

    # Family Compounds
    "grandfather": ["OLD", "FATHER"],
    "grandpa": ["OLD", "FATHER"],
    "සීයා": ["OLD", "FATHER"],
    "grandmother": ["OLD", "MOTHER"],
    "grandma": ["OLD", "MOTHER"],
    "ආච්චි": ["OLD", "MOTHER"],

    # Occupations
    "policeman": ["POLICE", "PERSON"],
    "police officer": ["POLICE", "PERSON"],
    "පොලිස් නිලධාරියා": ["POLICE", "PERSON"],
}


# Tier 2: Semantic Subsumption & Hypernym / Synonym Taxonomy
# Maps near-synonyms, hyponyms, or taxonomic categories to valid canonical signs with confidence score
SEMANTIC_TAXONOMY: Dict[str, Tuple[str, float, str]] = {
    # Medical
    "physician": ("DOCTOR", 0.92, "Near-synonym: physician -> DOCTOR"),
    "medic": ("DOCTOR", 0.90, "Near-synonym: medic -> DOCTOR"),
    "specialist": ("DOCTOR", 0.85, "Near-synonym: specialist -> DOCTOR"),
    "දොස්තර": ("DOCTOR", 0.95, "Sinhala synonym: දොස්තර -> DOCTOR"),
    "වෙද": ("DOCTOR", 0.88, "Sinhala root: වෙද -> DOCTOR"),

    # Transport & Vehicles
    "car": ("VEHICLE", 0.88, "Hypernym: car -> VEHICLE"),
    "automobile": ("VEHICLE", 0.88, "Hypernym: automobile -> VEHICLE"),
    "van": ("VEHICLE", 0.85, "Hypernym: van -> VEHICLE"),
    "lorry": ("VEHICLE", 0.85, "Hypernym: lorry -> VEHICLE"),
    "truck": ("VEHICLE", 0.85, "Hypernym: truck -> VEHICLE"),
    "කාර්": ("VEHICLE", 0.88, "Sinhala loanword: කාර් -> VEHICLE"),
    "ලොරි": ("VEHICLE", 0.85, "Sinhala loanword: ලොරි -> VEHICLE"),

    # Family
    "dad": ("FATHER", 0.98, "Synonym: dad -> FATHER"),
    "daddy": ("FATHER", 0.96, "Synonym: daddy -> FATHER"),
    "papa": ("FATHER", 0.95, "Synonym: papa -> FATHER"),
    "mom": ("MOTHER", 0.98, "Synonym: mom -> MOTHER"),
    "mommy": ("MOTHER", 0.96, "Synonym: mommy -> MOTHER"),
    "mama": ("MOTHER", 0.95, "Synonym: mama -> MOTHER"),

    # Emotions & States
    "glad": ("HAPPY", 0.90, "Near-synonym: glad -> HAPPY"),
    "delighted": ("HAPPY", 0.88, "Near-synonym: delighted -> HAPPY"),
    "cheerful": ("HAPPY", 0.88, "Near-synonym: cheerful -> HAPPY"),
    "joyful": ("HAPPY", 0.88, "Near-synonym: joyful -> HAPPY"),
    "සතුටුයි": ("HAPPY", 0.90, "Sinhala inflection: සතුටුයි -> HAPPY"),
    "සතුට": ("HAPPY", 0.90, "Sinhala noun: සතුට -> HAPPY"),
    "unhappy": ("SAD", 0.90, "Near-synonym: unhappy -> SAD"),
    "depressed": ("SAD", 0.85, "Near-synonym: depressed -> SAD"),
    "sorrowful": ("SAD", 0.85, "Near-synonym: sorrowful -> SAD"),
    "දුකයි": ("SAD", 0.90, "Sinhala inflection: දුකයි -> SAD"),

    # Movement & Speed
    "quick": ("FAST", 0.92, "Synonym: quick -> FAST"),
    "swift": ("FAST", 0.90, "Synonym: swift -> FAST"),
    "rapid": ("FAST", 0.90, "Synonym: rapid -> FAST"),
    "ඉක්මන්": ("FAST", 0.90, "Sinhala adjective: ඉක්මන් -> FAST"),
    "slowly": ("SLOW", 0.95, "Adverb derivation: slowly -> SLOW"),

    # Domestic & Entities
    "residence": ("HOME", 0.88, "Hypernym: residence -> HOME"),
    "house": ("HOME", 0.95, "Synonym: house -> HOME"),
    "shop": ("STORE", 0.95, "Synonym: shop -> STORE"),
    "boutique": ("STORE", 0.88, "Near-synonym: boutique -> STORE"),
    "කඩේ": ("STORE", 0.90, "Sinhala noun: කඩේ -> STORE"),
    "සාප්පුව": ("STORE", 0.90, "Sinhala noun: සාප්පුව -> STORE"),

    # Actions
    "examine": ("LOOK", 0.82, "Hypernym: examine -> LOOK"),
    "examined": ("LOOK", 0.82, "Hypernym: examined -> LOOK"),
    "inspect": ("LOOK", 0.82, "Hypernym: inspect -> LOOK"),
    "inspected": ("LOOK", 0.82, "Hypernym: inspected -> LOOK"),
    "view": ("LOOK", 0.90, "Synonym: view -> LOOK"),
    "watch": ("LOOK", 0.90, "Synonym: watch -> LOOK"),
    "speak": ("TELL", 0.88, "Near-synonym: speak -> TELL"),
    "talk": ("TELL", 0.88, "Near-synonym: talk -> TELL"),
    "chat": ("TELL", 0.85, "Near-synonym: chat -> TELL"),
    "hear": ("LISTEN", 0.92, "Near-synonym: hear -> LISTEN"),
    "පරීක්ෂා": ("LOOK", 0.85, "Sinhala synonym: පරීක්ෂා -> LOOK"),
    "පරීක්ෂණය": ("LOOK", 0.85, "Sinhala noun: පරීක්ෂණය -> LOOK"),
}


class SemanticFallbackResolver:
    """Resolves out-of-vocabulary words using Tiers 2 & 3 of the bilingual architecture."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def resolve(self, token_lower: str) -> Optional[ResolutionResult]:
        """
        Attempts to resolve an unmapped token through:
        1. Compound Sign Decomposition (Tier 3)
        2. Semantic Subsumption & Hypernym/Synonym Matching (Tier 2)
        """
        token = token_lower.strip()

        # Check Tier 3: Compound decomposition
        if token in COMPOUND_DECOMPOSITIONS:
            seq = COMPOUND_DECOMPOSITIONS[token]
            return ResolutionResult(
                glosses=seq,
                match_type="compound",
                confidence=0.85,
                note=f"Decomposed compound concept '{token}' -> {' + '.join(seq)}",
            )

        # Check Tier 2: Semantic Subsumption
        if token in SEMANTIC_TAXONOMY:
            target_gloss, conf, note = SEMANTIC_TAXONOMY[token]
            if conf >= self.confidence_threshold:
                return ResolutionResult(
                    glosses=[target_gloss],
                    match_type="semantic_approximation",
                    confidence=conf,
                    note=note,
                )

        return None
