"""
Mood classification and genre normalization for PlaylistGen.

MOODS               Maps canonical mood names to extensive keyword lists (500+ keywords).
GENRE_MOOD_FALLBACK Maps genre strings to moods for when no tag keywords match.
GENRE_NORMALIZE     Maps Last.fm tag strings to canonical iTunes-style genre names.

canonical_mood()    Convert a tag list + optional genre → one of the 10 canonical moods.
canonical_genre()   Convert a Last.fm tag string → a normalized genre name.
build_tag_counts()  Global tag frequency counter used for IDF weighting.
"""

import collections
import math
import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Mood keyword lists  (10 moods × 50+ keywords each)
# ---------------------------------------------------------------------------

MOODS: Dict[str, List[str]] = {
    "Happy": [
        "happy", "feel good", "feel-good", "cheerful", "uplifting", "joyful",
        "euphoric", "carefree", "celebratory", "summery", "summer", "bubblegum",
        "festive", "sunshine", "positive", "optimistic", "fun", "bright", "elated",
        "blissful", "playful", "lighthearted", "whimsical", "bouncy", "peppy",
        "lively", "vibrant", "exuberant", "gleeful", "jubilant", "merry", "jolly",
        "delightful", "glad", "content", "radiant", "giddy", "breezy", "airy",
        "sweet", "candy", "power pop", "upbeat", "good times", "summer vibes",
        "smile", "joyous", "ecstatic", "thrilled", "excited", "glee",
        "sunny", "springtime", "pop anthem", "good mood", "cheery",
        "feel great", "wonderful", "fantastic", "euphoria", "happiness",
        "carefree summer", "tropical", "beach", "festival pop", "indie pop",
    ],
    "Sad": [
        "sad", "melancholy", "melancholic", "heartbreak", "heartbroken", "somber",
        "grief", "lonely", "longing", "forlorn", "wistful", "bittersweet", "tragic",
        "depressing", "sorrowful", "mourning", "regret", "tearful", "tearjerker",
        "emotional", "vulnerable", "pain", "hurt", "loss", "broken", "despair",
        "desolate", "empty", "numb", "introspective", "emo", "blue", "downcast",
        "sullen", "gloomy", "mournful", "plaintive", "lament", "elegy", "weeping",
        "solemn", "heavy", "raw", "confessional", "wounded", "aching", "hopeless",
        "isolation", "alienation", "existential", "depression", "crying", "tears",
        "devastated", "shattered", "dejected", "heartache", "misery", "woeful",
        "despondent", "inconsolable", "bereft", "morose", "depress",
    ],
    "Angry": [
        "angry", "aggressive", "fierce", "rage", "brutal", "confrontational",
        "rebellious", "thrash", "hardcore", "anxiety", "paranoia", "intense",
        "hostile", "furious", "violent", "screamo", "noise rock", "noise",
        "abrasive", "dissonant", "crushing", "relentless", "savage", "ferocious",
        "vicious", "menacing", "sinister", "threatening", "ominous", "combative",
        "riot", "tension", "frustration", "agitated", "bitter", "resentment",
        "volatile", "seething", "anger", "hate", "hatred", "grudge", "revenge",
        "chaotic", "mayhem", "unhinged", "feral", "visceral", "primal",
        "death metal", "grindcore", "black metal", "power violence", "metalcore",
        "deathcore", "crust punk", "anarcho", "sludge", "doom",
    ],
    "Chill": [
        "chill", "chillout", "chill out", "mellow", "laid back", "laid-back",
        "relax", "relaxing", "soothing", "calm", "calming", "peaceful", "lo-fi",
        "lofi", "lounge", "bossa nova", "acoustic", "coffee shop", "easy listening",
        "slow", "quiet", "gentle", "soft", "smooth", "serene", "tranquil",
        "meditative", "meditation", "sleepy", "bedtime", "late night", "wind down",
        "drifting", "floaty", "foggy", "hazy", "muted", "hushed", "intimate",
        "cozy", "warm", "comfortable", "chillhop", "downtempo", "soft rock",
        "tea", "rainy day", "sunday morning", "lazy", "unwind", "de-stress",
        "breathe", "stillness", "easy", "gentle indie", "acoustic pop",
        "background music", "study", "focus", "concentration", "ambient pop",
        "trip hop", "cafe", "sunday", "weekend", "afternoon",
    ],
    "Energetic": [
        "energetic", "high energy", "party", "dance", "driving", "pumping",
        "rave", "club", "edm", "house music", "techno", "workout", "motivation",
        "running", "exercise", "sports", "adrenaline", "powerful", "fast", "upbeat",
        "propulsive", "relentless", "stadium", "festival", "euphoric club", "fire",
        "hype", "banger", "exhilarating", "fast-paced", "active", "loud", "explosive",
        "wild", "frenetic", "trance", "dubstep", "drum and bass", "dnb",
        "high bpm", "sprint", "beast mode", "electrifying", "charged", "frantic",
        "rush", "frenzy", "bang", "dancefloor", "dance music", "club music",
        "pump up", "get hyped", "turnt", "fired up", "energy drink", "boost",
        "power", "drive", "accelerate", "full throttle", "hardcore dance",
        "hard dance", "breakbeat", "speed", "maximum", "intensity",
    ],
    "Romantic": [
        "romantic", "love song", "love songs", "ballad", "sensual", "passionate",
        "slow jam", "tender", "devotional", "candlelight", "date night", "bedroom",
        "seduction", "flirtatious", "couple", "wedding", "anniversary", "first dance",
        "slow dance", "intimate", "lush", "heartfelt", "affectionate", "loving",
        "caring", "devoted", "forever", "together", "connection", "desire", "adore",
        "cherish", "soulmates", "amour", "love letter", "valentine", "crush",
        "infatuation", "sweetheart", "darling", "honey", "swoon",
        "romance", "lover", "beloved", "tenderness", "longing for love",
        "making love", "intimate evening", "date", "dinner music", "love",
        "soft kisses", "moonlit", "affection", "devotion", "serenade",
    ],
    "Epic": [
        "epic", "anthemic", "dramatic", "orchestral", "cinematic", "majestic",
        "triumphant", "heroic", "symphonic", "film score", "soundtrack", "video game",
        "fantasy", "adventure", "quest", "journey", "grandiose", "sweeping",
        "bombastic", "larger than life", "soaring", "towering", "legendary", "mythic",
        "transcendent", "glorious", "victory", "climactic", "battle", "war cry",
        "power ballad", "progressive rock", "prog rock", "complex", "layered",
        "post-rock", "instrumental rock", "epic metal", "overture", "opus", "finale",
        "zenith", "pinnacle", "grand", "magnificent", "awe-inspiring", "immense",
        "colossal", "vast", "movie music", "television score", "game music",
        "boss battle", "final boss", "big chorus", "stadium rock", "arena rock",
    ],
    "Dreamy": [
        "dreamy", "ethereal", "ambient", "spacey", "cosmic", "psychedelic",
        "shoegaze", "dream pop", "vaporwave", "reverb", "hypnagogic", "liminal",
        "surreal", "otherworldly", "celestial", "stars", "night sky", "clouds",
        "floating", "weightless", "hazy", "blurry", "distant", "fuzzy", "languid",
        "narcotic", "spaced out", "zoned out", "new age", "spiritual", "mystical",
        "impressionistic", "gauzy", "drifting", "trance-like", "mind-expanding",
        "lysergic", "kaleidoscopic", "aqueous", "gossamer", "misty", "wispy",
        "bedroom pop", "indietronica", "synth pop", "chillwave", "hypnotic",
        "somnambulant", "reverie", "daydream", "slumber", "half-asleep",
        "twilight", "dusk", "moonlight", "stargazing", "space", "cosmos",
        "noise pop", "wall of sound", "reverb-soaked", "tape hiss",
    ],
    "Groovy": [
        "groovy", "funky", "funk", "swing", "jazzy", "soul", "rhythm", "hip hop",
        "trap beat", "afrobeat", "salsa", "dancehall", "infectious", "bounce",
        "groove", "smooth jazz", "acid jazz", "neo soul", "motown", "detroit soul",
        "disco", "funk rock", "slap bass", "pocket", "club jazz", "boogie",
        "strut", "sway", "latin jazz", "boogaloo", "electric piano", "clavinet",
        "wah wah", "tight", "locked in", "shuffle", "syncopated", "polyrhythmic",
        "rhythmic", "bass-heavy", "swagger", "bass line", "drum groove",
        "head nod", "toe-tapping", "body roll", "hip", "suave",
        "cool", "laid-back groove", "funk soul", "r&b groove", "soulful",
        "afropop", "tropical", "cumbia", "merengue",
    ],
    "Nostalgic": [
        "nostalgia", "nostalgic", "retro", "oldies", "vintage", "throwback",
        "80s", "1980s", "90s", "1990s", "70s", "1970s", "60s", "1960s",
        "memories", "americana", "classic rock", "simpler times", "reminisce",
        "golden era", "classic", "timeless", "old school", "childhood", "youth",
        "sunset", "faded", "worn", "antique", "sepia", "analog", "cassette",
        "vinyl", "radio", "decade", "archive", "historical", "golden age",
        "bygone", "good old days", "back then", "past", "era", "revivalist",
        "pastiche", "homage", "tribute", "classic hits", "greatest hits",
        "back to basics", "reminiscing", "long ago", "remember", "memory",
        "golden age of rock", "new wave", "post punk", "hair metal",
    ],
}

# Priority tie-breaking order (first matching mood in this list wins)
PRIORITY: List[str] = [
    "Happy", "Energetic", "Groovy", "Romantic", "Chill",
    "Dreamy", "Sad", "Nostalgic", "Epic", "Angry",
]

# ---------------------------------------------------------------------------
# Genre → Mood fallback (used when no tags match any mood keyword)
# ---------------------------------------------------------------------------

GENRE_MOOD_FALLBACK: Dict[str, str] = {
    # Rock family
    "rock": "Energetic",
    "hard rock": "Energetic",
    "classic rock": "Nostalgic",
    "alternative rock": "Chill",
    "alternative": "Chill",
    "grunge": "Angry",
    "punk": "Angry",
    "punk rock": "Angry",
    "pop-punk": "Energetic",
    "pop punk": "Energetic",
    "post-punk": "Chill",
    "emo": "Sad",
    "screamo": "Angry",
    "thrash metal": "Angry",
    "death metal": "Angry",
    "heavy metal": "Angry",
    "metal": "Angry",
    "metalcore": "Angry",
    "progressive rock": "Epic",
    "prog rock": "Epic",
    "psychedelic rock": "Dreamy",
    "indie rock": "Chill",
    "indie": "Chill",
    "garage rock": "Energetic",
    "post-rock": "Epic",
    "shoegaze": "Dreamy",
    "dream pop": "Dreamy",
    # Pop family
    "pop": "Happy",
    "indie pop": "Happy",
    "synth-pop": "Dreamy",
    "bubblegum pop": "Happy",
    "k-pop": "Happy",
    "j-pop": "Happy",
    "electropop": "Energetic",
    "power pop": "Happy",
    # Electronic family
    "electronic": "Energetic",
    "electronica": "Dreamy",
    "edm": "Energetic",
    "house": "Energetic",
    "techno": "Energetic",
    "trance": "Energetic",
    "dubstep": "Energetic",
    "drum and bass": "Energetic",
    "ambient": "Dreamy",
    "trip hop": "Chill",
    "downtempo": "Chill",
    "chillwave": "Dreamy",
    "vaporwave": "Dreamy",
    "lo-fi": "Chill",
    "lofi": "Chill",
    # Hip-Hop / Urban
    "hip-hop": "Groovy",
    "hip hop": "Groovy",
    "rap": "Groovy",
    "trap": "Groovy",
    "r&b": "Romantic",
    "rnb": "Romantic",
    "soul": "Groovy",
    "neo soul": "Groovy",
    "funk": "Groovy",
    "disco": "Groovy",
    "motown": "Groovy",
    # Jazz / Blues
    "jazz": "Chill",
    "smooth jazz": "Chill",
    "bebop": "Groovy",
    "blues": "Sad",
    "blues rock": "Sad",
    # Classical / Orchestral
    "classical": "Epic",
    "orchestral": "Epic",
    "opera": "Epic",
    "chamber music": "Chill",
    "film score": "Epic",
    "soundtrack": "Epic",
    # Folk / Country / Americana
    "folk": "Sad",
    "indie folk": "Sad",
    "singer-songwriter": "Sad",
    "country": "Nostalgic",
    "americana": "Nostalgic",
    "bluegrass": "Nostalgic",
    # World
    "latin": "Groovy",
    "salsa": "Groovy",
    "bossa nova": "Chill",
    "afrobeat": "Groovy",
    "reggae": "Chill",
    "dancehall": "Groovy",
}

# ---------------------------------------------------------------------------
# Last.fm tag → canonical iTunes-style genre name
# ---------------------------------------------------------------------------

GENRE_NORMALIZE: Dict[str, str] = {
    # Rock
    "rock": "Rock",
    "hard rock": "Rock",
    "classic rock": "Rock",
    "alternative rock": "Alternative",
    "alt rock": "Alternative",
    "alternative": "Alternative",
    "alt-rock": "Alternative",
    "indie rock": "Indie",
    "indie pop": "Indie",
    "indie": "Indie",
    "grunge": "Alternative",
    "punk rock": "Punk",
    "punk": "Punk",
    "pop punk": "Punk",
    "pop-punk": "Punk",
    "post-punk": "Alternative",
    "emo": "Emo",
    "screamo": "Emo",
    "progressive rock": "Progressive Rock",
    "prog rock": "Progressive Rock",
    "prog": "Progressive Rock",
    "psychedelic rock": "Psychedelic",
    "psych rock": "Psychedelic",
    "shoegaze": "Indie",
    "dream pop": "Indie",
    "post-rock": "Post-Rock",
    "garage rock": "Rock",
    "stoner rock": "Rock",
    "desert rock": "Rock",
    "new wave": "New Wave",
    "krautrock": "Experimental",
    # Metal
    "metal": "Metal",
    "heavy metal": "Metal",
    "thrash metal": "Metal",
    "death metal": "Metal",
    "black metal": "Metal",
    "metalcore": "Metal",
    "deathcore": "Metal",
    "doom metal": "Metal",
    "power metal": "Metal",
    "symphonic metal": "Metal",
    "hair metal": "Metal",
    # Pop
    "pop": "Pop",
    "synth-pop": "Synth-Pop",
    "bubblegum": "Pop",
    "k-pop": "K-Pop",
    "j-pop": "J-Pop",
    "electropop": "Synth-Pop",
    "power pop": "Pop",
    "chamber pop": "Indie",
    "baroque pop": "Indie",
    # Electronic
    "electronic": "Electronic",
    "electronica": "Electronic",
    "edm": "Electronic",
    "house": "Electronic",
    "house music": "Electronic",
    "techno": "Electronic",
    "trance": "Electronic",
    "dubstep": "Electronic",
    "drum and bass": "Electronic",
    "dnb": "Electronic",
    "ambient": "Ambient",
    "trip hop": "Trip Hop",
    "downtempo": "Electronic",
    "chillwave": "Electronic",
    "vaporwave": "Electronic",
    "lo-fi": "Lo-Fi",
    "lofi": "Lo-Fi",
    "glitch": "Electronic",
    "idm": "Electronic",
    "breakbeat": "Electronic",
    "synthwave": "Electronic",
    "retrowave": "Electronic",
    # Hip-Hop / Urban
    "hip-hop": "Hip-Hop",
    "hip hop": "Hip-Hop",
    "rap": "Hip-Hop",
    "trap": "Hip-Hop",
    "grime": "Hip-Hop",
    "r&b": "R&B",
    "rnb": "R&B",
    "soul": "Soul",
    "neo soul": "Soul",
    "funk": "Funk",
    "disco": "Disco",
    "motown": "Soul",
    # Jazz / Blues
    "jazz": "Jazz",
    "smooth jazz": "Jazz",
    "bebop": "Jazz",
    "acid jazz": "Jazz",
    "fusion": "Jazz",
    "blues": "Blues",
    "blues rock": "Blues",
    "chicago blues": "Blues",
    "delta blues": "Blues",
    # Classical
    "classical": "Classical",
    "orchestral": "Classical",
    "opera": "Classical",
    "chamber music": "Classical",
    "film score": "Soundtrack",
    "soundtrack": "Soundtrack",
    "score": "Soundtrack",
    "video game music": "Soundtrack",
    # Folk / Country / Americana
    "folk": "Folk",
    "indie folk": "Folk",
    "singer-songwriter": "Singer-Songwriter",
    "singer songwriter": "Singer-Songwriter",
    "country": "Country",
    "americana": "Country",
    "bluegrass": "Country",
    "alt-country": "Country",
    "outlaw country": "Country",
    # World
    "latin": "Latin",
    "salsa": "Latin",
    "reggaeton": "Latin",
    "bossa nova": "Latin",
    "afrobeat": "World",
    "afropop": "World",
    "reggae": "Reggae",
    "dancehall": "Reggae",
    "ska": "Reggae",
    "world music": "World",
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_tag_counts(tag_db: dict) -> Dict[str, int]:
    """
    Count global tag frequency across all tracks in tag_db.
    Used for IDF (inverse document frequency) weighting in canonical_mood().

    tag_db format: {"artist - track" -> List[str]} or legacy {"artist - track" -> {"tags": [...]}}
    """
    counts: Dict[str, int] = collections.Counter()
    for val in tag_db.values():
        if isinstance(val, list):
            tag_list = val
        elif isinstance(val, dict):
            tag_list = val.get("tags", [])
        else:
            continue
        for t in tag_list:
            counts[t.lower()] += 1
    return dict(counts)


def canonical_mood(
    tags,
    genre: Optional[str] = None,
    tag_counts: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """
    Map a list of Last.fm tags (and optional genre) to one of the 10 canonical moods.

    Algorithm:
    1. For each tag, check all mood keyword lists using substring matching.
       Score each mood using IDF weighting (rarer tags carry more signal).
    2. If no tag matches, fall back to GENRE_MOOD_FALLBACK on the genre string.
    3. Return None only if both steps fail.

    Args:
        tags:        List of Last.fm tag strings (may be empty or None).
        genre:       iTunes-style genre string for fallback (may be None).
        tag_counts:  Global tag frequency dict from build_tag_counts().
    """
    scores: Dict[str, float] = {m: 0.0 for m in MOODS}
    total = max(sum(tag_counts.values()) if tag_counts else 0, 1)

    for raw_tag in (tags or []):
        cleaned = re.sub(r"[^\w\s-]", " ", raw_tag.lower()).strip()
        # IDF: common tags (like "seen live") carry almost no signal
        freq = (tag_counts or {}).get(cleaned, 0)
        idf = 1.0 / math.log1p(freq / total * 100 + 1)

        for mood, keywords in MOODS.items():
            for kw in keywords:
                if kw in cleaned:
                    scores[mood] += idf
                    break  # only count once per (tag, mood) pair

    best_score = max(scores.values())
    if best_score > 0:
        best_moods = [m for m, s in scores.items() if s == best_score]
        for m in PRIORITY:
            if m in best_moods:
                return m
        return best_moods[0]

    # Fallback: derive mood from genre string
    if genre:
        genre_key = genre.lower().strip()
        if genre_key in GENRE_MOOD_FALLBACK:
            return GENRE_MOOD_FALLBACK[genre_key]
        # Partial match (e.g. "Indie Rock" → "indie rock")
        for gk, mood in GENRE_MOOD_FALLBACK.items():
            if gk in genre_key or genre_key in gk:
                return mood

    return None


def canonical_genre(tag: str) -> Optional[str]:
    """
    Map a Last.fm tag string to a normalized iTunes-style genre name.
    Returns None if the tag has no known genre mapping.
    """
    if not tag:
        return None
    return GENRE_NORMALIZE.get(tag.lower().strip())
