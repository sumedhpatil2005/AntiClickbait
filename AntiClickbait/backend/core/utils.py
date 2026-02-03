import re

# ============================================================
# CONSTANTS
# ============================================================
EMOTIONAL_EMOJIS = ['😱', '🔥', '☠️', '💥', '🤯', '😶', '😭', '😡', '💀', '⚠️']

# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_duration(duration_str):
    if not duration_str: return 0.0
    duration_str = duration_str.replace('PT', '')
    hours = 0; minutes = 0; seconds = 0
    if 'H' in duration_str: hours, duration_str = duration_str.split('H'); hours = int(hours)
    if 'M' in duration_str: minutes, duration_str = duration_str.split('M'); minutes = int(minutes)
    if 'S' in duration_str: seconds = int(duration_str.replace('S', ''))
    return hours * 60 + minutes + seconds / 60

def clean_timestamp(ts, duration_min=None):
    """Converts raw seconds/strings to MM:SS format and validates against video duration."""
    if not ts: return None
    try:
        # Normalize input
        ts_str = str(ts).lower().replace('s', '').strip()
        
        # Filter out junk
        if ts_str in ["not applicable", "none", "n/a", "null", "unknown", "not found"]:
            return None

        total_seconds = 0
        
        # Case 1: HH:MM:SS or MM:SS
        if ":" in ts_str:
            parts = ts_str.split(":")
            if len(parts) == 2:
                # MM:SS
                m, s = float(parts[0]), float(parts[1])
                # Fix typos like 31:679 -> 31.679
                if s >= 60:
                    total_seconds = float(ts_str.replace(":", "."))
                else:
                    total_seconds = m * 60 + s
            elif len(parts) == 3:
                # HH:MM:SS
                h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                total_seconds = h * 3600 + m * 60 + s
            else:
                return ts_str # Unknown format
        else:
            # Case 2: Pure numbers (seconds)
            total_seconds = float(ts_str)

        # Validation against duration
        if duration_min:
            max_seconds = duration_min * 60
            # If timestamp is impossible, ignore it
            if total_seconds > max_seconds:
                return None
            # If timestamp is near zero but not quite, it might be a hallucination (e.g. 1s)
            # but usually we trust short ones.
        
        # Format back to MM:SS or HH:MM:SS
        m_total = int(total_seconds // 60)
        final_s = int(total_seconds % 60)
        if m_total >= 60:
            h = m_total // 60
            m = m_total % 60
            return f"{h:02}:{m:02}:{final_s:02}"
        return f"{m_total:02}:{final_s:02}"
        
    except:
        return ts

# ============================================================
# SENTIMENT HELPERS
# ============================================================
def simple_polarity(text):
    text = str(text).lower()
    positive = ['good', 'great', 'best', 'amazing', 'awesome', 'excellent', 'love', 'happy', 'perfect']
    negative = ['bad', 'worst', 'terrible', 'horrible', 'hate', 'ugly', 'awful', 'angry', 'scam', 'fake']
    pos = sum(1 for w in positive if w in text)
    neg = sum(1 for w in negative if w in text)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0

def simple_subjectivity(text):
    text = str(text).lower()
    sub = ['i think', 'believe', 'opinion', 'probably', 'maybe', 'seems', 'best', 'worst', 'amazing', 'terrible']
    count = sum(1 for w in sub if w in text)
    wc = len(text.split())
    return min(count / (wc / 10 + 1), 1.0) if wc > 0 else 0.0

def word_overlap(text1, text2):
    w1 = set(str(text1).lower().split())
    w2 = set(str(text2).lower().split())
    if not w1 or not w2: return 0.0
    return len(w1.intersection(w2)) / min(len(w1), len(w2))

def tfidf_similarity(text1, text2):
    w1 = set(str(text1).lower().split())
    w2 = set(str(text2).lower().split())
    if not w1 or not w2: return 0.0
    u = w1.union(w2)
    return len(w1.intersection(w2)) / len(u) if u else 0.0

def count_keywords(text, keywords):
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw in text_lower)
