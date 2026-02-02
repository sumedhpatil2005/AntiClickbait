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

def clean_timestamp(ts):
    """Converts raw seconds/strings (e.g., '29.5s', '120', '31:679') to MM:SS format."""
    if not ts: return None
    try:
        # Normalize input
        ts_str = str(ts).lower().replace('s', '').strip()
        
        # Filter out "not applicable", "none", "n/a", "not found"
        if ts_str in ["not applicable", "none", "n/a", "null", "unknown", "not found"]:
            return None

        # Handle "MM:SS" or "MM:SS:ms" or typos like "31:679"
        if ":" in ts_str:
            parts = ts_str.split(":")
            # Standard MM:SS
            if len(parts) == 2:
                m = float(parts[0])
                s_part = float(parts[1])
                
                # If seconds > 59, it likely matches a typo like "31:679" which meant 31.679s but got colon-ed
                if s_part >= 60: 
                    # Treat colon as decimal
                    total_seconds = float(ts_str.replace(":", "."))
                    m = int(total_seconds // 60)
                    s = int(total_seconds % 60)
                    return f"{m:02}:{s:02}"
                
                # Valid MM:SS
                return f"{int(m):02}:{int(s_part):02}"
                
            return ts_str # Return HH:MM:SS as is
        
        # Handle pure numbers (90 -> 01:30)
        seconds = float(ts_str)
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02}:{s:02}"
        
    except:
        return ts # Return original if parsing fails

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
