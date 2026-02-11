import json
from cerebras.cloud.sdk import Cerebras
from config import CEREBRAS_API_KEY

if not CEREBRAS_API_KEY:
    raise ValueError("Cerebras API Key missing! Check config.py/env vars.")

client = Cerebras(api_key=CEREBRAS_API_KEY)

# ============================================================
# LLM PROMPTS
# ============================================================
SYSTEM_PROMPT = """
You are a Clickbait Detection Expert. Your goal is to identify deceptive patterns.

You may be provided with a VIDEO TRANSCRIPT. 
IF TRANSCRIPT IS PROVIDED:
1. **Verify Promise**: Does the video actually contain the content promised in the Title?
2. **Find Timestamp**: If found, extract the timestamp (e.g. "04:20") where the promise is fulfilled.
3. **Flag "Missing Content"**: If the entire transcript fails to mention or show the promised topic, flag it.

Analyze for these flags:
1. **Piracy/Hack**: Downloads, cracks, mods, free money, hacks.
2. **Fake News/Scam**: Factually false, known hoaxes, "Full Movie" fakes.
3. **Duration Scam**: "Full Movie" claim but short duration.
4. **False Official**: "Official Trailer", "Full Movie", "Full Episode", or "Full Match" from random/personal channel. ONLY official studios/broadcasters (Netflix, Hotstar, T-Series, Sports channels) upload full copyrighted content.
   - **EXCEPTION**: "Full Tutorial", "Full Course", "Full Gameplay" are OK from personal channels.
   - **RULE**: If "Full [Movie/Episode/Match]" AND Channel is Unknown -> FLAG IT.
5. **Missing Promise** (ONLY if transcript provided): The video does not contain the promised content.

Return JSON only:
{
    "is_clickbait": boolean,
    "confidence": 0.0 to 1.0,
    "reason": "Detailed explanation.",
    "flags": ["list", "of", "flags"],
    "verification_timestamp": "MM:SS" (e.g., "02:15". ONLY provide if found in the TRANSCRIPT SNIPPET. If unsure or not found, return null.)
}
"""

def analyze_semantic_flags(video_data, transcript=None):
    """
    Use Llama 3 to detect flags. If transcript is present, verify title promises.
    """
    transcript_snippet = f"Transcript Snippet: {str(transcript)[:1500]}..." if transcript else "Transcript: Not Available"
    
    user_content = f"""
    Title: {video_data['title']}
    Channel: {video_data['channel']}
    Duration: {video_data['duration_min']:.1f} minutes
    Stats: {video_data['views']} views, {video_data['likes']} likes
    Description: {video_data['description'][:500]}
    
    {transcript_snippet}
    
    Check for flags.
    """
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            model="llama3.1-8b",
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"is_clickbait": False, "confidence": 0, "flags": [], "verification_timestamp": None}
