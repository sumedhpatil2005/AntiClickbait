def ensemble_predict(gbm_prob, llm_result):
    """
    Combine LightGBM probability with LLM Semantic Flags.
    """
    final_prob = gbm_prob
    reason = "Statistical Analysis"
    
    # Normalize flags to ensure matching works regardless of LLM output format
    raw_flags = llm_result.get("flags", [])
    # Convert all to lowercase string for easy searching
    flags_str = " ".join([str(f).lower() for f in raw_flags])
    
    # Check for specific keywords in the flags
    if "piracy" in flags_str or "crack" in flags_str:
        final_prob = max(final_prob, 0.95)
        reason = "Detected Piracy/Mods/Hacks"
    elif "fake_news" in flags_str or "scam" in flags_str or "fake news" in flags_str:
        final_prob = max(final_prob, 0.95)
        reason = "Detected Fake News/Scam"
    elif "duration" in flags_str: # duration_mismatch or Duration Scam
        final_prob = max(final_prob, 0.90)
        reason = "Duration does not match content claims"
    elif "false_official" in flags_str or "official" in flags_str:
        final_prob = max(final_prob, 0.85)
        reason = "Unofficial channel claiming official content"
    elif "promise" in flags_str: # missing_promise
        final_prob = max(final_prob, 0.85)
        reason = "Video does not deliver on title promise"
    
    # Soft boost from LLM confidence if generic clickbait
    if llm_result.get("is_clickbait") and not raw_flags:
        llm_conf = llm_result.get("confidence", 0.5)
        # Average the probabilities if LLM is confident
        if llm_conf > 0.7:
             final_prob = (gbm_prob + llm_conf) / 2
             reason = "Combined AI consensus"

    # SPECIAL CASE: LLM Veto (Fix for "Full Movie" False Positives)
    # If LLM says "Not Clickbait" and is confident, we trust it over the statistical model
    elif not llm_result.get("is_clickbait") and not raw_flags:
        llm_conf = llm_result.get("confidence", 0.5)
        if llm_conf > 0.8:
            # If LLM is very sure it's safe (e.g. "Shemaroo channel"), force low probability
            final_prob = 0.1 
            reason = "LLM Overrode Statistical Model (Trusted Source)"
        # REMOVED THE AVERAGING LOGIC - let the transcript verification handle this
            
    return final_prob, reason, raw_flags
