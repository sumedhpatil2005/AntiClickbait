import pandas as pd

# Load the relabeled dataset
df = pd.read_csv('MASTER_DATASET_RELABELED.csv')

# Find videos marked as misleading
misleading = df[df['clickbait_relabeled'] == 1]

print("=" * 80)
print(" VIDEOS MARKED AS MISLEADING (Clickbait)")
print("=" * 80)
print(f"\nTotal misleading videos: {len(misleading)}\n")

for idx, row in misleading.iterrows():
    line_num = idx + 2  # +2 because pandas is 0-indexed and CSV has header
    print(f"📍 Line Number: {line_num}")
    print(f"🆔 Video ID: {row['video_id']}")
    print(f"📺 Title: {row['title']}")
    print(f"📝 Transcript Length: {int(row['transcript_length'])} characters")
    print(f"⚠️  Why Misleading: ", end="")
    
    title_lower = str(row['title']).lower()
    
    # Check why it was marked misleading
    if 'full movie' in title_lower or 'पूरी फिल्म' in title_lower:
        print("Claims 'Full Movie' but transcript too short")
    elif row['transcript_length'] < 100:
        print("Very short transcript + clickbait keywords")
    else:
        print("Clickbait keywords + short content")
    
    print("-" * 80)

print("\n✅ Complete list generated!")
