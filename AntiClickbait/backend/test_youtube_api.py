"""Test YouTube API v3"""
import requests

API_KEY = 'AIzaSyDwDvb0qVWCPPTv8sRVOojA4FvOA9-6Zfg'
video_id = 'UF8uR6Z6KLc'  # Steve Jobs Stanford speech

url = 'https://www.googleapis.com/youtube/v3/videos'
params = {
    'part': 'snippet,statistics,contentDetails',
    'id': video_id,
    'key': API_KEY
}

response = requests.get(url, params=params, timeout=10)
data = response.json()

if 'items' in data and len(data['items']) > 0:
    item = data['items'][0]
    snippet = item['snippet']
    stats = item['statistics']
    content = item['contentDetails']
    
    print('SUCCESS! YouTube API working')
    print(f"Title: {snippet['title']}")
    print(f"Views: {stats['viewCount']}")
    print(f"Likes: {stats.get('likeCount', 'N/A')}")
    print(f"Duration: {content['duration']}")
    print(f"Category ID: {snippet['categoryId']}")
else:
    print('ERROR:', data)
