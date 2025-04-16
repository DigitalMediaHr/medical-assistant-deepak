from youtube_transcript_api import YouTubeTranscriptApi
def get_youtube_transcript(url):
    video_id = url.split("v=")[-1]
    print(f"Video ID: {video_id}")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = ""
    for entry in transcript:
        text += entry['text']
    return text