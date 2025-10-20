from src.transcript.get_vid_id import get_video_id
from youtube_transcript_api import YouTubeTranscriptApi
import sys
import os

def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)

    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()

    # Fetch the list of available transcripts for the given YouTube video
    youtube_transcript = ytt_api.fetch(video_id, languages=['en'])
    youtube_transcript_in_raw = youtube_transcript.to_raw_data()
    
    transcript = " ".join(chunk["text"] for chunk in youtube_transcript_in_raw)

    return transcript



