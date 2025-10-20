from src.transcript.get_vid_id import get_video_id
from src.transcript.get_youtube_transcript import get_transcript

def mains(url):
    video_id = get_video_id(url)
    transcript = get_transcript(url)
    return transcript , video_id


if __name__ == "__main__":    
    url = input("Enter the URL of the YouTube video: ")
    transcript , video_id = mains(url)
    print(transcript)
    print(video_id)
    