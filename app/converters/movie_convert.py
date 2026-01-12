import os
import argparse
from moviepy.editor import VideoFileClip

def convert_videos_to_mp3(input_folder: str, output_folder: str):
    """
    Convert all video files in input_folder to mp3 and save them to output_folder.

    Args:
        input_folder (str): Path containing video files.
        output_folder (str): Path where MP3 files will be saved.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported video file extensions
    video_extensions = (".mp4", ".mkv", ".avi", ".mov", ".flv")

    # Iterate over files
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(video_extensions):
            input_path = os.path.join(input_folder, filename)
            # Generate output path: replace extension with .mp3
            output_filename = os.path.splitext(filename)[0] + ".mp3"
            output_path = os.path.join(output_folder, output_filename)

            try:
                # Load video and extract audio
                video_clip = VideoFileClip(input_path)
                audio_clip = video_clip.audio
                if audio_clip is None:
                    print(f"No audio in {filename}, skipping.")
                    continue
                # Write audio to mp3
                audio_clip.write_audiofile(output_path, logger=None)
                audio_clip.close()
                video_clip.close()
                print(f"Converted {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert all videos in a folder to MP3.")
    parser.add_argument("-i", "--input", required=True, help="Path to input folder containing videos")
    parser.add_argument("-o", "--output", required=True, help="Path to output folder for MP3s")

    args = parser.parse_args()
    convert_videos_to_mp3(args.input, args.output)

if __name__ == "__main__":
    main()

