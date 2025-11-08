import cv2
import numpy as np
import moviepy.editor as mpy
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import time
import psutil

def unique_filename(base="test", ext="mp4"):
    i = 1
    while os.path.exists(f"{base}{i}.{ext}"):
        i += 1
    return f"{base}{i}.{ext}"

def record_webcam_video(out_vid, duration=30, fps=60, width=640, height=480):
    cap = cv2.VideoCapture(0)
    # Set webcam resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_vid, fourcc, fps, (width, height))
    print(f"Recording webcam video for {duration}s at {fps} FPS and {width}x{height} resolution.")
    start = time.time()
    frame_count = 0
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        cv2.imshow("Webcam (press Q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Webcam video saved as {out_vid}, captured {frame_count} frames.")
    return out_vid, frame_count

def extract_frames(vid_path, target_fps=60, width=640, height=480):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(1, int(round(actual_fps / target_fps)))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            resized_frame = cv2.resize(frame, (width, height))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        count += 1
    cap.release()
    return frames, target_fps

def record_audio(duration=30, fs=44100, filename="audio.wav"):
    print(f"Recording audio for {duration}s at {fs} Hz sample rate...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"Audio recorded and saved as {filename}")
    return filename, fs

def combine_audio_video(frames, audio_path, fps, out_path="result.mp4", width=640, height=480):
    print("Encoding video and combining with audio...")
    start_time = time.time()
    temp_vid = "temp_video.mp4"
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    # Set bitrate for good quality and 480p resolution
    clip.write_videofile(temp_vid, codec="libx264", audio=False, bitrate="1500k", logger=None)
    video_clip = mpy.VideoFileClip(temp_vid)
    audio_clip = mpy.AudioFileClip(audio_path)
    duration = min(video_clip.duration, audio_clip.duration)
    final_clip = video_clip.set_audio(audio_clip.subclip(0, duration)).subclip(0, duration)
    final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", bitrate="1500k", logger=None)
    os.remove(temp_vid)
    end_time = time.time()
    print(f"Output saved as {out_path} in {end_time - start_time:.2f} seconds.")
    return end_time - start_time

def print_stats(source_files, output_file, encode_time, audio_sample_rate, fps, resolution):
    print("\n--- Compression and Encoding Statistics ---")
    for f in source_files:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"{os.path.basename(f)} size: {size_mb:.2f} MB")
    if os.path.exists(output_file):
        out_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Output file size: {out_size:.2f} MB")
    print(f"Encoding time: {encode_time:.2f} seconds")
    mem = psutil.virtual_memory()
    print(f"System Memory usage: Used {mem.used / (1024**3):.2f} GB / Total {mem.total / (1024**3):.2f} GB")
    print(f"Audio sample rate: {audio_sample_rate} Hz")
    print(f"Video FPS: {fps}")
    print(f"Video resolution: {resolution[0]}x{resolution[1]}")

def main():
    print("Select input method:")
    print("1: Import video file")
    print("2: Record from webcam")
    choice = input("Enter 1 or 2: ").strip()

    duration = 30
    fps = 60
    width, height = 640, 480

    start_time = time.time()

    if choice == "1":
        video_path = input("Enter path to video file: ").strip()
        frames, actual_fps = extract_frames(video_path, target_fps=fps, width=width, height=height)
        clip = mpy.VideoFileClip(video_path)
        if clip.audio:
            audio_path = "extracted_audio.wav"
            clip.audio.write_audiofile(audio_path, logger=None)
            audio_sample_rate = clip.audio.fps
        else:
            audio_path, audio_sample_rate = record_audio(duration=duration)
    elif choice == "2":
        video_path = unique_filename("test", "mp4")
        video_path, recorded_frames = record_webcam_video(video_path, duration=duration, fps=fps, width=width, height=height)
        frames, actual_fps = extract_frames(video_path, target_fps=fps, width=width, height=height)
        audio_path, audio_sample_rate = record_audio(duration=duration)
    else:
        print("Invalid choice, exiting.")
        return

    encode_time = combine_audio_video(frames, audio_path, fps=actual_fps, out_path="result.mp4", width=width, height=height)

    end_time = time.time()

    print_stats([video_path, audio_path], "result.mp4", encode_time, audio_sample_rate, fps, (width, height))

    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
