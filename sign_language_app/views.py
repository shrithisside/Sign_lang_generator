import os
import re
import json
import wave
import logging
import numpy as np
from PIL import Image
# Ensure backward compatibility for MoviePy with Pillow 10+
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

import django
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
import vosk

# Ensure NLTK punkt tokenizer is downloaded
from nltk import download
try:
    download('punkt', quiet=True)
except Exception:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)

# Dynamic Paths
MODEL_PATH = os.getenv('SMD_MODEL_PATH', os.path.join(settings.BASE_DIR, 'models', 'smodel.keras'))
VOSK_MODEL_PATH = os.getenv('VOSK_MODEL_PATH', os.path.join(settings.BASE_DIR, 'models', 'vosk-model-small-en-us-0.15'))
ASL_DATASET_PATH = os.path.join(settings.BASE_DIR, 'archive', 'asl_dataset')

# Tokenizer setup
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts('abcdefghijklmnopqrstuvwxyz0123456789')
MAX_LABEL_LENGTH = 5

_model = None

def get_sign_model():
    """Lazily loads the trained sign language model."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                logging.info(f"Loading sign language model from: {MODEL_PATH}")
                _model = tf.keras.models.load_model(MODEL_PATH)
            except Exception as e:
                logging.warning(f"Could not load model at {MODEL_PATH}: {e}")
                return None
        else:
            logging.warning(f"Model file not found at {MODEL_PATH}.")
            return None
    return _model

def get_real_sign_image(char):
    """Fetches high-resolution real ASL hand gesture image from dataset."""
    c = char.lower()
    letter_folder = os.path.join(ASL_DATASET_PATH, c)
    if os.path.isdir(letter_folder):
        files = sorted([f for f in os.listdir(letter_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if files:
            img_path = os.path.join(letter_folder, files[0])
            try:
                return Image.open(img_path).convert('RGB')
            except Exception as e:
                logging.error(f"Error loading real sign image: {e}")
    return None

def generate_image_from_letter(model, letter, max_label_length=MAX_LABEL_LENGTH):
    """
    Returns a PIL Image of the sign language gesture for the given letter.
    Uses real ASL hand signs if available, with model prediction fallback.
    """
    real_img = get_real_sign_image(letter)
    if real_img is not None:
        return real_img

    if model is not None:
        seq = tokenizer.texts_to_sequences([letter.lower()])
        padded = pad_sequences(seq, maxlen=max_label_length)
        pred = model.predict(padded, verbose=0)
        arr = np.clip(pred.squeeze(), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    # Fallback blank white canvas with letter drawn
    fallback = Image.new('RGB', (128, 128), (255, 255, 255))
    return fallback

def combine_images_into_word(images, word_spacing=5):
    """Stitches letter images horizontally into a single word banner."""
    if not images:
        return Image.new('RGB', (100, 100), (255, 255, 255))
    
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + word_spacing * (len(images) - 1)
    max_height = max(heights)

    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, (max_height - img.height) // 2))
        x_offset += img.width + word_spacing

    return combined

def ensure_dirs():
    """Ensures media directories exist."""
    letters_dir = os.path.join(settings.MEDIA_ROOT, 'letters')
    words_dir = os.path.join(settings.MEDIA_ROOT, 'words')
    os.makedirs(letters_dir, exist_ok=True)
    os.makedirs(words_dir, exist_ok=True)
    return letters_dir, words_dir

def process_text_into_signs(text):
    """
    Converts text into structured sign language data, saving letter and word images.
    Returns list of word data and flat list of letters.
    """
    letters_dir, words_dir = ensure_dirs()
    model = get_sign_model()
    
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
    words = clean_text.split()
    
    words_data = []
    all_letters = []

    for word in words:
        letter_objects = []
        letter_images = []
        
        for char in word:
            c = char.lower()
            img = generate_image_from_letter(model, c)
            letter_images.append(img)
            
            # Save individual letter sign image
            letter_filename = f"{c}.png"
            letter_filepath = os.path.join(letters_dir, letter_filename)
            if not os.path.exists(letter_filepath):
                img.save(letter_filepath)
                
            letter_url = f"{settings.MEDIA_URL}letters/{letter_filename}"
            letter_info = {
                'char': c.upper(),
                'image_url': letter_url
            }
            letter_objects.append(letter_info)
            all_letters.append(letter_info)

        # Save combined word banner
        word_filename = f"{word.lower()}.png"
        word_filepath = os.path.join(words_dir, word_filename)
        word_banner = combine_images_into_word(letter_images)
        word_banner.save(word_filepath)
        
        words_data.append({
            'word': word.upper(),
            'word_image': word_filepath,
            'word_image_url': f"{settings.MEDIA_URL}words/{word_filename}",
            'letters': letter_objects
        })

    return words_data, all_letters

def extract_audio_from_video(video_path, audio_path):
    """Extracts a 16kHz mono 16-bit PCM WAV audio track for Vosk speech recognition."""
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            logging.warning("The uploaded video file has no audio track.")
            return False

        # Export as 16kHz mono WAV suitable for Vosk
        video.audio.write_audiofile(
            audio_path,
            fps=16000,
            nbytes=2,
            codec='pcm_s16le',
            ffmpeg_params=['-ac', '1'],
            verbose=False,
            logger=None
        )
        return True
    except Exception as e:
        logging.error(f"Error extracting audio from video: {e}")
        return False

def transcribe_audio_to_text(audio_path):
    """Transcribes an audio WAV file using the Vosk Kaldi speech recognizer."""
    if not os.path.exists(audio_path):
        return ""
        
    if not os.path.exists(VOSK_MODEL_PATH):
        logging.warning(f"Vosk model not found at {VOSK_MODEL_PATH}")
        return ""

    try:
        wf = wave.open(audio_path, "rb")
        vosk_model = vosk.Model(VOSK_MODEL_PATH)
        recognizer = vosk.KaldiRecognizer(vosk_model, wf.getframerate())
        recognizer.SetWords(True)

        transcript_parts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                txt = res.get("text", "")
                if txt:
                    transcript_parts.append(txt)

        final_res = json.loads(recognizer.FinalResult())
        final_txt = final_res.get("text", "")
        if final_txt:
            transcript_parts.append(final_txt)

        wf.close()
        return " ".join(transcript_parts).strip()
    except Exception as e:
        logging.error(f"Error in Vosk transcription: {e}")
        return ""

def overlay_images_on_video(video_path, images_data, output_video_path):
    """Overlays sign language word banners on the original video."""
    try:
        video = VideoFileClip(video_path)
        if not images_data or video.duration <= 0:
            video.write_videofile(output_video_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            return output_video_path

        clips = [video]
        duration_per_image = max(0.5, video.duration / len(images_data))
        current_time = 0.0

        for word_data in images_data:
            word_image_path = word_data['word_image']
            if os.path.exists(word_image_path):
                img_clip = (
                    ImageClip(word_image_path)
                    .set_duration(min(duration_per_image, video.duration - current_time))
                    .set_start(current_time)
                    .resize(height=110)
                    .margin(bottom=20, opacity=0)
                    .set_pos(('center', 'bottom'))
                )
                clips.append(img_clip)
            current_time += duration_per_image
            if current_time >= video.duration:
                break

        final_video = CompositeVideoClip(clips)
        if video.audio:
            final_video = final_video.set_audio(video.audio)

        final_video.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        return output_video_path
    except Exception as e:
        logging.error(f"Error overlaying signs onto video: {e}")
        return video_path

def index(request):
    """Serves the main Voice2Gesture web interface."""
    return render(request, 'index.html')

@csrf_exempt
def upload_video(request):
    """
    Handles video uploads (both file uploads and live camera recordings).
    Extracts speech, generates sign language gestures, and overlays them onto the video.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method. Please use POST.'}, status=405)

    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No video file provided.'}, status=400)

    # Prepare media folders
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    
    # Save uploaded video
    filename = video_file.name
    # Standardize extension for camera recording blobs
    if not os.path.splitext(filename)[1]:
        filename += '.webm'
    
    video_path = os.path.join(settings.MEDIA_ROOT, filename)
    with open(video_path, 'wb+') as dest:
        for chunk in video_file.chunks():
            dest.write(chunk)

    # 1. Transcribe speech
    browser_transcript = request.POST.get('browser_transcript', '').strip()
    audio_path = os.path.join(settings.MEDIA_ROOT, 'extracted_audio.wav')
    has_audio = extract_audio_from_video(video_path, audio_path)
    
    vosk_transcript = transcribe_audio_to_text(audio_path) if has_audio else ""
    
    # Choose best transcript (prioritize non-empty, use browser live transcript as reliable partner)
    transcript = vosk_transcript if vosk_transcript else browser_transcript
    if not transcript:
        transcript = browser_transcript

    logging.info(f"Final Transcript: '{transcript}' (Vosk: '{vosk_transcript}', Browser: '{browser_transcript}')")

    if not transcript:
        return JsonResponse({
            'warning': 'No spoken speech could be detected in the video or recording. Please speak clearly into the microphone and try again.',
            'transcript': '',
            'words': [],
            'all_letters': [],
            'output_video_url': settings.MEDIA_URL + filename
        })

    # 2. Convert speech into Sign Language gestures
    words_data, all_letters = process_text_into_signs(transcript)

    # 3. Create video with sign language overlay
    output_video_filename = f"overlay_{os.path.splitext(filename)[0]}.mp4"
    output_video_path = os.path.join(settings.MEDIA_ROOT, output_video_filename)
    overlay_images_on_video(video_path, words_data, output_video_path)

    return JsonResponse({
        'success': True,
        'transcript': transcript,
        'output_video_url': f"{settings.MEDIA_URL}{output_video_filename}",
        'words': words_data,
        'all_letters': all_letters
    })

@csrf_exempt
def translate_text(request):
    """
    Translates raw text or live speech directly into sign language without requiring video encoding.
    """
    text = ""
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            text = body.get('text', '')
        except Exception:
            text = request.POST.get('text', '')
    else:
        text = request.GET.get('text', '')

    text = text.strip()
    if not text:
        return JsonResponse({'error': 'No text provided for translation.'}, status=400)

    words_data, all_letters = process_text_into_signs(text)
    return JsonResponse({
        'success': True,
        'transcript': text,
        'words': words_data,
        'all_letters': all_letters
    })
