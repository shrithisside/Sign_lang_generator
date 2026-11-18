import os
import re
import json
import logging
import requests

logger = logging.getLogger(__name__)

# Check for SpeechRecognition
try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

# Check for deep_translator
try:
    from deep_translator import MyMemoryTranslator, GoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    HAS_DEEP_TRANSLATOR = False

# Devanagari Unicode Range: \u0900 to \u097F
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')

# Common Distinct Marathi Vocabulary for Auto-Detection
MARATHI_MARKERS = {
    'नमस्कार', 'आभार', 'कसा', 'कशी', 'कसे', 'आहात', 'आहे', 'आहेत', 'नाही',
    'मी', 'आम्ही', 'तू', 'तुम्ही', 'तो', 'ती', 'ते', 'काय', 'कुठे', 'कधी',
    'कशासाठी', 'छान', 'पाणी', 'जेवण', 'नाव', 'माझे', 'माझं', 'तुमचे', 'तुमचं',
    'होय', 'करा', 'बोला', 'पहा', 'बघा', 'झाले', 'वेळ', 'दिवस', 'खूप', 'थोडं'
}

# Extensive Offline Hindi & Marathi to English Dictionary
COMMON_INDIC_DICT = {
    # Greetings & Common Expressions
    'नमस्ते': 'hello', 'नमस्कार': 'hello', 'प्रणाम': 'greetings', 'सुप्रभात': 'good morning',
    'शुभ': 'good', 'संध्या': 'evening', 'रात्री': 'night', 'रात्र': 'night', 'अलविदा': 'goodbye',
    # Politeness
    'धन्यवाद': 'thank you', 'शुक्रिया': 'thank you', 'आभार': 'thank you', 'कृपया': 'please',
    'माफ': 'sorry', 'क्षमा': 'excuse me', 'स्वागत': 'welcome',
    # Pronouns
    'मैं': 'i', 'मी': 'i', 'हम': 'we', 'आम्ही': 'we',
    'आप': 'you', 'तुम्ही': 'you', 'तुम': 'you', 'तू': 'you',
    'वह': 'he', 'तो': 'he', 'ती': 'she', 'ते': 'they', 'वे': 'they',
    'यह': 'this', 'हे': 'this', 'हा': 'this', 'ही': 'this',
    'मेरा': 'my', 'मेरी': 'my', 'मेरे': 'my', 'माझे': 'my', 'माझा': 'my', 'माझी': 'my', 'माझं': 'my',
    'आपका': 'your', 'आपकी': 'your', 'आपके': 'your', 'तुमचे': 'your', 'तुमचा': 'your', 'तुमची': 'your', 'तुमचं': 'your',
    'तेरा': 'your', 'तुझा': 'your', 'तुझी': 'your',
    # Questions
    'क्या': 'what', 'काय': 'what',
    'कैसे': 'how', 'कैसा': 'how', 'कैसी': 'how', 'कसा': 'how', 'कशी': 'how', 'कसे': 'how',
    'क्यों': 'why', 'का': 'why', 'कशासाठी': 'why',
    'कहाँ': 'where', 'कुठे': 'where',
    'कब': 'when', 'कधी': 'when',
    'कौन': 'who', 'कोण': 'who',
    'कितना': 'how much', 'कितने': 'how many', 'किती': 'how much',
    # Helping verbs & State
    'है': 'is', 'आहे': 'is',
    'हैं': 'are', 'आहात': 'are', 'आहेत': 'are',
    'हूँ': 'am', 'आहे': 'am',
    'था': 'was', 'होता': 'was', 'थी': 'was', 'होती': 'was', 'थे': 'were', 'होते': 'were',
    'होगा': 'will be', 'असेल': 'will be',
    # Responses
    'हाँ': 'yes', 'हो': 'yes', 'होय': 'yes',
    'नहीं': 'no', 'ना': 'no', 'नाही': 'no',
    'अच्छा': 'good', 'छान': 'good', 'बढ़िया': 'great', 'सुंदर': 'beautiful',
    'बुरा': 'bad', 'वाईट': 'bad',
    'बहुत': 'very', 'खूप': 'very',
    'कम': 'less', 'थोड़ा': 'little', 'थोडे': 'little',
    # Common Nouns
    'नाम': 'name', 'नाव': 'name',
    'दोस्त': 'friend', 'मित्र': 'friend',
    'मदद': 'help', 'सहायता': 'help',
    'पानी': 'water', 'पाणी': 'water',
    'खाना': 'food', 'जेवण': 'food',
    'चाय': 'tea', 'चहा': 'tea',
    'घर': 'home',
    'काम': 'work',
    'समय': 'time', 'वेळ': 'time',
    'दिन': 'day', 'दिवस': 'day',
    'भारत': 'india', 'महाराष्ट्र': 'maharashtra',
    'स्कूल': 'school', 'शाळा': 'school',
    'कॉलेज': 'college',
    'किताब': 'book', 'पुस्तक': 'book',
    'बात': 'talk', 'गोष्ट': 'story',
    # Verbs
    'करो': 'do', 'करा': 'do',
    'बोलो': 'speak', 'बोला': 'speak',
    'सुनो': 'listen', 'ऐका': 'listen',
    'देखो': 'look', 'पहा': 'look', 'बघा': 'look',
    'जाओ': 'go', 'जा': 'go',
    'आओ': 'come', 'या': 'come',
    'चाहिए': 'want', 'हवे': 'want'
}

def is_devanagari(text):
    """Checks whether the text contains Devanagari script characters."""
    if not text:
        return False
    return bool(DEVANAGARI_REGEX.search(text))

def detect_language(text, hint='auto'):
    """
    Detects whether text is Hindi ('hi'), Marathi ('mr'), or English ('en').
    Uses optional hint or lexical markers.
    """
    if hint in ['hi', 'mr', 'en']:
        return hint
        
    if not is_devanagari(text):
        return 'en'
        
    # Check if text contains Marathi-specific vocabulary
    words = set(re.findall(r'[\u0900-\u097F]+', text))
    if words.intersection(MARATHI_MARKERS):
        return 'mr'
    return 'hi'

def devanagari_to_latin(text):
    """
    Transliterates Devanagari script into phonetic Latin letters.
    Essential for fingerspelling untranslated proper nouns, names, and regional terms.
    """
    vowels = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
        'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'am', 'अः': 'ah'
    }
    matras = {
        'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
        'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ँ': 'n', 'ः': 'h', '्': ''
    }
    consonants = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'ळ': 'l', 'क्ष': 'ksh', 'ज्ञ': 'gy'
    }
    
    res = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in vowels:
            res.append(vowels[char])
        elif char in consonants:
            base = consonants[char]
            if i + 1 < n and text[i + 1] == '्':
                res.append(base)
                i += 1
            elif i + 1 < n and text[i + 1] in matras:
                res.append(base + matras[text[i + 1]])
                i += 1
            else:
                res.append(base + 'a')
        elif char in matras:
            res.append(matras[char])
        else:
            res.append(char)
        i += 1
        
    return ''.join(res)

def translate_to_english(text, source_lang='auto'):
    """
    Translates input text (Hindi, Marathi, or English) into clean English.
    Applies multi-tiered translation strategies:
    1. Online MyMemory / Google Translator
    2. Fallback offline Indic dictionary
    3. Devanagari-to-Latin phonetic transliteration for names / OOV words
    """
    if not text or not text.strip():
        return "", "en"

    clean_text = text.strip()
    
    # If text has no Devanagari script and user indicated English or auto, pass through
    if not is_devanagari(clean_text) and source_lang in ['en', 'auto']:
        return clean_text, "en"

    detected_lang = detect_language(clean_text, hint=source_lang)
    
    # Strategy 1: MyMemory Translator
    if HAS_DEEP_TRANSLATOR:
        src_code = 'hi-IN' if detected_lang == 'hi' else 'mr-IN'
        try:
            translated = MyMemoryTranslator(source=src_code, target='en-GB').translate(clean_text)
            if translated and translated.strip():
                logger.info(f"MyMemory translation ({detected_lang} -> en): '{clean_text}' -> '{translated}'")
                return translated.strip(), detected_lang
        except Exception as e:
            logger.warning(f"MyMemory translation failed: {e}")

    # Strategy 2: Google Translator via deep_translator or public endpoint
    if HAS_DEEP_TRANSLATOR:
        try:
            g_src = 'hi' if detected_lang == 'hi' else 'mr'
            translated = GoogleTranslator(source=g_src, target='en').translate(clean_text)
            if translated and translated.strip():
                logger.info(f"GoogleTranslator ({detected_lang} -> en): '{clean_text}' -> '{translated}'")
                return translated.strip(), detected_lang
        except Exception as e:
            logger.warning(f"GoogleTranslator failed: {e}")

    # Strategy 3: Public Google Translate API endpoint via requests
    try:
        g_src = 'hi' if detected_lang == 'hi' else 'mr'
        url = 'https://translate.googleapis.com/translate_a/single'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        params = {'client': 'gtx', 'sl': g_src, 'tl': 'en', 'dt': 't', 'q': clean_text}
        resp = requests.get(url, params=params, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            translated = ''.join([part[0] for part in data[0] if part and part[0]])
            if translated and translated.strip():
                logger.info(f"GTX translation ({detected_lang} -> en): '{clean_text}' -> '{translated}'")
                return translated.strip(), detected_lang
    except Exception as e:
        logger.warning(f"GTX endpoint translation failed: {e}")

    # Strategy 4: Offline Dictionary Lookup & Phrase Replacement
    words = clean_text.split()
    translated_tokens = []
    for w in words:
        # Strip punctuation for lookup
        stripped = re.sub(r'[^\u0900-\u097F]', '', w)
        if stripped in COMMON_INDIC_DICT:
            translated_tokens.append(COMMON_INDIC_DICT[stripped])
        elif is_devanagari(w):
            # Transliterate proper nouns / unrecognized words to Latin characters
            translated_tokens.append(devanagari_to_latin(w))
        else:
            translated_tokens.append(w)

    fallback_result = ' '.join(translated_tokens)
    logger.info(f"Fallback dictionary translation ({detected_lang} -> en): '{clean_text}' -> '{fallback_result}'")
    return fallback_result, detected_lang

def transcribe_multilingual_audio(audio_path, language='en'):
    """
    Transcribes audio supporting English ('en'), Hindi ('hi'), and Marathi ('mr').
    Uses SpeechRecognition with language codes, or falls back to offline Vosk.
    """
    if not os.path.exists(audio_path):
        return ""

    lang_code_map = {
        'hi': 'hi-IN',
        'mr': 'mr-IN',
        'en': 'en-US'
    }
    target_lang_code = lang_code_map.get(language, 'en-US')

    # 1. Try SpeechRecognition (Google Speech Recognition API)
    if HAS_SR:
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = r.record(source)
                logger.info(f"Transcribing audio with SpeechRecognition (lang={target_lang_code})...")
                transcript = r.recognize_google(audio_data, language=target_lang_code)
                if transcript and transcript.strip():
                    logger.info(f"SpeechRecognition result: '{transcript}'")
                    return transcript.strip()
        except Exception as e:
            logger.warning(f"SpeechRecognition failed for {target_lang_code}: {e}")

    return ""
