import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SYSTEM_MESSAGE = "You will reply as shortly as possible. Preferably a single sentence."

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.005  # RMS level below which is considered silence
SILENCE_DURATION = 1.5     # Seconds of silence before stopping recording
CHUNK_DURATION = 2.0       # Duration in seconds for each audio chunk
OVERLAP_DURATION = 0.5     # Overlap between chunks to avoid missing words

# TTS settings
TTS_RATE = 1.0             # Speech rate multiplier
MIN_CHUNK_LENGTH = 10      # Minimum characters to speak
MAX_BUFFER_SIZE = 150      # Maximum buffer size before forcing speech

# Model settings
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_LANGUAGE = "en"
DEFAULT_LLM_MODEL = "gpt-4.1-nano"
MAX_OUTPUT_TOKENS = 128