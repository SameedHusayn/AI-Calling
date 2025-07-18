import asyncio
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import time
import json
from pathlib import Path
from faster_whisper import WhisperModel
from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
import threading
from queue import Queue
import subprocess
import platform

load_dotenv() 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
system_message = "You will reply as shortly as possible. Preferably a single sentence."

class PerformanceTracker:
    """Utility class to track performance metrics"""
    def __init__(self):
        self.stages = {}
        self.current_stage = None
        self.current_stage_start = None
    
    def start_stage(self, stage_name):
        """Start timing a new stage"""
        if self.current_stage:
            self.end_stage()
        self.current_stage = stage_name
        self.current_stage_start = time.time()
        print(f"⏱️ Starting stage: {stage_name}")
        return self.current_stage_start
    
    def end_stage(self):
        """End timing the current stage"""
        if self.current_stage and self.current_stage_start:
            duration = time.time() - self.current_stage_start
            self.stages[self.current_stage] = duration
            print(f"✅ Completed stage: {self.current_stage} in {duration:.3f}s")
            self.current_stage = None
            self.current_stage_start = None
            return duration
        return 0
    
    def get_summary(self):
        """Get a summary of all timed stages"""
        return {
            "stages": self.stages,
            "total_time": sum(self.stages.values()),
            "longest_stage": max(self.stages.items(), key=lambda x: x[1]) if self.stages else None
        }

class StreamingTTS:
    """A streaming TTS class that can speak text incrementally"""
    def __init__(self, rate=1.0):
        self.rate = rate
        self.system = platform.system()
        print(f"Initializing StreamingTTS on {self.system} platform")
        
        # For tracking the sentence-in-progress
        self.text_buffer = ""
        self.min_chunk_length = 10  # Minimum characters to speak
        self.max_buffer_size = 150  # Maximum buffer size before forcing speech
        
        # Semaphore to control concurrent TTS calls
        self.tts_semaphore = threading.Semaphore(1)
    
    def speak_chunk(self, text, final=False):
        """Speak a chunk of text, buffering until we have a good stopping point"""
        if not text and not final:
            return False
            
        # Add this text to buffer
        self.text_buffer += text
        
        # If buffer is empty, nothing to do
        if not self.text_buffer:
            return True
            
        # Only speak if we have enough text, hit a sentence boundary, or this is the final chunk
        should_speak = (
            len(self.text_buffer) >= self.max_buffer_size or
            final or
            (len(self.text_buffer) > self.min_chunk_length and 
             any(p in self.text_buffer for p in ['.', '!', '?', ';', ':', '\n']))
        )
        
        if should_speak:
            # Launch in a thread to avoid blocking
            threading.Thread(
                target=self._speak_text_now,
                args=(self.text_buffer,),
                daemon=True
            ).start()
            # Clear the buffer after speaking
            self.text_buffer = ""
            
        return True
    
    def _speak_text_now(self, text):
        """Immediately speak the given text"""
        if not text:
            return
        
        # Acquire semaphore to limit concurrent TTS operations
        with self.tts_semaphore:
            if self.system == "Windows":
                self._speak_windows(text)
    
    def _speak_windows(self, text):
        """Use PowerShell to speak on Windows with better character escaping"""
        try:
            # Create a temporary file to avoid command line escaping issues
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w', encoding='utf-8') as f:
                f.write(text)
                temp_file = f.name
            
            # PowerShell command to read from file and speak
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; ' \
                  f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; ' \
                  f'$speak.Rate = {int((self.rate - 1) * 10)}; ' \
                  f'$speak.Speak([System.IO.File]::ReadAllText(\'{temp_file}\'));"'
            
            print(f"🔊 Speaking: \"{text}\"")
            subprocess.call(cmd, shell=True)
            
            # Clean up temp file
            os.unlink(temp_file)
            return True
        except Exception as e:
            print(f"Error with Windows TTS: {e}")
            return False

class ConversationSystem:
    def __init__(self, openai_api_key=OPENAI_API_KEY, model_size="base", language="en", 
                 llm_model="gpt-4.1-nano", silence_threshold=0.005, silence_duration=1.5,
                 tts_rate=1.0):
        # Performance tracking
        self.tracker = PerformanceTracker()
        self.tracker.start_stage("initialization")
        
        # Speech-to-text components
        self.model_size = model_size
        self.language = language
        self.sample_rate = 16000
        self.channels = 1
        self.is_recording = False
        self.frames = []
        self.transcriptions = []
        self.temp_dir = tempfile.mkdtemp()
        
        # Silence detection parameters
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.last_sound_time = 0
        self.auto_stop = False
        
        # Chunk processing parameters
        self.chunk_duration = 2.0
        self.overlap_duration = 0.5
        
        # LLM components
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.async_openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.conversation_history = [{"role": "system", "content": system_message}]
        
        # TTS components
        self.tts = StreamingTTS(rate=tts_rate)
        
        print(f"Using temporary directory: {self.temp_dir}")
        
        # Initialize the Whisper model directly
        print(f"Loading Whisper {model_size} model...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"Model loaded successfully! Using {llm_model} for responses.")
        
        self.tracker.end_stage()  # End initialization
    
    def audio_callback(self, indata, frames, time_info, status):
        """Store audio chunks and detect silence"""
        if status:
            print(f"Audio status: {status}")
            
        self.frames.append(indata.copy())
        
        # Check if there's sound in this chunk
        if self.auto_stop and self.is_recording:
            rms = np.sqrt(np.mean(indata**2))
                
            # If amplitude is above threshold, update last_sound_time
            if rms > self.silence_threshold:
                self.last_sound_time = time.time()
            else:
                # Check if silence has persisted for the specified duration
                silence_time = time.time() - self.last_sound_time
                if silence_time >= self.silence_duration and self.last_sound_time > 0:
                    print(f"Detected {silence_time:.2f}s of silence. Auto-stopping recording.")
                    self.is_recording = False
    
    async def process_audio_chunks(self):
        """Save audio chunks to file and transcribe periodically"""
        chunk_counter = 0
        buffer = np.array([], dtype=np.float32)
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        overlap_samples = int(self.sample_rate * self.overlap_duration)
        
        while self.is_recording:
            # Get all new frames
            if self.frames:
                new_frames = self.frames.copy()
                self.frames = []
                
                # Convert and append to buffer
                for frame in new_frames:
                    buffer = np.append(buffer, frame.flatten())
                
                # Process in chunks with overlap
                if len(buffer) >= chunk_samples:
                    # Extract chunk
                    chunk = buffer[:chunk_samples]
                    # Keep overlap portion for next chunk
                    buffer = buffer[chunk_samples - overlap_samples:]
                    
                    # Save to temporary WAV file
                    chunk_file = os.path.join(self.temp_dir, f"chunk_{chunk_counter}.wav")
                    sf.write(chunk_file, chunk, self.sample_rate)
                    
                    # Process the chunk in a separate task
                    asyncio.create_task(self.transcribe_file(chunk_file))
                    chunk_counter += 1
            
            await asyncio.sleep(0.1)
            
        # Process any remaining audio in the buffer
        if len(buffer) > 0:
            # Save to temporary WAV file
            final_chunk = os.path.join(self.temp_dir, f"chunk_final.wav")
            sf.write(final_chunk, buffer, self.sample_rate)
            await self.transcribe_file(final_chunk)
    
    async def transcribe_file(self, file_path):
        """Transcribe an audio file using Whisper"""
        try:
            chunk_name = os.path.basename(file_path)
            self.tracker.start_stage(f"transcribe_{chunk_name}")
            print(f"Transcribing {chunk_name}...")
            
            # Run transcription in a thread pool
            segments, _ = await asyncio.to_thread(
                self.model.transcribe, 
                file_path, 
                language=self.language,
                vad_filter=True
            )
            
            # Process the segments
            text = ""
            for segment in segments:
                text += segment.text + " "
            
            text = text.strip()
            if text:
                print(f"Transcription: {text}")
                # Store with timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                self.transcriptions.append({
                    "timestamp": timestamp,
                    "text": text
                })
                
            # Cleanup temp file
            os.remove(file_path)
            self.tracker.end_stage()  # End transcription timing
            
        except Exception as e:
            print(f"Error transcribing file {file_path}: {e}")
            self.tracker.end_stage()  # End timing even if there was an error
    
    async def start_recording(self, auto_stop=True):
        """Start recording audio"""
        self.tracker.start_stage("audio_recording")
        
        self.is_recording = True
        self.frames = []  # Clear any old frames
        self.transcriptions = []  # Clear previous transcriptions
        self.auto_stop = auto_stop
        self.last_sound_time = time.time()  # Initialize with current time
        
        # Start processing task
        process_task = asyncio.create_task(self.process_audio_chunks())
        
        # Start audio recording
        print("Starting audio recording...")
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype='float32',
            blocksize=int(self.sample_rate * 0.1)
        )
        self.stream.start()
        
        print(f"Recording started! Speak into your microphone...")
        print(f"Will auto-stop after {self.silence_duration} seconds of silence")
        
        # Check recording status in a loop
        check_task = asyncio.create_task(self._check_recording_status())
        
        # If not using auto-stop, wait for user input
        if not auto_stop:
            await asyncio.get_event_loop().run_in_executor(None, input)
            self.is_recording = False
        
        # Wait for the check task
        await check_task
        
        # Cleanup
        self.stream.stop()
        self.stream.close()
        process_task.cancel()
        
        self.tracker.end_stage()  # End recording timing
    
    async def _check_recording_status(self):
        """Check if recording should be stopped"""
        while self.is_recording:
            await asyncio.sleep(0.1)
    
    async def process_with_llm(self):
        """Process the transcribed text with OpenAI's GPT-4.1 Nano using streaming"""
        try:
            # Get the full transcription
            user_input = self.get_full_text()
            if not user_input:
                print("No transcription to process.")
                return None
            
            # Start LLM timing    
            self.tracker.start_stage("llm_processing")
            print(f"\nSending to {self.llm_model}: \"{user_input}\"")
            
            # Prepare messages with conversation history
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": user_input})
            
            # First token timing
            first_token_received = False
            first_token_time = None
            
            # Full response collection
            full_response = ""
            
            # Stream the response
            stream = await self.async_openai_client.responses.create(
                model=self.llm_model,
                input=messages,
                max_output_tokens=128,
                stream=True
            )
            
            # Process the streaming response
            print("\nGPT Response: ", end="", flush=True)
            
            async for chunk in stream:
                # Track first token time
                if not first_token_received:
                    first_token_time = time.time()
                    print("\n⏱️ First token received in", 
                          f"{first_token_time - self.tracker.current_stage_start:.2f}s")
                    first_token_received = True
                
                # Get the chunk text
                if hasattr(chunk, 'output_text'):
                    chunk_text = chunk.output_text
                    if chunk_text:
                        print(chunk_text, end="", flush=True)
                        full_response += chunk_text
                        
                        # Send to TTS in real-time
                        self.tts.speak_chunk(chunk_text)
            
            # Make sure to speak any remaining text
            self.tts.speak_chunk("", final=True)
            
            print()  # New line after streaming completes
            llm_time = self.tracker.end_stage()  # End LLM timing
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            print(f"\nFull GPT Response ({llm_time:.2f}s):\n{full_response}")
            
            return full_response
            
        except Exception as e:
            print(f"Error processing with LLM: {e}")
            import traceback
            traceback.print_exc()
            self.tracker.end_stage()  # End LLM timing even if there was an error
            return None
    
    def get_full_text(self):
        """Get the full transcribed text"""
        return " ".join([item["text"] for item in self.transcriptions])
    
    def save_conversation(self, filename="conversation.json"):
        """Save the conversation to a JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "transcriptions": self.transcriptions,
                "conversation": self.conversation_history,
                "performance_metrics": self.tracker.get_summary()
            }, f, indent=2)
        print(f"Conversation saved to {filename}")
    
    def cleanup(self):
        """Clean up temporary files"""
        for file in Path(self.temp_dir).glob("chunk_*.wav"):
            try:
                os.remove(file)
            except:
                pass
        try:
            os.rmdir(self.temp_dir)
            print(f"Cleaned up temporary directory {self.temp_dir}")
        except:
            print(f"Could not remove temporary directory {self.temp_dir}")


async def main():
    # Set up overall timing
    start_time = time.time()
    
    # Get API key from environment variable or direct input
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    
    # Create conversation system with TTS
    system = ConversationSystem(
        openai_api_key=api_key,
        model_size="base",
        language="en",
        llm_model="gpt-4.1-nano",  # Using the fastest model
        silence_threshold=0.005,
        silence_duration=1.5,
        tts_rate=1.0  # Normal speech rate
    )
    
    try:
        conversation_count = 0
        while True:
            conversation_count += 1
            print(f"\n🔄 Starting conversation turn #{conversation_count}")
            turn_start = time.time()
            
            # Start recording
            await system.start_recording(auto_stop=True)
            
            # Give some time for final processing
            print("Finalizing transcriptions...")
            await asyncio.sleep(1)
            
            # Print the full text
            print("\nComplete transcription:")
            print(system.get_full_text())
            
            # Process with LLM if we have a transcription
            if system.transcriptions:
                await system.process_with_llm()
                
                # Small delay to ensure final TTS processing
                await asyncio.sleep(1)
            
            # Calculate turn duration
            turn_duration = time.time() - turn_start
            print(f"\n⏱️ Turn #{conversation_count} completed in {turn_duration:.2f}s")
            
            # Ask if user wants to continue
            print("\nDo you want to ask another question? (y/n)")
            response = await asyncio.get_event_loop().run_in_executor(None, input)
            if response.lower() != 'y':
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Calculate and display total runtime
        total_runtime = time.time() - start_time
        print(f"\n🕒 Total runtime: {total_runtime:.2f}s")
        
        # Display performance metrics
        print("\n📊 Performance Summary:")
        summary = system.tracker.get_summary()
        
        if summary["longest_stage"]:
            stage, duration = summary["longest_stage"]
            print(f"- Longest stage: {stage} ({duration:.2f}s)")
            
        print(f"- Total timed operations: {summary['total_time']:.2f}s")
        
        # Show metrics for each stage
        print("\nDetailed stage timings:")
        for stage, duration in sorted(summary["stages"].items(), key=lambda x: x[1], reverse=True):
            print(f"- {stage}: {duration:.2f}s")
        
        # Save conversation and cleanup
        system.save_conversation()
        system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())