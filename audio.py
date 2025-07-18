"""Audio recording and processing"""
import numpy as np
import sounddevice as sd
import asyncio
import time
import os
from config import SAMPLE_RATE, CHANNELS, SILENCE_THRESHOLD, SILENCE_DURATION

class AudioRecorder:
    """Class to handle audio recording and silence detection"""
    
    def __init__(self, silence_threshold=SILENCE_THRESHOLD, silence_duration=SILENCE_DURATION):
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        self.is_recording = False
        self.frames = []
        self.last_sound_time = 0
        self.auto_stop = False
        self.stream = None
        
        # Tracking if speech has been detected
        self.speech_detected = False
        
        # For real-time transcription
        self.processing_queue = asyncio.Queue()
        self.transcription_results = []
        
    def audio_callback(self, indata, frames, time_info, status):
        """Store audio chunks and detect silence"""
        if status:
            print(f"Audio status: {status}")
            
        # Add to frames buffer for processing
        self.frames.append(indata.copy())
        
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(indata**2))
        
        # Check if there's sound in this chunk
        if self.auto_stop and self.is_recording:
            # If amplitude is above threshold, we've detected speech
            if rms > self.silence_threshold:
                if not self.speech_detected:
                    self.speech_detected = True
                    print("🎙️ Speech detected, listening...")
                
                # Update last sound time only if speech has been detected
                self.last_sound_time = time.time()
            else:
                # Only check for silence after speech has been detected
                if self.speech_detected:
                    silence_time = time.time() - self.last_sound_time
                    if silence_time >= self.silence_duration:
                        print(f"Detected {silence_time:.2f}s of silence. Auto-stopping recording.")
                        self.is_recording = False
    
    async def start_recording(self, auto_stop=True):
        """Start recording audio"""
        self.is_recording = True
        self.frames = []  # Clear any old frames
        self.auto_stop = auto_stop
        self.last_sound_time = time.time()  # Initialize with current time
        self.speech_detected = False  # Reset speech detection flag
        self.transcription_results = []  # Reset transcription results
        
        # Start audio recording
        print("Starting audio recording...")
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype='float32',
            blocksize=int(self.sample_rate * 0.5)  # 500ms blocks for better processing
        )
        self.stream.start()
        
        print(f"Recording started! Speak into your microphone...")
        print(f"Waiting for speech, then will auto-stop after {self.silence_duration} seconds of silence")
        
        # Check recording status in a loop
        await self._check_recording_status()
        
        # Cleanup
        self.stream.stop()
        self.stream.close()
    
    async def _check_recording_status(self):
        """Check if recording should be stopped"""
        while self.is_recording:
            await asyncio.sleep(0.1)
    
    async def process_audio_realtime(self, transcriber):
        """Process audio frames in real-time through the transcriber"""
        buffer_size = int(self.sample_rate * 2)  # 2-second buffer
        buffer = np.array([], dtype=np.float32)
        last_process_time = time.time()
        
        # Process while recording
        while self.is_recording:
            # Get all new frames
            if self.frames:
                new_frames = self.frames.copy()
                self.frames = []
                
                # Convert and append to buffer
                for frame in new_frames:
                    # Ensure we're dealing with the right shape - flatten if multi-channel
                    if len(frame.shape) > 1 and frame.shape[1] > 1:
                        # Just take the first channel if we have multiple
                        frame_data = frame[:, 0]
                    else:
                        frame_data = frame.flatten()
                    
                    buffer = np.append(buffer, frame_data)
            
            # Process buffer if it's large enough or enough time has passed
            current_time = time.time()
            if len(buffer) >= buffer_size or (current_time - last_process_time > 2.0 and len(buffer) > 0):
                # Send to transcriber
                text = await transcriber.process_audio_chunk(buffer)
                
                # Store result if we got text
                if text:
                    self.transcription_results.append({
                        "timestamp": time.time(),
                        "text": text
                    })
                
                # Clear buffer and reset timer
                buffer = np.array([], dtype=np.float32)
                last_process_time = current_time
                
            await asyncio.sleep(0.1)
            
        # Process any remaining audio
        if len(buffer) > 0:
            text = await transcriber.process_audio_chunk(buffer)
            if text:
                self.transcription_results.append({
                    "timestamp": time.time(),
                    "text": text
                })
                
        # Final cleanup - process any audio that might be in the transcriber's buffer
        text = await transcriber.finalize()
        if text:
            self.transcription_results.append({
                "timestamp": time.time(),
                "text": text
            })
        
    def get_transcription_results(self):
        """Get the list of transcription results"""
        return self.transcription_results