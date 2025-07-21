import asyncio
import json
import time
from voice_assistant.performance import PerformanceTracker
from voice_assistant.audio import AudioRecorder
from voice_assistant.tts import SeamlessTTS  # Changed import
from voice_assistant.llm import LLMProcessor
from voice_assistant.config import OPENAI_API_KEY, SYSTEM_MESSAGE

class ConversationSystem:
    """Main conversation system that ties everything together"""
    
    def __init__(self, openai_api_key=OPENAI_API_KEY, model_size="tiny.en", language="en",
                 llm_model="gpt-4.1-nano", silence_threshold=0.005, silence_duration=1.5,
                 tts_rate=1.0):
        # Performance tracking
        self.tracker = PerformanceTracker()
        self.tracker.start_stage("initialization")
        
        # Initialize components
        self.tts = SeamlessTTS(rate=tts_rate)  # Changed to new TTS class
        self.audio_recorder = AudioRecorder(silence_threshold, silence_duration)
        self.llm = LLMProcessor(openai_api_key, llm_model)
        
        # Configuration
        self.model_size = model_size
        self.language = language
        
        # Conversation state
        self.conversation_history = [{"role": "system", "content": SYSTEM_MESSAGE}]
        
        self.tracker.end_stage()  # End initialization
        
    async def start_conversation_turn(self):
        """Run a single turn of conversation"""
        # Start timing
        turn_start = time.time()
        
        # Start recording audio and processing it in real-time
        self.tracker.start_stage("audio_recording")
        await self.audio_recorder.start_recording(auto_stop=True)
        self.tracker.end_stage()
        
        # Get the transcribed text
        full_text = self.audio_recorder.get_transcribed_text()
        
        # Print the full text
        print("\nComplete transcription:")
        print(full_text)
        
        # Process with LLM if we have a transcription
        response = None
        if full_text:
            # Prepare messages
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": full_text})
            
            # Process with LLM
            response = await self.llm.process_stream(messages, self.tts, self.tracker)
            
            if response:
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": full_text})
                self.conversation_history.append({"role": "assistant", "content": response})
            
            # Small delay to ensure TTS has time to process
            await asyncio.sleep(1)
        
        # Calculate and report turn duration
        turn_duration = time.time() - turn_start
        print(f"\n⏱️ Conversation turn completed in {turn_duration:.2f}s")
        
        return response
        
    def save_conversation(self, filename="conversation.json"):
        """Save the conversation to a JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "conversation": self.conversation_history,
                "performance_metrics": self.tracker.get_summary()
            }, f, indent=2)
        print(f"Conversation saved to {filename}")
        
    def cleanup(self):
        """Clean up resources"""
        pass