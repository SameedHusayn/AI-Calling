import time
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, MAX_OUTPUT_TOKENS

class LLMProcessor:
    """Class for handling language model interactions"""
    
    def __init__(self, api_key=OPENAI_API_KEY, model=DEFAULT_LLM_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        
    async def process_stream(self, messages, tts, tracker=None):
        """Process text with the language model using streaming"""
        try:
            # Start timing if we have a tracker
            if tracker:
                tracker.start_stage("llm_processing")
                
            print(f"\nSending to {self.model}: \"{messages[-1]['content']}\"")
            
            # First token timing
            first_token_received = False
            first_token_time = None
            tracker_start_time = time.time() if tracker else None
            
            # Full response collection
            full_response = ""
            # Stream the response
            stream = await self.client.responses.create(
                model=self.model,
                input=messages,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                stream=True
            )
            
            # Process the streaming response
            print("\nGPT Response:")
            
            async for chunk in stream:
                # Track first token time
                if not first_token_received:
                    first_token_time = time.time()
                    time_diff = first_token_time - tracker_start_time if tracker_start_time else 0
                    print(f"⏱️ First token received in {time_diff:.2f}s")
                    first_token_received = True
                
                # Extract the text from the chunk based on its type
                if hasattr(chunk, 'type') and chunk.type == "response.output_text.delta":
                    if hasattr(chunk, 'delta'):
                        chunk_text = chunk.delta or ""
                        if chunk_text:
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                            
                            # Send to TTS in real-time
                            tts.speak_chunk(chunk_text)
            
            print()  # New line after streaming completes
            
            # Make sure to speak any remaining text
            tts.speak_chunk("", final=True)
            
            if tracker:
                tracker.end_stage()  # End LLM timing
                
            return full_response
            
        except Exception as e:
            print(f"Error processing with LLM: {e}")
            import traceback
            traceback.print_exc()
            if tracker:
                tracker.end_stage()  # End LLM timing even if there was an error
            return None