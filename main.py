import asyncio
import time
import signal
from conversation import ConversationSystem
from config import OPENAI_API_KEY

# Global flag for graceful shutdown
running = True

async def main():
    
    # Set up overall timing
    start_time = time.time()
    
    # Get API key from environment variable or direct input
    api_key = OPENAI_API_KEY
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    
    # Create conversation system
    system = ConversationSystem(
        openai_api_key=api_key,
        model_size="tiny.en",
        language="en",
        llm_model="gpt-4.1-nano",
        silence_threshold=0.005,
        silence_duration=1.5,
        tts_rate=1.0
    )
    
    try:
        conversation_count = 0
        
        # Instructions for the user
        print("\n🎙️ Voice Assistant Ready!")
        print("Start speaking whenever you're ready")
        print("Press Ctrl+C to exit the conversation at any time")
        
        # Continuous conversation loop
        global running
        while running:
            conversation_count += 1
            print(f"\n🔄 Starting conversation turn #{conversation_count}")
            
            # Run a conversation turn
            await system.start_conversation_turn()
            
            # Small pause between turns
            await asyncio.sleep(0.5)
            print("\n👂 Listening for your next question...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Calculate and display total runtime
        total_runtime = time.time() - start_time
        print(f"\n🕒 Total runtime: {total_runtime:.2f}s")
        
        # Display performance metrics
        system.tracker.print_summary()
        
        # Save conversation and cleanup
        # system.save_conversation()
        system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())