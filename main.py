"""Main entry point for the voice assistant"""
import asyncio
import time
from conversation import ConversationSystem
from config import OPENAI_API_KEY

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
        model_size="base",
        language="en",
        llm_model="gpt-4.1-nano",
        silence_threshold=0.005,
        silence_duration=1.5,
        tts_rate=1.0
    )
    
    try:
        conversation_count = 0
        while True:
            conversation_count += 1
            print(f"\n🔄 Starting conversation turn #{conversation_count}")
            
            # Run a conversation turn
            await system.start_conversation_turn()
            
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
        system.tracker.print_summary()
        
        # Save conversation and cleanup
        system.save_conversation()
        system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())