import whisper

model = whisper.load_model("turbo")
result = model.transcribe(r"D:\Code\AI-Calling\data\hows-it-going.m4a")
print(result["text"])