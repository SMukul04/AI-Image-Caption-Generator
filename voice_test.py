from gtts import gTTS
import os

try:

    text = "Hello. I am your AI assistant."

    print("Creating speech...")

    tts = gTTS(text=text, lang='en')

    print("Saving file...")

    tts.save("voice.mp3")

    print("File saved successfully!")

    print("Opening audio...")

    os.system("start voice.mp3")

except Exception as e:

    print("ERROR:")
    print(e)