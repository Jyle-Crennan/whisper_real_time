# Whisper Transcription for use in ASL Translation

This is a demo model of real time speech-to-text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings. The transcriptions are saved to a text file to be preprocessed for use in English-to-ASL natural language processing.

To install dependencies simply run:
```
pip install -r requirements.txt
```
in an environment of your choosing.

Whisper also requires the [`ffmpeg`](https://ffmpeg.org/) command-line tool to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

More information on Whisper: https://github.com/openai/whisper