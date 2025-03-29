# Insanely Fast Whisper Gradio App

This is a Gradio application that uses the insanely-fast-whisper model for high-speed speech recognition. The application provides a simple interface to upload audio files and get their transcriptions.

## Features

- Fast transcription using Whisper Large v3
- Support for various audio formats (MP3, WAV, etc.)
- Simple and intuitive Gradio interface
- Ready to deploy on Hugging Face Spaces

## Local Installation

1. Clone this repository
2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

## Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose "Gradio" as the SDK
3. Push this repository to your Space
4. The Space will automatically build and deploy the application

## Usage

1. Open the application in your browser
2. Upload an audio file using the file uploader
3. Wait for the transcription to complete
4. The transcribed text will appear in the output box

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- See requirements.txt for all dependencies

## License

MIT License
