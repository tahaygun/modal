# Insanely Fast Whisper Transcription

This Space provides a fast and efficient way to transcribe audio files using various Whisper models and optimization techniques. The application features real-time transcription with progress tracking and multiple model configurations.

## Features

- Multiple Whisper model configurations:
  - Whisper Large v3 (Standard, Optimized, Flash Attention)
  - Distil Large v2 (Optimized, Flash Attention)
  - Whisper Large v2 (Faster-8bit)
- Real-time transcription streaming
- Progress tracking for model loading and transcription
- Support for various audio formats (MP3, WAV, M4A, FLAC)
- Optimized performance with Flash Attention 2 and BetterTransformer

## Performance Metrics

Based on 150 minutes of audio:

- Whisper Large v3 (Standard): ~31 minutes
- Whisper Large v3 (Optimized): ~5 minutes
- Whisper Large v3 (Flash Attention): ~1.6 minutes
- Distil Large v2 (Optimized): ~3.3 minutes
- Distil Large v2 (Flash Attention): ~1.3 minutes
- Whisper Large v2 (Faster-8bit): ~8.2 minutes

## Usage

1. Upload an audio file using the file uploader
2. Select your preferred model configuration from the dropdown
3. Watch the real-time transcription progress
4. The transcribed text will appear gradually as it's processed

## Technical Details

- Built with Gradio
- Uses Hugging Face Transformers
- Optimized with Flash Attention 2 and BetterTransformer
- Supports both CPU and GPU inference

## License

MIT License
