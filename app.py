import gradio as gr
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import time

# Available models and their configurations
MODEL_CONFIGS = {
    "Whisper Large v3 (Standard)": {
        "model": "openai/whisper-large-v3",
        "torch_dtype": torch.float32,
        "use_bettertransformer": False,
        "use_flash_attention": False,
        "performance": "~31 min for 150min audio"
    },
    "Whisper Large v3 (Optimized)": {
        "model": "openai/whisper-large-v3",
        "torch_dtype": torch.float16,
        "use_bettertransformer": True,
        "use_flash_attention": False,
        "performance": "~5 min for 150min audio"
    },
    "Whisper Large v3 (Flash Attention)": {
        "model": "openai/whisper-large-v3",
        "torch_dtype": torch.float16,
        "use_bettertransformer": False,
        "use_flash_attention": True,
        "performance": "~1.6 min for 150min audio"
    },
    "Distil Large v2 (Optimized)": {
        "model": "distil-whisper/distil-large-v2",
        "torch_dtype": torch.float16,
        "use_bettertransformer": True,
        "use_flash_attention": False,
        "performance": "~3.3 min for 150min audio"
    },
    "Distil Large v2 (Flash Attention)": {
        "model": "distil-whisper/distil-large-v2",
        "torch_dtype": torch.float16,
        "use_bettertransformer": False,
        "use_flash_attention": True,
        "performance": "~1.3 min for 150min audio"
    },
    "Whisper Large v2 (Faster-8bit)": {
        "model": "openai/whisper-large-v2",
        "torch_dtype": torch.float16,
        "use_bettertransformer": False,
        "use_flash_attention": False,
        "performance": "~8.2 min for 150min audio"
    }
}

# Initialize the pipeline with default model
def get_pipeline(config_name, progress=gr.Progress()):
    config = MODEL_CONFIGS[config_name]
    model_kwargs = {}
    
    progress(0, desc="Initializing pipeline...")
    
    if config["use_flash_attention"] and is_flash_attn_2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif config["use_bettertransformer"]:
        model_kwargs["use_bettertransformer"] = True
    
    progress(0.3, desc="Loading model...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=config["model"],
        torch_dtype=config["torch_dtype"],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        model_kwargs=model_kwargs,
    )
    progress(1.0, desc="Model loaded successfully!")
    return pipe

# Initialize with default model
pipe = get_pipeline("Whisper Large v3 (Flash Attention)")

def transcribe_audio(audio, config_name, progress=gr.Progress()):
    if audio is None:
        return "Please upload an audio file"
    
    # Get the pipeline for the selected configuration
    global pipe
    progress(0, desc="Loading model configuration...")
    pipe = get_pipeline(config_name, progress)
    
    progress(0.1, desc="Starting transcription...")
    
    # Use streaming for real-time output
    outputs = pipe(
        audio,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
        stream=True,
        generate_kwargs={"streamer": True}
    )
    
    full_text = ""
    for output in outputs:
        if output.get("text"):
            full_text += output["text"]
            yield full_text
        progress(0.1 + (output.get("progress", 0) * 0.9), desc="Transcribing...")
    
    progress(1.0, desc="Transcription complete!")

# Create the Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            value="Whisper Large v3 (Flash Attention)",
            label="Select Model & Optimization",
            info="Choose the model and optimization method. Performance metrics are based on 150 minutes of audio."
        )
    ],
    outputs=gr.Textbox(label="Transcription", lines=10),
    title="Insanely Fast Whisper Transcription",
    description="Upload an audio file and select a model configuration to transcribe it. Supports various audio formats including MP3, WAV, and more.",
    examples=[
        ["example1.mp3", "Whisper Large v3 (Flash Attention)"],
        ["example2.wav", "Distil Large v2 (Flash Attention)"]
    ],
    live=True  # Enable real-time updates
)

if __name__ == "__main__":
    iface.launch() 