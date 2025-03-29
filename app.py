import gradio as gr
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

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
def get_pipeline(config_name):
    config = MODEL_CONFIGS[config_name]
    model_kwargs = {}
    
    if config["use_flash_attention"] and is_flash_attn_2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif config["use_bettertransformer"]:
        model_kwargs["use_bettertransformer"] = True
    
    return pipeline(
        "automatic-speech-recognition",
        model=config["model"],
        torch_dtype=config["torch_dtype"],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        model_kwargs=model_kwargs,
    )

# Initialize with default model
pipe = get_pipeline("Whisper Large v3 (Flash Attention)")

def transcribe_audio(audio, config_name):
    if audio is None:
        return "Please upload an audio file"
    
    # Get the pipeline for the selected configuration
    global pipe
    pipe = get_pipeline(config_name)
    
    outputs = pipe(
        audio,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )
    
    return outputs["text"]

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
    outputs=gr.Textbox(label="Transcription"),
    title="Insanely Fast Whisper Transcription",
    description="Upload an audio file and select a model configuration to transcribe it. Supports various audio formats including MP3, WAV, and more.",
    examples=[
        ["example1.mp3", "Whisper Large v3 (Flash Attention)"],
        ["example2.wav", "Distil Large v2 (Flash Attention)"]
    ]
)

if __name__ == "__main__":
    iface.launch() 