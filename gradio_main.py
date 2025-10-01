import os
import pickle

import gradio as gr
import torch
from gradio import SimpleCSVLogger
from torchvision import transforms

from model import DecoderRNN, EncoderCNN
from nlp_utils import clean_sentence


# Defining a deterministic transform to pre-process testing images.
transform_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model config
encoder_file = "encoder-3.pkl"
decoder_file = "decoder-3.pkl"
models_dir = "./models"

# Hyperparameters must match those used during training
embed_size = 256
hidden_size = 512


def load_vocab(vocab_path: str = "./vocab.pkl"):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"vocab.pkl not found at {vocab_path}. Make sure it exists in the project root."
        )
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    # Expecting object with word2idx/idx2word attributes
    if not hasattr(vocab, "idx2word") or not hasattr(vocab, "word2idx"):
        raise ValueError("Loaded vocab object is missing required attributes.")
    return vocab


# Load vocabulary to infer vocab size and mapping
vocab = load_vocab()
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
decoder.eval()


def safe_load_model(module: torch.nn.Module, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Model file not found: {file_path}. Please download trained weights into {models_dir}."
        )
    state = torch.load(file_path, map_location=device)
    module.load_state_dict(state)


# Load the trained weights.
safe_load_model(encoder, os.path.join(models_dir, encoder_file))
safe_load_model(decoder, os.path.join(models_dir, decoder_file))

# Move models to device
encoder.to(device)
decoder.to(device)


def predict_caption(image):
    if image is None:
        return "Please select an image"

    image = transform_test(image).unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        features = encoder(image).unsqueeze(1)
        output = decoder.sample(features)

    sentence = clean_sentence(output, vocab.idx2word)
    return sentence


gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(type="pil", image_mode="RGB"),
    outputs=gr.Textbox(label="Predicted caption"),
    flagging_dir="./gradio_logs",
    flagging_callback=SimpleCSVLogger(),
).launch(share=True, server_port=7860)
