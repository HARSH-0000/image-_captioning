import os
import argparse
import time
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageFont, ImageDraw

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, add

# OpenCV for webcam/display
import cv2

# For tokenizer loading/saving
import pickle

# Detectron2
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    import torch
    DETECTRON2_AVAILABLE = True
except Exception as e:
    print("Warning: detectron2 import failed. Segmentation will be disabled. Error:", e)
    DETECTRON2_AVAILABLE = False

# ---------------------------
# CONFIG / PATHS (edit these)
# ---------------------------
CAPTION_MODEL_WEIGHTS = "caption_decoder_weights.h5"   # Put your trained decoder weights here
TOKENIZER_PATH = "tokenizer.pickle"                    # Put your tokenizer here (pickle)
VOCAB_SIZE = 10000  # replace with actual vocab_size used during training
MAX_CAPTION_LEN = 34  # replace with your sequence length used during training

DETECTRON2_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
DETECTRON2_WEIGHTS = None  # leave None to use model_zoo default weights (will auto-download if allowed)

# Controls (for streaming)
CAPTION_INTERVAL_SECONDS = 2.0  # generate caption every N seconds for performance
SEGMENTATION_CONFIDENCE_THRESHOLD = 0.5

# ---------------------------
# Build/load InceptionV3 feature extractor
# ---------------------------
def build_feature_extractor():
    # Load pre-trained InceptionV3 and remove top
    base_model = InceptionV3(weights='imagenet')
    # Create model that outputs the final avg_pool features (2048-dim)
    model = Model(base_model.input, base_model.get_layer("avg_pool").output)
    return model

feature_extractor = build_feature_extractor()

def extract_features_from_pil(img: Image.Image) -> np.ndarray:
    """Return 2048-d feature vector for a PIL Image"""
    img = img.resize((299, 299))
    x = np.array(img).astype('float32')
    if x.shape[-1] == 4:
        x = x[..., :3]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = feature_extractor.predict(x)
    feat = feat.reshape((1, 2048))
    return feat

def extract_features_from_path(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return extract_features_from_pil(img)

# ---------------------------
# Caption model architecture (same as training)
# ---------------------------
def build_caption_model(vocab_size: int, max_length: int):
    # image feature input
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(inputs1)
    fe2 = tf.keras.layers.RepeatVector(1)(fe1)  # expand dims to combine with seq branch (optional)

    # sequence input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)

    # combine
    decoder1 = add([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# We'll load weights if available
caption_model = None
tokenizer = None

def load_captioning(vocab_size: int, max_len: int):
    global caption_model, tokenizer
    caption_model = build_caption_model(vocab_size, max_len)
    if os.path.exists(CAPTION_MODEL_WEIGHTS):
        try:
            caption_model.load_weights(CAPTION_MODEL_WEIGHTS)
            print("Loaded caption model weights from", CAPTION_MODEL_WEIGHTS)
        except Exception as e:
            print("Failed to load captioning weights:", e)
    else:
        print("Caption weights file not found at", CAPTION_MODEL_WEIGHTS, "- using untrained model (predictions will not be meaningful).")

    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Loaded tokenizer from", TOKENIZER_PATH)
    else:
        print("Tokenizer not found at", TOKENIZER_PATH)
        tokenizer = None

# ---------------------------
# Caption generation (greedy)
# ---------------------------
def word_for_id(integer, tokenizer):
    for w, index in tokenizer.word_index.items():
        if index == integer:
            return w
    return None

def generate_caption_greedy(photo_feat: np.ndarray, tokenizer, max_length: int) -> str:
    """
    Greedy generation using the caption_model.
    Requires tokenizer mapping and trained weights to be meaningful.
    """
    if tokenizer is None or caption_model is None:
        return "caption_model_or_tokenizer_missing"

    # start token assumed to be 'startseq' and end token 'endseq' - adjust if different
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo_feat, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.replace('startseq', '').replace('endseq', '').strip()
    if final == '':
        return "could_not_generate_caption"
    return final

# ---------------------------
# Detectron2 setup
# ---------------------------
detectron2_predictor = None
def init_detectron2(config_name=DETECTRON2_CONFIG, weights_override=None):
    global detectron2_predictor
    if not DETECTRON2_AVAILABLE:
        print("Detectron2 not available; segmentation disabled.")
        return None
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    if weights_override:
        cfg.MODEL.WEIGHTS = weights_override
    else:
        # use model zoo's default trained weights (COCO)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SEGMENTATION_CONFIDENCE_THRESHOLD
    predictor = DefaultPredictor(cfg)
    detectron2_predictor = predictor
    print("Detectron2 initialized with config:", config_name)
    return predictor

def segment_image_opencv(image_bgr: np.ndarray):
    """
    Run Detectron2 predictor on an OpenCV BGR image.
    Returns the visualized image (BGR) and list of detected classes/masks
    """
    if not DETECTRON2_AVAILABLE or detectron2_predictor is None:
        return image_bgr, []

    im = image_bgr[:, :, ::-1]  # BGR->RGB
    outputs = detectron2_predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = out.get_image()[:, :, ::-1]
    instances = outputs.get("instances").to("cpu")
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []
    return result, classes

# ---------------------------
# Display helpers (PIL overlays)
# ---------------------------
def overlay_caption_on_image_pil(pil_img: Image.Image, caption: str) -> Image.Image:
    draw = ImageDraw.Draw(pil_img)
    # choose a font if available (fallback to default)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    text = caption
    # position at bottom-left
    margin = 10
    x = margin
    y = pil_img.height - 30 - margin
    # draw rectangle behind text
    text_w, text_h = draw.textsize(text, font=font)
    draw.rectangle([x-5, y-5, x+text_w+5, y+text_h+5], fill=(0,0,0,160))
    draw.text((x, y), text, fill=(255,255,255), font=font)
    return pil_img

# ---------------------------
# Integrate: process_image
# ---------------------------
def process_image(image_path: str, show: bool = True) -> Tuple[str, np.ndarray]:
    """
    Accept image path, run caption generation (on extracted features) and segmentation (detectron2).
    Returns (caption_text, result_bgr_image)
    """
    # Read image
    pil_img = Image.open(image_path).convert("RGB")
    photo_feat = extract_features_from_pil(pil_img)  # shape (1,2048)
    caption = generate_caption_greedy(photo_feat, tokenizer, MAX_CAPTION_LEN)

    # Convert to OpenCV BGR for segmentation
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    seg_result_bgr, classes = segment_image_opencv(img_cv)

    # Overlay caption
    seg_pil = Image.fromarray(cv2.cvtColor(seg_result_bgr, cv2.COLOR_BGR2RGB))
    overlayed = overlay_caption_on_image_pil(seg_pil, caption)

    result_bgr = cv2.cvtColor(np.array(overlayed), cv2.COLOR_RGB2BGR)
    if show:
        cv2.imshow("Caption + Segmentation", result_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return caption, result_bgr

# ---------------------------
# Webcam / Video streaming
# ---------------------------
def run_webcam(device=0):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("Could not open webcam/device", device)
        return

    last_caption = ""
    last_caption_time = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break

            now = time.time()
            # Resize frame for faster processing (optional)
            proc_frame = frame.copy()

            # segmentation (always)
            seg_frame_bgr, classes = segment_image_opencv(proc_frame)
            display_frame = seg_frame_bgr

            # caption periodically to save compute
            if now - last_caption_time > CAPTION_INTERVAL_SECONDS:
                # extract features from frame -> PIL
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                feat = extract_features_from_pil(pil_frame)
                last_caption = generate_caption_greedy(feat, tokenizer, MAX_CAPTION_LEN)
                last_caption_time = now

            # overlay caption text on display_frame
            display_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            display_pil = overlay_caption_on_image_pil(display_pil, last_caption)
            display_frame = cv2.cvtColor(np.array(display_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow("Webcam - Caption + Segmentation", display_frame)

            # exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------
# CLI entrypoint
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="image", choices=["image", "webcam"], help="mode: image or webcam")
    parser.add_argument("--image_path", type=str, default=None, help="image file path (for image mode)")
    parser.add_argument("--device", type=int, default=0, help="webcam device index (for webcam mode)")
    args = parser.parse_args()

    # Load captioning components
    load_captioning(VOCAB_SIZE, MAX_CAPTION_LEN)

    # init detectron2
    init_detectron2(DETECTRON2_CONFIG, DETECTRON2_WEIGHTS)

    if args.mode == "image":
        if not args.image_path:
            print("Please provide --image_path for image mode.")
            return
        caption, out_img = process_image(args.image_path, show=True)
        print("Caption:", caption)
    elif args.mode == "webcam":
        print("Starting webcam. Press 'q' to quit.")
        run_webcam(device=args.device)

if __name__ == "__main__":
    main()
