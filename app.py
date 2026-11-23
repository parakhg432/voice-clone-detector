import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# --- Page Config ---
st.set_page_config(page_title="Voice Clone Detector", page_icon="🎙️", layout="wide")

SAMPLE_RATE = 16000
DURATION = 2
N_MELS = 128
TARGET_LEN = SAMPLE_RATE * DURATION
MODEL_PATH = "voice_cnn_best.pth"


# --- Model Definition (Must match Colab training script) ---
class VoiceSpoofCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = VoiceSpoofCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def chunk_to_melspec(y_chunk):
    if len(y_chunk) < TARGET_LEN:
        y_chunk = np.pad(y_chunk, (0, TARGET_LEN - len(y_chunk)))
    else:
        y_chunk = y_chunk[:TARGET_LEN]

    mel = librosa.feature.melspectrogram(y=y_chunk, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0), mel_db


def analyse_audio(audio_bytes, model):
    # Load audio strictly from RAM (never saved to disk)
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)

    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < int(0.5 * SAMPLE_RATE):
        y_trimmed = y

    # Sliding-window inference (2-second windows with 1-second stride)
    step = SAMPLE_RATE  # 1-second hop
    probs = []
    first_mel_db = None

    if len(y_trimmed) <= TARGET_LEN:
        chunks = [y_trimmed]
    else:
        chunks = [
            y_trimmed[start : start + TARGET_LEN]
            for start in range(0, len(y_trimmed) - TARGET_LEN + 1, step)
        ]

    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            spec_tensor, mel_db = chunk_to_melspec(chunk)
            if i == 0:
                first_mel_db = mel_db
            prob = torch.sigmoid(model(spec_tensor)).item() * 100
            probs.append(prob)

    avg_spoof_prob = float(np.mean(probs))
    return avg_spoof_prob, probs, y_trimmed, first_mel_db


# --- UI Layout ---
st.title("🎙️ AI Voice Cloning & Impersonation Detector")
st.caption("Real-Time Audio Anti-Spoofing (In-Memory Privacy-First Inference)")

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Could not find `{MODEL_PATH}` in this folder. Place your downloaded model file next to `app.py` and refresh.")
    st.stop()

tab1, tab2 = st.tabs(["🎤 Live Microphone Check", "📁 Upload Call Recording"])

audio_bytes = None

with tab1:
    st.write("Record 3–5 seconds of speech to test live voice authenticity:")
    mic_audio = st.audio_input("Click the mic icon to record", sample_rate=SAMPLE_RATE)
    if mic_audio is not None:
        audio_bytes = mic_audio.read()

with tab2:
    uploaded_file = st.file_uploader("Upload a `.wav` or `.mp3` audio clip", type=["wav", "mp3", "ogg"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        audio_bytes = uploaded_file.read()

# --- Prediction & Dashboard ---
if audio_bytes:
    with st.spinner("Analysing speech frequency patterns..."):
        spoof_prob, window_probs, y_audio, mel_db = analyse_audio(audio_bytes, model)

    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Detection Result")
        st.metric(label="AI Clone Probability", value=f"{spoof_prob:.1f}%")
        st.progress(min(int(spoof_prob), 100))

        if spoof_prob <= 40:
            st.success("🟢 **GREEN — Safe (Likely Genuine Human Voice)**")
            st.write("No significant synthetic voice artefacts detected.")
        elif spoof_prob <= 70:
            st.warning("🟡 **AMBER — Uncertain (Verify Caller)**")
            st.write("Ask a personal verification question or call back on a known number before sharing sensitive info.")
        else:
            st.error("🔴 **RED — High Risk (Likely AI-Cloned Voice)**")
            st.write("⚠️ **Prevention Alert:** Do NOT authorise UPI transfers, OTPs, or urgent financial requests.")

        st.caption(f"Analysed across {len(window_probs)} sliding window(s) • Audio discarded from RAM after inference.")

    with col2:
        st.subheader("Audio Forensics Visualisation")
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5))

        # Waveform
        librosa.display.waveshow(y_audio, sr=SAMPLE_RATE, ax=axes[0], color="#1f77b4")
        axes[0].set_title("Speech Waveform (Silence Trimmed)")
        axes[0].set_xlabel("")

        # Mel-Spectrogram
        img = librosa.display.specshow(mel_db, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=axes[1])
        axes[1].set_title("Mel-Spectrogram (CNN Input Features)")
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

        plt.tight_layout()
        st.pyplot(fig)