# MAIA MUSE — AI Music Detector Demo

Drag-and-drop Streamlit app that detects AI-generated music using spectral fakeprint analysis.

## Setup

### 1. Copy model files

```bash
cp ../ai-music-detector/src/models/ai_music_detector.onnx models/
cp ../ai-music-detector/src/python/config.yaml .
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Folder structure

```
muse_demo/
├── app.py              # Streamlit UI
├── detector.py         # Fakeprint inference wrapper
├── config.yaml         # Audio + model config  (copy from ai-music-detector)
├── requirements.txt
├── models/
│   └── ai_music_detector.onnx   (copy from ai-music-detector)
├── assets/             # Drop CEO logos / slides here
└── .streamlit/
    └── config.toml     # Dark theme
```

## Supported formats

MP3, WAV, FLAC, OGG, M4A — up to 100 MB
