# PixelHertz
# PixelHertz

Multimodal deepfake detection for images, video, and audio.

PixelHertz checks each media type with its own model, and fuses visual and audio signals for video. It also explains its predictions with Grad-CAM heatmaps.

## Features

- **Multimodal detection:** separate pipelines for images, video, and audio
- **Image detection:** EfficientNet-B0 classifier (~91% accuracy)
- **Video detection:** EfficientNet-B0 frame analysis fused with MFCC audio features
- **Audio detection:** MLP trained on the Fake-or-Real (FoR) dataset
- **Grad-CAM explainability:** heatmaps show which regions drove the decision
- **Temporal inconsistency score:** flags frame-to-frame irregularities in video
- **Low-resource friendly:** disk-based feature caching keeps memory usage low
- **Streamlit app:** upload a file and get a real/fake verdict with confidence

## Tech Stack

Python, PyTorch, EfficientNet-B0, librosa (MFCC), OpenCV, Streamlit

## Screenshots

| Home | Image Result | Video Result |
|------|--------------|--------------|
| ![Home](screenshots/home.png) | ![Image result](screenshots/image_result.png) | ![Video result](screenshots/video_result.png) |

## Proposed Methodology

1. **Input handling:** the app detects whether the upload is an image, video, or audio file.
2. **Preprocessing:**
   - Images: face crop, resize, normalize
   - Video: sample frames, crop faces, extract the audio track
   - Audio: resample and extract MFCC features
3. **Feature extraction:** EfficientNet-B0 for visual features, MFCCs for audio features. Features are cached to disk to reduce memory use.
4. **Classification:**
   - Images: EfficientNet-B0 classifier
   - Audio: MLP classifier
   - Video: frame-level visual predictions fused with audio features
5. **Temporal analysis:** for video, a temporal inconsistency score captures changes across frames.
6. **Explainability:** Grad-CAM highlights the regions that influenced the prediction.
7. **Output:** real/fake verdict with confidence, shown in the Streamlit app.

## Setup (VS Code)

1. Clone the repo and open the folder in VS Code.

   ```bash
   git clone https://github.com/<your-username>/PixelHertz.git
   cd PixelHertz
   ```

2. Install dependencies.

   ```bash
   pip install torch torchvision timm numpy opencv-python librosa scikit-learn pillow streamlit
   ```

## Usage

Run the app:

```bash
streamlit run app.py
```

Upload an image, video, or audio file to get a real/fake prediction.

## Results

| Modality | Model | Result |
|----------|-------|--------|
| Image | EfficientNet-B0 | ~91% accuracy |
| Video | EfficientNet-B0 + MFCC fusion | — |
| Audio | MLP (FoR dataset) | — |

## Datasets

- Face image/video datasets for the visual models
- [Fake-or-Real (FoR)](https://bil.eecs.yorku.ca/datasets/) for audio

Datasets are not included in this repo.

## License

Free to use for academic purposes only.
