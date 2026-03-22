# 🎬 Clip-Match

Real-time video similarity search using **frame-level NCC scoring** and **OpenCV** to detect whether a query clip exists within a reference dataset.

![Python](https://img.shields.io/badge/Python-3.8--3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **Frame Extraction** – Samples frames from both query and dataset videos at a configurable FPS
- **Block Feature Vectors** – Divides each frame into a 4×4 grid and computes mean intensities
- **NCC Scoring** – Measures similarity between query and dataset frames using Normalized Cross-Correlation
- **Match Verdict** – Returns the best-matching video with a similarity score and a clear ✅ / ❌ result
- **Configurable** – All key parameters defined at the top of `match_pipeline.py`

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/AtharvaShaha/Clip-Match.git
cd Clip-Match

# 2. Install dependencies
pip install opencv-python numpy

# 3. Add your videos
#    - Reference videos → dataset/
#    - Query video      → query/query_01.mp4

# 4. Run
python match_pipeline.py
```

---

## 📁 Project Structure

```
Clip-Match/
├── match_pipeline.py        # Main pipeline — run this
├── dataset/                 # Reference videos to search against
├── query/                   # Query video(s) to identify
├── features/                # Reserved: precomputed dataset features
├── query_features/          # Reserved: precomputed query features
├── temp_frames/             # Auto-generated: dataset frames (do not edit)
└── temp_query_frames/       # Auto-generated: query frames (do not edit)
```

---

## ⚙️ Configuration

All parameters are set at the top of `match_pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `DATASET_PATH` | `"Dataset"` | Folder containing reference videos |
| `QUERY_VIDEO` | `"query/query_01.mp4"` | Path to the query video |
| `FRAME_PATH` | `"temp_frames"` | Output folder for dataset frames |
| `QUERY_FRAME_PATH` | `"temp_query_frames"` | Output folder for query frames |
| `FPS_TARGET` | `32` | Frames per second to sample |
| `MATCH_THRESHOLD` | `0.80` | Minimum NCC score to declare a match |

---

## 🔍 How It Works

```
Query Video ──► Frame Extraction ──► Block Feature Vectors ──┐
                                                              ▼
Dataset Videos ► Frame Extraction ──► Block Feature Vectors ──► NCC Scoring ──► Match Verdict
```

1. **Extract** – Both videos are sampled at `FPS_TARGET`. Each frame is resized to `256×256` and converted to grayscale.
2. **Describe** – Each frame is split into a `4×4` grid; mean pixel intensity per block gives a 16-dimensional feature vector.
3. **Compare** – Query and dataset frame vectors are compared using NCC. Scores are averaged across all frames.
4. **Decide** – The dataset video with the highest average NCC score wins. If it exceeds `MATCH_THRESHOLD`, the clip is confirmed present.

---

## 📊 Output

```
video_01 | Frame 0 NCC = 0.943
video_01 | Frame 1 NCC = 0.921
...
Average NCC with video_01 = 0.934

=================================
BEST MATCH: video_01
SIMILARITY: 0.934
RESULT: VIDEO EXISTS IN DATASET ✅
```

If no video clears the threshold:

```
RESULT: VIDEO NOT FOUND ❌
```

---

## ⚠️ Limitations

- Frame comparison is **index-aligned** — the query must start at the same point as the reference video. Mid-clip queries will not match correctly.
- Dataset frames are **re-extracted on every run** — no caching yet.
- Feature descriptor is **not robust** to rotation, scaling, or encoding differences.

---

**Made with ❤️ by Atharva** | © 2026 All Rights Reserved
