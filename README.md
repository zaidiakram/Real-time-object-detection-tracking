# Real-Time Object Detection & Tracking with YOLO26s + SORT

A computer vision project for **object detection and multi-object tracking** using **YOLO26s** and **SORT**.

YOLO26s detects objects in each frame, while SORT associates detections across consecutive frames and assigns tracking IDs.

<p align="center">
  <img src="assets/system_pipeline.png"
       alt="YOLO26s and SORT system pipeline"
       width="900">
</p>

---

## Features

- Object detection with YOLO26s
- Multi-object tracking with SORT
- Tracking IDs for detected objects
- Image processing
- Video processing
- Live webcam processing
- Class-specific detection
- FPS display during video and webcam processing
- Active track count
- Processed output saving
- Separate detection-only pipeline
- Tested with CPU inference

---

## Project Structure

```text
Real-Time Object Detection & Tracking System/
│
├── assets/
│   └── system_pipeline.png
│
├── obj_det_and_trk.py      # Detection + tracking
├── ob_detect.py            # Detection only
├── sort.py                 # SORT tracker
├── utils/                  # Utilities used by ob_detect.py
│
├── yolo26s.pt
├── test.jpg
├── test.mp4
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Webcam

```bash
python obj_det_and_trk.py --source 0 --view-img
```

### Video

```bash
python obj_det_and_trk.py --source test.mp4 --view-img
```

### Image

```bash
python obj_det_and_trk.py --source test.jpg --view-img
```

Press **Q** to stop video or webcam processing.

### Selected Class

Example using class ID `0`:

```bash
python obj_det_and_trk.py --source test.mp4 --classes 0 --view-img
```

---

## Detection Only

### Image

```bash
python ob_detect.py --weights yolo26s.pt --source test.jpg --view-img
```

### Video

```bash
python ob_detect.py --weights yolo26s.pt --source test.mp4 --view-img
```

---

## Configuration

| Parameter | Value |
|---|---:|
| Detector | YOLO26s |
| Weights | `yolo26s.pt` |
| Image Size | 640 |
| Confidence Threshold | 0.60 |
| IoU Threshold | 0.45 |
| Device | CPU |
| Tracker | SORT |
| SORT `max_age` | 15 |
| SORT `min_hits` | 2 |
| SORT IoU | 0.20 |

YOLO26s inference in the tracking pipeline uses:

```text
end2end=False
```

---

## How It Works

YOLO26s produces detections containing:

```text
x1, y1, x2, y2, confidence, class_id
```

These detections are passed to SORT.

SORT uses a **Kalman Filter** for motion estimation and **IoU-based association** to match detections with existing tracks.

Tracked objects are displayed in the form:

```text
person | ID: 1
car | ID: 2
```

For video and webcam input, the runtime overlay displays:

```text
FPS: <current FPS> | Active Tracks: <count>
```

`Active Tracks` represents the number of tracks returned by SORT for the current frame.

---

## Output

Processed results are saved under:

```text
runs/detect/
```

Each run uses an incremented directory:

```text
runs/detect/exp/
runs/detect/exp2/
runs/detect/exp3/
...
```

For a video such as `test.mp4`, the tracking pipeline saves output with a filename such as:

```text
test_tracked.mp4
```

---

## Limitations

- Detection depends on the classes supported by the loaded pretrained model.
- Detection results can vary with lighting, object size, motion blur, camera angle, and occlusion.
- SORT uses motion and bounding-box overlap rather than appearance-based re-identification.
- Tracking IDs may change during difficult or extended occlusions.
- Processing speed depends on the input and hardware.

---

## License

This project is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See [`LICENSE`](LICENSE) for the complete license terms.

---

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [SORT: Simple Online and Realtime Tracking](https://github.com/abewley/sort)