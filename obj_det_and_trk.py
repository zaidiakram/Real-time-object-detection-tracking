import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort, KalmanBoxTracker


ROOT = Path(__file__).resolve().parent

IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_BOX_COLOR = (255, 191, 0)
BLUR_RATIO = 40


def increment_path(path: Path, exist_ok: bool = False) -> Path:
    """
    Creates:
        runs/detect/exp
        runs/detect/exp2
        runs/detect/exp3
        ...
    """
    if exist_ok or not path.exists():
        return path

    index = 2

    while True:
        new_path = Path(f"{path}{index}")

        if not new_path.exists():
            return new_path

        index += 1


def compute_color_for_label(track_id: int):
    """Generate a stable color for each tracking ID."""
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    return tuple(
        int((p * (track_id ** 2 - track_id + 1)) % 255)
        for p in palette
    )


def get_class_name(names, class_id: int) -> str:
    """Return class name safely for list or dict model.names."""
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))

    if 0 <= class_id < len(names):
        return names[class_id]

    return str(class_id)


def draw_boxes(
    frame,
    tracked_detections,
    names,
    color_box=False,
    line_thickness=2,
):
    """Draw tracked bounding boxes with class name and tracking ID."""

    for track in tracked_detections:

        # SORT output:
        # 0:4 -> bounding box
        # 4   -> class id
        # 8   -> tracking id
        x1, y1, x2, y2 = map(int, track[:4])

        class_id = int(track[4])
        track_id = int(track[8])

        class_name = get_class_name(names, class_id)

        label = f"{class_name} | ID: {track_id}"

        if color_box:
            color = compute_color_for_label(track_id)
        else:
            color = DEFAULT_BOX_COLOR

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            line_thickness,
        )

        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1,
        )

        label_y1 = max(0, y1 - text_height - 10)

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + text_width + 8, y1),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def blur_detections(frame, detections):
    """Blur objects detected by YOLO."""

    frame_height, frame_width = frame.shape[:2]

    for detection in detections:

        x1, y1, x2, y2 = map(int, detection[:4])

        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(0, min(x2, frame_width))

        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        frame[y1:y2, x1:x2] = cv2.blur(
            crop,
            (BLUR_RATIO, BLUR_RATIO),
        )


def run_yolo(
    model,
    frame,
    imgsz,
    conf_thres,
    iou_thres,
    max_det,
    classes,
    agnostic_nms,
    augment,
    device,
):
    """Run YOLO26 detection on one frame."""

    result = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        classes=classes,
        agnostic_nms=agnostic_nms,
        augment=augment,

        # YOLO26 one-to-many head + NMS.
        # Kept False because detection accuracy is the priority.
        end2end=False,

        device=device,
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = result.boxes

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy()

    detections = np.column_stack(
        (
            xyxy,
            confidence,
            class_ids,
        )
    )

    return detections.astype(np.float32)


def process_frame(
    frame,
    model,
    tracker,
    names,
    imgsz,
    conf_thres,
    iou_thres,
    max_det,
    classes,
    agnostic_nms,
    augment,
    device,
    blur_obj,
    color_box,
    line_thickness,
):
    """Detect objects, update SORT and draw tracking results."""

    detections = run_yolo(
        model=model,
        frame=frame,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        classes=classes,
        agnostic_nms=agnostic_nms,
        augment=augment,
        device=device,
    )

    if blur_obj and len(detections) > 0:
        blur_detections(frame, detections)

    # SORT must be updated on every frame.
    # Empty detections are also passed when YOLO finds nothing.
    tracked_detections = tracker.update(detections)

    if len(tracked_detections) > 0:
        draw_boxes(
            frame=frame,
            tracked_detections=tracked_detections,
            names=names,
            color_box=color_box,
            line_thickness=line_thickness,
        )

    return frame, len(tracked_detections)


def process_image(
    source,
    model,
    tracker,
    names,
    save_dir,
    view_img,
    nosave,
    **kwargs,
):
    """Process a single image."""

    frame = cv2.imread(source)

    if frame is None:
        raise RuntimeError(f"Could not read image: {source}")

    frame,_ = process_frame(
        frame=frame,
        model=model,
        tracker=tracker,
        names=names,
        **kwargs,
    )

    if not nosave:
        output_path = save_dir / Path(source).name

        cv2.imwrite(
            str(output_path),
            frame,
        )

        print(f"Output saved: {output_path}")

    if view_img:
        cv2.imshow(
            "YOLO26 Object Detection & Tracking",
            frame,
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    source,
    model,
    tracker,
    names,
    save_dir,
    view_img,
    nosave,
    **kwargs,
):
    """Process webcam, video file or network stream."""

    if source.isdigit():
        video_source = int(source)
        source_name = "webcam"
    else:
        video_source = source

        if source.startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        ):
            source_name = "stream"
        else:
            source_name = Path(source).stem

    capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    writer = None
    output_path = None
    previous_time = time.perf_counter()
    display_fps = 0.0

    if not nosave:

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 1 or fps > 120:
            fps = 30

        output_path = (
            save_dir /
            f"{source_name}_tracked.mp4"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()

            raise RuntimeError(
                "Could not create output video."
            )

    print("Detection and tracking started.")

    if view_img:
        print("Press Q to stop.")

    try:

        while True:

            success, frame = capture.read()

            if not success:
                break

            frame, active_tracks  = process_frame(
                frame=frame,
                model=model,
                tracker=tracker,
                names=names,
                **kwargs,
            )
            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed

                if display_fps == 0:
                    display_fps = current_fps
                else:
                    display_fps = 0.9 * display_fps + 0.1 * current_fps

            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f} | Active Tracks: {active_tracks}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            
            if writer is not None:
                writer.write(frame)

            if view_img:

                cv2.imshow(
                    "YOLO26 Object Detection & Tracking",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Stopped by user.")
                    break

    finally:

        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Output saved: {output_path}")


def detect(
    weights,
    source,
    imgsz,
    conf_thres,
    iou_thres,
    max_det,
    device,
    view_img,
    nosave,
    classes,
    agnostic_nms,
    augment,
    project,
    name,
    exist_ok,
    line_thickness,
    blur_obj,
    color_box,
    sort_max_age,
    sort_min_hits,
    sort_iou_thres,
):
    """Main detection and tracking pipeline."""

    weights_path = Path(weights)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model not found: {weights}"
        )

    # ---------------- YOLO26 ----------------

    model = YOLO(str(weights_path))

    names = model.names

    # ---------------- SORT ----------------

    # Reset tracker IDs for every fresh run.
    KalmanBoxTracker.count = 0

    tracker = Sort(
        max_age=sort_max_age,
        min_hits=sort_min_hits,
        iou_threshold=sort_iou_thres,
    )

    # ---------------- Output ----------------

    save_dir = increment_path(
        Path(project) / name,
        exist_ok=exist_ok,
    )

    if not nosave:
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("\n==============================")
    print(
        f"DETECTOR : {weights_path.stem}"
    )
    print(
        f"WEIGHTS  : {weights_path.name}"
    )
    print("TRACKER  : SORT")
    print(
        f"DEVICE   : {device.upper()}"
    )
    print(
        f"IMAGE SIZE : {imgsz}"
    )
    print(
        f"CONFIDENCE : {conf_thres}"
    )
    print(
        f"IOU        : {iou_thres}"
    )
    print("==============================\n")

    common_args = dict(
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        classes=classes,
        agnostic_nms=agnostic_nms,
        augment=augment,
        device=device,
        blur_obj=blur_obj,
        color_box=color_box,
        line_thickness=line_thickness,
    )

    source_path = Path(source)

    if (
        not source.isdigit()
        and source_path.suffix.lower() in IMAGE_FORMATS
    ):

        if not source_path.exists():
            raise FileNotFoundError(
                f"Image not found: {source}"
            )

        process_image(
            source=source,
            model=model,
            tracker=tracker,
            names=names,
            save_dir=save_dir,
            view_img=view_img,
            nosave=nosave,
            **common_args,
        )

    else:

        if (
            not source.isdigit()
            and not source.startswith(
                (
                    "rtsp://",
                    "rtmp://",
                    "http://",
                    "https://",
                )
            )
            and not source_path.exists()
        ):
            raise FileNotFoundError(
                f"Video not found: {source}"
            )

        process_video(
            source=source,
            model=model,
            tracker=tracker,
            names=names,
            save_dir=save_dir,
            view_img=view_img,
            nosave=nosave,
            **common_args,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Real-Time Object Detection "
            "and Tracking using YOLO26 + SORT"
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "yolo26s.pt"),
        help="YOLO26 model path",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index, image, video or stream URL",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.60,
        help="Detection confidence threshold",
    )

    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per frame",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device, e.g. cpu or 0",
    )

    parser.add_argument(
        "--view-img",
        action="store_true",
        help="Display detection window",
    )

    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Do not save output",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Filter by COCO class IDs",
    )

    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use augmented inference",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=str(ROOT / "runs" / "detect"),
        help="Output directory",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Run name",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Reuse existing run directory",
    )

    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding box thickness",
    )

    parser.add_argument(
        "--blur-obj",
        action="store_true",
        help="Blur detected objects",
    )

    parser.add_argument(
        "--color-box",
        action="store_true",
        help="Use different box colors for tracking IDs",
    )

    parser.add_argument(
        "--sort-max-age",
        type=int,
        default=15,
        help="Frames to keep an unmatched track alive",
    )

    parser.add_argument(
        "--sort-min-hits",
        type=int,
        default=2,
        help="Minimum detections required for a track",
    )

    parser.add_argument(
        "--sort-iou-thres",
        type=float,
        default=0.2,
        help="SORT IoU matching threshold",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    detect(
        weights=args.weights,
        source=args.source,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        device=args.device,
        view_img=args.view_img,
        nosave=args.nosave,
        classes=args.classes,
        agnostic_nms=args.agnostic_nms,
        augment=args.augment,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        line_thickness=args.line_thickness,
        blur_obj=args.blur_obj,
        color_box=args.color_box,
        sort_max_age=args.sort_max_age,
        sort_min_hits=args.sort_min_hits,
        sort_iou_thres=args.sort_iou_thres,
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as error:
        print(f"\nError: {error}")