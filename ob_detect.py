# YOLO26s Object Detection

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def increment_path(path: Path, exist_ok: bool = False) -> Path:
    """Return an incremented path such as exp, exp2, exp3, ..."""
    path = Path(path)

    if exist_ok or not path.exists():
        return path

    for index in range(2, 10000):
        candidate = Path(f"{path}{index}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create a new output directory for: {path}")


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
):
    """Run YOLO26s on one OpenCV frame."""
    return model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        classes=classes,
        agnostic_nms=agnostic_nms,
        augment=augment,
        end2end=False,
        device="cpu",
        verbose=False,
    )[0]


def detection_summary(result, names):
    """Create a short class-count summary for one result."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "no detections"

    class_ids = boxes.cls.detach().cpu().int().tolist()
    counts = Counter(class_ids)

    parts = []
    for class_id in sorted(counts):
        count = counts[class_id]
        name = names[class_id]
        suffix = "s" if count > 1 else ""
        parts.append(f"{count} {name}{suffix}")

    return ", ".join(parts)


def annotate_result(result, line_thickness, hide_labels, hide_conf):
    """Draw YOLO detections on the original frame."""
    return result.plot(
        line_width=line_thickness,
        labels=not hide_labels,
        conf=not hide_conf,
    )


def process_image(
    model,
    source_path,
    save_dir,
    save_img,
    view_img,
    imgsz,
    conf_thres,
    iou_thres,
    max_det,
    classes,
    agnostic_nms,
    augment,
    line_thickness,
    hide_labels,
    hide_conf,
):
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {source_path}")

    result = run_yolo(
        model,
        frame,
        imgsz,
        conf_thres,
        iou_thres,
        max_det,
        classes,
        agnostic_nms,
        augment,
    )

    annotated = annotate_result(
        result,
        line_thickness,
        hide_labels,
        hide_conf,
    )

    print(f"{source_path.name}: {detection_summary(result, model.names)}")

    if save_img:
        output_path = save_dir / source_path.name
        cv2.imwrite(str(output_path), annotated)
        print(f"Output saved: {output_path.resolve()}")

    if view_img:
        cv2.imshow("YOLO26s Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    model,
    source,
    source_name,
    save_dir,
    save_img,
    view_img,
    imgsz,
    conf_thres,
    iou_thres,
    max_det,
    classes,
    agnostic_nms,
    augment,
    line_thickness,
    hide_labels,
    hide_conf,
    vid_stride,
):
    capture_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(capture_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_fps <= 0:
        source_fps = 30.0

    output_fps = max(source_fps / max(vid_stride, 1), 1.0)
    writer = None
    output_path = None

    if save_img:
        output_name = (
            "webcam_detected.mp4"
            if str(source).isdigit()
            else f"{Path(source_name).stem}_detected.mp4"
        )
        output_path = save_dir / output_name
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )

    print("Detection started.")
    if view_img:
        print("Press Q to stop.")

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if vid_stride > 1 and (frame_index - 1) % vid_stride != 0:
                continue

            result = run_yolo(
                model,
                frame,
                imgsz,
                conf_thres,
                iou_thres,
                max_det,
                classes,
                agnostic_nms,
                augment,
            )

            annotated = annotate_result(
                result,
                line_thickness,
                hide_labels,
                hide_conf,
            )

            if writer is not None:
                writer.write(annotated)

            if view_img:
                cv2.imshow("YOLO26s Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Output saved: {output_path.resolve()}")


def run(
    weights="yolo26s.pt",
    source="0",
    imgsz=640,
    conf_thres=0.60,
    iou_thres=0.45,
    max_det=1000,
    classes=None,
    agnostic_nms=False,
    augment=False,
    view_img=False,
    nosave=False,
    project="runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    vid_stride=1,
):
    source = str(source)
    weights_path = Path(weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    if not nosave:
        save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    print(f"DETECTOR   : {weights_path.stem}")
    print("DEVICE     : CPU")
    print(f"IMAGE SIZE : {imgsz}")
    print(f"CONFIDENCE : {conf_thres}")
    print(f"IOU        : {iou_thres}")

    source_path = Path(source)
    suffix = source_path.suffix.lower()

    if source_path.is_file() and suffix in IMAGE_EXTENSIONS:
        process_image(
            model,
            source_path,
            save_dir,
            not nosave,
            view_img,
            imgsz,
            conf_thres,
            iou_thres,
            max_det,
            classes,
            agnostic_nms,
            augment,
            line_thickness,
            hide_labels,
            hide_conf,
        )
        return

    if (
        source.isdigit()
        or (source_path.is_file() and suffix in VIDEO_EXTENSIONS)
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    ):
        process_video(
            model,
            source,
            source_path.name if source_path.name else "stream",
            save_dir,
            not nosave,
            view_img,
            imgsz,
            conf_thres,
            iou_thres,
            max_det,
            classes,
            agnostic_nms,
            augment,
            line_thickness,
            hide_labels,
            hide_conf,
            max(1, vid_stride),
        )
        return

    raise ValueError(
        "Unsupported source. Use an image file, video file, webcam index such as 0, "
        "or a supported video stream URL."
    )


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO26s object detection")

    parser.add_argument("--weights", type=str, default="yolo26s.pt", help="model weights path")
    parser.add_argument("--source", type=str, default="0", help="image, video, webcam index, or stream URL")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.60, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per frame")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class IDs")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--nosave", action="store_true", help="do not save output")
    parser.add_argument("--project", type=str, default="runs/detect", help="output directory")
    parser.add_argument("--name", type=str, default="exp", help="run directory name")
    parser.add_argument("--exist-ok", action="store_true", help="reuse existing run directory")
    parser.add_argument("--line-thickness", type=int, default=3, help="bounding-box thickness")
    parser.add_argument("--hide-labels", action="store_true", help="hide class labels")
    parser.add_argument("--hide-conf", action="store_true", help="hide confidence values")
    parser.add_argument("--vid-stride", type=int, default=1, help="process every Nth video frame")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_opt()
    run(**vars(args))
