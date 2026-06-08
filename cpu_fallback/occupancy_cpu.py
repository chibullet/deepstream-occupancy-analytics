#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    center: tuple[int, int]
    prev_center: tuple[int, int]
    missed: int = 0
    count_cooldown: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU fallback for occupancy analytics using OpenCV HOG person detector"
    )
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="cpu_output.mp4", help="Output video path")
    parser.add_argument("--report", default="cpu_report.json", help="JSON report path")
    parser.add_argument(
        "--line-angle-deg",
        type=float,
        default=0.0,
        help="Counting line angle in degrees (0=horizontal, 90=vertical)",
    )
    parser.add_argument(
        "--line-center-x-ratio",
        type=float,
        default=0.5,
        help="Line center X as ratio of frame width [0..1]",
    )
    parser.add_argument(
        "--line-center-y-ratio",
        type=float,
        default=0.5,
        help="Line center Y as ratio of frame height [0..1]",
    )
    parser.add_argument(
        "--flow-direction",
        choices=["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"],
        default="left_to_right",
        help="Primary IN direction. Opposite direction is counted as OUT",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=90.0,
        help="Max centroid distance for track association",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=12,
        help="Frames tolerated without matching detection",
    )
    return parser.parse_args()


def center_of(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def detect_people(frame, hog):
    rects, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.03,
    )

    # Remove nested boxes to reduce duplicates.
    keep = []
    for i, r in enumerate(rects):
        x1, y1, w1, h1 = r
        area1 = w1 * h1
        nested = False
        for j, q in enumerate(rects):
            if i == j:
                continue
            x2, y2, w2, h2 = q
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                area2 = w2 * h2
                if area2 >= area1:
                    nested = True
                    break
        if not nested:
            keep.append(r)
    return keep


def associate(tracks, detections, next_id, max_distance):
    detection_centers = [center_of(r) for r in detections]
    unmatched_dets = set(range(len(detection_centers)))

    for t in tracks:
        t.prev_center = t.center

    for t in tracks:
        best_idx = -1
        best_dist = float("inf")
        for idx in list(unmatched_dets):
            cx, cy = detection_centers[idx]
            d = math.dist((cx, cy), t.center)
            if d < best_dist and d <= max_distance:
                best_dist = d
                best_idx = idx
        if best_idx >= 0:
            t.center = detection_centers[best_idx]
            t.missed = 0
            unmatched_dets.remove(best_idx)
        else:
            t.missed += 1

    for idx in unmatched_dets:
        tracks.append(
            Track(
                track_id=next_id,
                center=detection_centers[idx],
                prev_center=detection_centers[idx],
                missed=0,
            )
        )
        next_id += 1

    return tracks, next_id


def flow_vector(flow_direction):
    if flow_direction == "left_to_right":
        return np.array([1.0, 0.0], dtype=np.float32)
    if flow_direction == "right_to_left":
        return np.array([-1.0, 0.0], dtype=np.float32)
    if flow_direction == "top_to_bottom":
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([0.0, -1.0], dtype=np.float32)


def line_geometry(width, height, line_angle_deg, line_center_x_ratio, line_center_y_ratio):
    cx = float(np.clip(line_center_x_ratio, 0.0, 1.0) * width)
    cy = float(np.clip(line_center_y_ratio, 0.0, 1.0) * height)
    theta = math.radians(line_angle_deg)
    line_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    # Left-hand normal used for signed-distance side test.
    line_normal = np.array([-line_dir[1], line_dir[0]], dtype=np.float32)
    return np.array([cx, cy], dtype=np.float32), line_dir, line_normal


def signed_side(point, line_center, line_normal):
    p = np.array([float(point[0]), float(point[1])], dtype=np.float32)
    return float(np.dot(p - line_center, line_normal))


def draw_infinite_line(frame, line_center, line_dir, color=(0, 255, 255), thickness=2):
    h, w = frame.shape[:2]
    length = int(math.hypot(w, h))
    p1 = line_center - line_dir * length
    p2 = line_center + line_dir * length
    pt1 = (int(p1[0]), int(p1[1]))
    pt2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, pt1, pt2, color, thickness)


def run_occupancy(
    input_path,
    output_path,
    report_path,
    line_angle_deg,
    line_center_x_ratio,
    line_center_y_ratio,
    flow_direction,
    max_distance,
    max_missed,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    line_center, line_dir, line_normal = line_geometry(
        width, height, line_angle_deg, line_center_x_ratio, line_center_y_ratio
    )
    desired_flow = flow_vector(flow_direction)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracks = []
    next_id = 1
    total_in = 0
    total_out = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detect_people(frame, hog)
        tracks, next_id = associate(tracks, detections, next_id, max_distance)
        tracks = [t for t in tracks if t.missed <= max_missed]

        for t in tracks:
            if t.count_cooldown > 0:
                t.count_cooldown -= 1

            prev_side = signed_side(t.prev_center, line_center, line_normal)
            curr_side = signed_side(t.center, line_center, line_normal)
            crossed = prev_side * curr_side < 0.0 or abs(prev_side) <= 1.0 or abs(curr_side) <= 1.0

            move = np.array(
                [
                    float(t.center[0] - t.prev_center[0]),
                    float(t.center[1] - t.prev_center[1]),
                ],
                dtype=np.float32,
            )
            move_norm = float(np.linalg.norm(move))

            if crossed and move_norm >= 2.0 and t.count_cooldown == 0:
                score = float(np.dot(move / move_norm, desired_flow))
                if score >= 0:
                    total_in += 1
                else:
                    total_out += 1
                t.count_cooldown = 15

        draw_infinite_line(frame, line_center, line_dir)

        for r in detections:
            x, y, w, h = r
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        for t in tracks:
            cx, cy = t.center
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID {t.track_id}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"IN: {total_in}  OUT: {total_out}  OCCUPANCY: {total_in - total_out}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FLOW: {flow_direction}  ANGLE: {line_angle_deg:.0f} deg",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    report = {
        "frames": frame_idx,
        "line_angle_deg": line_angle_deg,
        "line_center_x_ratio": line_center_x_ratio,
        "line_center_y_ratio": line_center_y_ratio,
        "flow_direction": flow_direction,
        "in": total_in,
        "out": total_out,
        "occupancy": total_in - total_out,
        "input": input_path,
        "output": output_path,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    args = parse_args()
    report = run_occupancy(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        line_angle_deg=args.line_angle_deg,
        line_center_x_ratio=args.line_center_x_ratio,
        line_center_y_ratio=args.line_center_y_ratio,
        flow_direction=args.flow_direction,
        max_distance=args.max_distance,
        max_missed=args.max_missed,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
