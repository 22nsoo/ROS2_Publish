#!/usr/bin/env python3
import cv2
import numpy as np
import time
from pyk4a import Config, PyK4A, ColorResolution, DepthMode
from ultralytics import YOLO

# ===== ROS2 =====
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# ------------------------------
# Global ROS node & publisher
# ------------------------------
_ros_node = None
_pub_count = None

def _ensure_ros():
    """Initialize ROS2 node & publisher once."""
    global _ros_node, _pub_count
    if not rclpy.ok():
        rclpy.init()
    if _ros_node is None:
        _ros_node = Node("roi_people_counter")
        _pub_count = _ros_node.create_publisher(Int32, "/roi/people_count", 10)

def _publish_count(n):
    """Publish people count and spin once for callbacks."""
    msg = Int32()
    msg.data = int(n)
    _pub_count.publish(msg)
    # process ROS events briefly
    rclpy.spin_once(_ros_node, timeout_sec=0.0)

def run_tracking():
    """
    Azure Kinect + YOLOv8 Pose tracking.
    Publish only the number of people whose BOTH ankles (kpts 15,16) are inside ROI.
    """
    _ensure_ros()

    # ===== Azure Kinect init =====
    config = Config(
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=DepthMode.OFF,
        synchronized_images_only=False
    )
    k4a = PyK4A(config)
    k4a.start()

    # ===== YOLOv8 Pose model =====
    model = YOLO("yolov8m-pose.pt")

    # ===== ROI polygon (edit as needed) =====
    roi_pts = np.array([
        [586, 428],
        [1091, 434],
        [1253, 1078],
        [231, 1079]
    ], dtype=np.int32)

    # ===== State tables (kept minimal; we only publish count) =====
    roi_status = {}      # tid -> bool(in_roi)
    roi_entry_time = {}  # tid -> enter timestamp (not used for publishing)

    # Pose skeleton for visualization (optional)
    SKELETON_CONNECTIONS = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 11), (6, 12),
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (0, 6)
    ]

    conf_thresh = 0.7
    show_window = True   # set False for headless

    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is None:
                _publish_count(0)
                continue

            frame_bgr = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

            # Run tracking (only class 0: person)
            results = model.track(
                source=frame_bgr,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                verbose=False
            )

            result = results[0]
            current_ids = set()
            ids_in_roi = []

            if getattr(result, "keypoints", None) is not None and getattr(result, "boxes", None) is not None:
                track_ids = result.boxes.id  # may be None / tensor / ndarray
                if track_ids is not None:
                    for box, conf, track_id, keypoint_set in zip(
                        result.boxes.xyxy,
                        result.boxes.conf,
                        track_ids,
                        result.keypoints.xy
                    ):
                        if conf is None or conf < conf_thresh:
                            continue
                        if track_id is None:
                            continue

                        # normalize track id to int
                        try:
                            tid = int(track_id.item())
                        except Exception:
                            tid = int(track_id)

                        current_ids.add(tid)

                        # BOTH ankles (15,16) inside ROI
                        foot_in_roi = []
                        for kp_idx in [15, 16]:
                            x, y = keypoint_set[kp_idx]
                            if x > 0 and y > 0:
                                in_roi = cv2.pointPolygonTest(roi_pts, (int(x), int(y)), False) >= 0
                                foot_in_roi.append(in_roi)
                            else:
                                foot_in_roi.append(False)

                        in_roi = all(foot_in_roi)
                        if in_roi:
                            ids_in_roi.append(tid)
                            if not roi_status.get(tid, False):
                                roi_status[tid] = True
                                roi_entry_time[tid] = time.time()
                        else:
                            if roi_status.get(tid, False):
                                roi_status[tid] = False
                            roi_entry_time.pop(tid, None)

                        # ===== visualization (optional) =====
                        x1a, y1a, x2a, y2a = map(int, box.tolist())
                        cv2.rectangle(frame_bgr, (x1a, y1a), (x2a, y2a), (0, 255, 0), 2)
                        cv2.putText(
                            frame_bgr, f"ID:{tid} Conf:{float(conf):.2f}", (x1a, y1a - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )
                        for i, (x, y) in enumerate(keypoint_set):
                            if x > 0 and y > 0:
                                xi, yi = int(x), int(y)
                                cv2.circle(frame_bgr, (xi, yi), 3, (255, 0, 255), -1)
                                cv2.putText(frame_bgr, str(i), (xi + 3, yi - 3),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                        for a, b in SKELETON_CONNECTIONS:
                            x1, y1 = keypoint_set[a]
                            x2, y2 = keypoint_set[b]
                            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                cv2.line(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            # Keep states for currently visible IDs only
            roi_status = {tid: roi_status[tid] for tid in roi_status if tid in current_ids}
            roi_entry_time = {tid: roi_entry_time[tid] for tid in roi_entry_time if tid in current_ids}

            # ===== Publish only the count =====
            count = len(set(ids_in_roi))
            _publish_count(count)

            # ===== Visualization of ROI & count =====
            if show_window:
                cv2.polylines(frame_bgr, [roi_pts], isClosed=True, color=(255, 0, 0), thickness=2)
                cv2.putText(frame_bgr, f"ROI Count: {count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                cv2.imshow("ROI + YOLOv8 Pose Tracking (people_count only)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        # Graceful shutdown
        try:
            k4a.stop()
        except Exception:
            pass
        if show_window:
            cv2.destroyAllWindows()
        # shutdown ROS
        if _ros_node is not None:
            _ros_node.destroy_node()
        rclpy.shutdown()
    return 0  # normal exit

if __name__ == "__main__":
    run_tracking()
