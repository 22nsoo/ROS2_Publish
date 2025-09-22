#!/usr/bin/env python3
"""
ROS2 node: Webcam + YOLOv8 (COCO 80 classes by default) → publish, for **every detected object**:
  - "/detections/object"  (std_msgs/String):  "<class_name>, ID : <track_id>"
  - "/detections/summary" (std_msgs/String):  comma-joined summary for the frame

Notes
- Uses Ultralytics tracking (ByteTrack/BotSORT) to obtain persistent track IDs per object.
- If your weights are COCO-pretrained (e.g., yolov8m.pt), classes=80.
- For custom datasets, class names/ids come from model.names.

Install
  python3 -m pip install ultralytics==8.* opencv-python rclpy lapx filterpy
Run
  source /opt/ros/$ROS_DISTRO/setup.bash
  python3 ros2_detect_all_classes_strings.py --ros-args \
    -p model_path:=yolov8m.pt -p device:=cuda:0 -p conf:=0.35 -p show:=False -p tracking:=True
"""

import os
import time
from typing import List

import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics is required: pip install ultralytics") from e

class DetectAllClassesStrings(Node):
    def __init__(self):
        super().__init__('detect_all_classes_strings')

        # ---------- Parameters ----------
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('model_path', 'yolov8m.pt')
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('target_fps', 15)
        self.declare_parameter('show', False)
        self.declare_parameter('tracking', True)
        self.declare_parameter('tracker', 'bytetrack.yaml')

        self.camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.model_path   = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf         = self.get_parameter('conf').get_parameter_value().double_value
        self.imgsz        = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.device       = self.get_parameter('device').get_parameter_value().string_value
        self.target_fps   = self.get_parameter('target_fps').get_parameter_value().integer_value
        self.show         = self.get_parameter('show').get_parameter_value().bool_value
        self.tracking     = self.get_parameter('tracking').get_parameter_value().bool_value
        self.tracker      = self.get_parameter('tracker').get_parameter_value().string_value

        # ---------- Model ----------
        if not os.path.exists(self.model_path):
            self.get_logger().warn(f"model_path '{self.model_path}' not found; trying ultralytics cache/hub...")
        self.model = YOLO(self.model_path)
        self.names = self.model.names  # dict: id -> name
        self.get_logger().info(f"Loaded YOLO: {self.model_path}; classes={len(self.names)}")

        # ---------- Camera ----------
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index {self.camera_index}")

        # ---------- Publishers ----------
        self.pub_object  = self.create_publisher(String, '/detections/object', 20)
        self.pub_summary = self.create_publisher(String, '/detections/summary', 10)

        # ---------- Timer Loop ----------
        period = 1.0 / max(1, self.target_fps)
        self.frame_interval = period
        self.last_t = 0.0
        self.timer = self.create_timer(period, self.on_timer)

    def on_timer(self):
        now = time.time()
        if now - self.last_t < self.frame_interval:
            return
        self.last_t = now

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warning('Failed to read frame from webcam.')
            return

        try:
            if self.tracking:
                results = self.model.track(
                    frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                    tracker=self.tracker, persist=True, verbose=False
                )
            else:
                results = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        except Exception as e:
            self.get_logger().error(f"YOLO error: {e}")
            return

        if not results or len(results) == 0:
            return
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is None or boxes.cls is None or len(boxes) == 0:
            # No detections — you may publish an empty summary if desired
            return

        cls_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        tids: List[int] = []
        if hasattr(boxes, 'id') and boxes.id is not None:
            tids = boxes.id.detach().cpu().numpy().astype(int).tolist()
        else:
            tids = [None] * len(cls_ids)

        # Per-detection publish and build summary
        parts: List[str] = []
        for i, cid in enumerate(cls_ids):
            cname = self.names.get(int(cid), str(cid))
            tid   = tids[i] if i < len(tids) else None
            text  = f"{cname}, ID : {tid if tid is not None else 'None'}"

            msg_obj = String(); msg_obj.data = text
            self.pub_object.publish(msg_obj)  # publish per detection
            parts.append(text)

        # Frame summary (comma-joined)
        msg_sum = String(); msg_sum.data = ', '.join(parts)
        self.pub_summary.publish(msg_sum)

        # Optional preview
        if self.show:
            try:
                annotated = results[0].plot()
                cv2.imshow('YOLOv8 all-classes strings', annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    rclpy.shutdown()
            except Exception:
                self.show = False

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = DetectAllClassesStrings()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
