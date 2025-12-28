#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 webcam → ROS2 state publisher (debounced) + JSON(payload + counts)
+ ROI(poly, 1920x1080 기준 좌표 → 프레임 크기로 자동 스케일링)

Publishes:
  1) Plain state code bits to --topic
     - Order: ORDERED_KEYS 순서
  2) Presence JSON (0/1) to --json-topic
  3) Counts JSON (종류 수 + 개수) to --json-topic-counts

추가:
  - MediaPipe Hands 사용
    * 손 관절 시각화(랜드마크 + 연결선)
    * "손이 ROI 안에 있을 때만" YOLO 추론 및 ROS2 publish 차단
    * 손이 ROI 밖에 있으면 YOLO는 그대로 동작, 손 관절도 화면에 표시
  - MediaPipe는 매 N프레임(HAND_INTERVAL)마다만 실행 → FPS 향상
"""

import argparse
import sys
from typing import Dict, List, Tuple, Optional
import socket
from collections import Counter

import cv2
from ultralytics import YOLO
import torch
import numpy as np
import mediapipe as mp  # ✅ MediaPipe

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


# ============================
# MediaPipe globals
# ============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ============================
# Class mapping (custom model)
# ============================
CLASS_IDS = {
    "basket": 1,
    "book": 2,
    "bowl": 3,
    "cup": 4,
    "dish": 5,
    "laptop": 6,
    "remote": 7,
    "snack": 8,
}
initial_num = {
    "basket": 1,
    "book": 2,
    "bowl": 1,
    "cup": 1,
    "dish": 1,
    "laptop": 1,
    "remote": 1,
    "snack": 1,
}
ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}

# 고정 순서 (state code 및 JSON 출력 순서)
ORDERED_KEYS = ["cup", "book", "bowl", "dish",
                "laptop", "remote", "snack", "basket"]


# ============================
# ROI (1920x1080 기준 좌표) #############################
# ============================
roi_pts_1080p = np.array([
    [1547,  212],
    [   1,  253],
    [   2, 1079],
    [1692, 1078],

], dtype=np.int32)
base_w, base_h = 1920, 1080


# -------------------------
# Local IP auto detection
# -------------------------
def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# -------------------------
# ROI helpers
# -------------------------
class ROI:
    def __init__(self, poly: Optional[np.ndarray] = None):
        self.poly = poly  # np.ndarray (N,2) int32

    @property
    def enabled(self) -> bool:
        return self.poly is not None

    def contains_point(self, x: float, y: float) -> bool:
        if not self.enabled:
            return True
        return cv2.pointPolygonTest(self.poly.astype(np.float32),
                                    (float(x), float(y)), False) >= 0

    def draw(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        pts = self.poly.reshape((-1, 1, 2))
        overlay = frame.copy()
        alpha = 0.25
        color = (0, 255, 255)
        thick = 3
        cv2.fillPoly(overlay, [pts], color=color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thick)

def scale_poly_to_frame(poly, base_w, base_h, frame_w, frame_h) -> np.ndarray:
    sx = float(frame_w) / float(base_w)
    sy = float(frame_h) / float(base_h)
    p = poly.astype(np.float32).copy()
    p[:, 0] *= sx
    p[:, 1] *= sy
    return p.astype(np.int32)


def clamp_poly_to_size(poly: np.ndarray, w, h) -> np.ndarray:
    poly = poly.copy()
    poly[:, 0] = np.clip(poly[:, 0], 0, max(0, w - 1))
    poly[:, 1] = np.clip(poly[:, 1], 0, max(0, h - 1))
    return poly


# -------------------------
# Helper functions
# -------------------------
def build_counts(detections: List[Dict]) -> Dict[str, int]:
    cnt = Counter(d.get("class_name") for d in detections)
    return {k: int(cnt.get(k, 0)) for k in ORDERED_KEYS}


def bits_from_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return {k: 1 if int(counts.get(k, 0)) > 0 else 0 for k in ORDERED_KEYS}


def build_state_code_from_bits(bits: Dict[str, int]) -> str:
    return "".join(str(int(bits.get(k, 0))) for k in ORDERED_KEYS)


def build_state_signature(bits: Dict[str, int], counts: Dict[str, int]) -> str:
    bit_str = "".join(str(int(bits.get(k, 0))) for k in ORDERED_KEYS)
    count_str = ",".join(str(int(counts.get(k, 0))) for k in ORDERED_KEYS)
    return f"{bit_str}|{count_str}"


# -------------------------
# Draw boxes & confidence bars (+ ROI 필터)
# -------------------------
def draw_boxes_and_bars(frame, result, conf_thres: float,
                        roi: ROI) -> Tuple[List[Dict], any]:
    detections: List[Dict] = []
    annotated = frame

    if roi.enabled:
        roi.draw(annotated)

    if result is None or not hasattr(result, 'boxes') or result.boxes is None:
        return detections, annotated

    boxes = result.boxes
    cls = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    conf = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []

    for c, s, (x1, y1, x2, y2) in zip(cls, conf, xyxy):
        cid = int(c)
        if cid not in ID_TO_NAME:
            continue

        name = ID_TO_NAME[cid]
        score = float(s)

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        in_roi = roi.contains_point(cx, cy) if roi.enabled else True

        base_color = (0, 200, 0) if score >= conf_thres else (0, 0, 255)
        color = base_color if in_roi else (0, 0, 180)
        thickness = 2 if in_roi else 1

        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), color, thickness)

        label = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        if in_roi:
            cv2.rectangle(annotated,
                          (x1i, max(0, y1i - th - 6)),
                          (x1i + tw + 6, y1i),
                          color, -1)
            cv2.putText(annotated, label, (x1i + 3, y1i - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, label,
                        (x1i + 3, max(15, y1i - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 1, cv2.LINE_AA)

        if in_roi:
            detections.append({
                "class_id": cid,
                "class_name": name,
                "score": round(score, 4),
                "bbox_xyxy": [x1i, y1i, x2i, y2i],
                "center_xy": [int(cx), int(cy)],
            })

    return detections, annotated


# -------------------------
# ROS2 Node
# -------------------------
class StatePublisher(Node):
    def __init__(self, topic_name: str, json_topic_name: str, server_ip: str,
                 debounce_frames: int = 20,
                 json_topic_counts: str = '/object_detector/detected_counts_json'):
        super().__init__('object_detector')
        self.pub_state = self.create_publisher(String, topic_name, 10)
        self.pub_json = self.create_publisher(String, json_topic_name, 10)
        self.pub_json_counts = self.create_publisher(String, json_topic_counts, 10)

        self.server_ip = server_ip
        self.local_ip = get_local_ip()

        self.last_signature = None
        self.pending_signature = None
        self.pending_bits = None
        self.pending_counts = None
        self.pending_count = 0
        self.required = max(1, int(debounce_frames))

    def _publish(self, state_code: str, bits: Dict[str, int], counts: Dict[str, int]):
        msg = String()
        msg.data = state_code
        self.pub_state.publish(msg)

        payload = {
            "FromUrl": self.local_ip,
            "ToUrl": self.server_ip,
            **{k: int(bits.get(k, 0)) for k in ORDERED_KEYS}
        }
        json_msg = String()
        json_msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_json.publish(json_msg)

        type_num = len(ORDERED_KEYS)
        payload_counts = {
            "FromUrl": self.local_ip,
            "ToUrl": self.server_ip,
            "num": type_num,
            **{k: (int(counts.get(k, 0)), int(initial_num[k]))
               for k in ORDERED_KEYS}
        }

        json_msg2 = String()
        json_msg2.data = json.dumps(payload_counts, ensure_ascii=False)
        self.pub_json_counts.publish(json_msg2)

        self.last_signature = build_state_signature(bits, counts)

    def maybe_publish(self, state_code: str, bits: Dict[str, int],
                      counts: Dict[str, int]):
        current_sig = build_state_signature(bits, counts)

        if self.last_signature == current_sig:
            self.pending_signature = None
            self.pending_bits = None
            self.pending_counts = None
            self.pending_count = 0
            return

        if self.pending_signature == current_sig:
            self.pending_count += 1
        else:
            self.pending_signature = current_sig
            self.pending_bits = bits
            self.pending_counts = counts
            self.pending_count = 1

        if self.pending_count >= self.required:
            self._publish(state_code, self.pending_bits, self.pending_counts)
            self.pending_signature = None
            self.pending_bits = None
            self.pending_counts = None
            self.pending_count = 0


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo11is.pt')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.35) #######################
    parser.add_argument('--topic', type=str,
                        default='/object_detector/detected_state')
    parser.add_argument('--json-topic', type=str,
                        default='/object_detector/detected_state_json')
    parser.add_argument('--json-topic-counts', type=str,
                        default='/object_detector/detected_counts_json')
    parser.add_argument('--server-ip', type=str, default="192.168.10.110")
    parser.add_argument('--debounce-frames', type=int, default=15)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--half', dest='half', action='store_true')
    parser.add_argument('--no-half', dest='half', action='store_false')
    parser.set_defaults(half=True)
    parser.add_argument('--no-window', action='store_true')

    args = parser.parse_args()

    # Device
    if args.device == 'cpu':
        device = 'cpu'
    elif args.device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_half = bool(args.half and device == 'cuda')
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    rclpy.init(args=None)
    node = StatePublisher(
        topic_name=args.topic,
        json_topic_name=args.json_topic,
        server_ip=args.server_ip,
        debounce_frames=args.debounce_frames,
        json_topic_counts=args.json_topic_counts
    )

    # ✅ MediaPipe Hands 초기화 (loop 밖에서 한 번만 생성)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        model = YOLO(args.model)
    except Exception as e:
        node.get_logger().error(f'Failed to load model: {e}')
        hands.close()
        rclpy.shutdown()
        sys.exit(1)

    try:
        model.to(device)
        node.get_logger().info(f'Using device={device} half={use_half}')
    except Exception as e:
        node.get_logger().warn(f'Failed to move model to device {device}: {e}')

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        node.get_logger().error(f'Cannot open camera index {args.camera}')
        hands.close()
        rclpy.shutdown()
        sys.exit(1)

    win = 'YOLOv8 State + ROI(poly) + Hand block (ROI only)'
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

    ok, first = cap.read()
    if not ok:
        node.get_logger().error('Failed to read first frame')
        hands.close()
        rclpy.shutdown()
        sys.exit(1)
    H, W = first.shape[:2]

    roi_scaled = scale_poly_to_frame(roi_pts_1080p, base_w, base_h, W, H)
    roi_scaled = clamp_poly_to_size(roi_scaled, W, H)
    roi_obj = ROI(poly=roi_scaled)
    node.get_logger().info(
        f'ROI scaled to frame (W,H= {W},{H}): {roi_scaled.tolist()}')

    frame_buffer = first
    frame_idx = 0  # ✅ 프레임 카운터

    # MediaPipe 실행 프레임 간격 (2면 절반 프레임만 손 검출)
    HAND_INTERVAL = 2

    try:
        while rclpy.ok():
            frame = frame_buffer
            frame_buffer = None

            if frame is None:
                ok, frame = cap.read()
                if not ok:
                    node.get_logger().warn('Failed to read frame')
                    rclpy.spin_once(node, timeout_sec=0.0)
                    continue

            frame_idx += 1
            H_cur, W_cur = frame.shape[:2]

            hand_result = None
            hand_detected = False
            hand_in_roi = False

            # ================================
            # ✅ MediaPipe로 손 검출 + ROI 판정 (N프레임마다)
            # ================================
            if frame_idx % HAND_INTERVAL == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_result = hands.process(rgb)
                hand_detected = (
                    hand_result.multi_hand_landmarks is not None and
                    len(hand_result.multi_hand_landmarks) > 0
                )

                if hand_detected and roi_obj.enabled:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            px = lm.x * W_cur
                            py = lm.y * H_cur
                            if roi_obj.contains_point(px, py):
                                hand_in_roi = True
                                break
                        if hand_in_roi:
                            break

            # ------------------------
            # 손이 ROI 안에 있으면: YOLO/publish 차단
            # ------------------------
            if hand_detected and hand_in_roi:
                overlay = frame.copy()
                roi_obj.draw(overlay)

                if hand_result is not None:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            overlay,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )


                if not args.no_window:
                    cv2.imshow(win, overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                rclpy.spin_once(node, timeout_sec=0.0)
                continue
            # ================================

            # 여기까지 왔다는 건:
            # - 손이 없거나
            # - 손이 있어도 ROI 밖
            # → YOLO 정상 동작
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                half=use_half,
                verbose=False
            )
            r = results[0] if results else None

            detections, overlay = draw_boxes_and_bars(
                frame.copy(), r, args.conf, roi_obj)

            # 손이 있지만 ROI 밖이면, 관절만 표시 (block 없이)
            if hand_detected and hand_result is not None:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        overlay,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            counts = build_counts(detections)
            bits = bits_from_counts(counts)
            state_now = build_state_code_from_bits(bits)

            node.maybe_publish(state_now, bits, counts)

            if not args.no_window:
                cv2.imshow(win, overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            rclpy.spin_once(node, timeout_sec=0.0)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        cap.release()
        hands.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
