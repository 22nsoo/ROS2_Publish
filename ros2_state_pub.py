#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 webcam → ROS2 state publisher (debounced) + minimal JSON payload
Publishes:
  1) Plain "BWC" bits to --topic
  2) JSON with keys (only these five):
     {
       "server_ip": "...",   # from --server-ip
       "local_ip": "...",    # auto-detected
       "umbrella": 0/1,
       "book": 0/1,
       "apple": 0/1
     }
to --json-topic when the debounced state changes.
DB 연동 없음.
"""

import argparse
import sys
from typing import Dict, List, Tuple
import socket
from datetime import datetime

import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json

# ============================
# Class mapping (custom model)
# ============================
CLASS_IDS = {
    "apple": 0,
    "book": 1,
    "umbrella": 2,
}
ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}


# -------------------------
# Local IP auto detection
# -------------------------
def get_local_ip() -> str:
    """
    현재 장치의 로컬 IPv4 주소를 자동으로 탐색합니다.
    네트워크 인터페이스 중 실제로 사용 가능한 IP를 반환하며,
    실패 시 127.0.0.1을 반환합니다.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 실제로 연결되지 않아도 OS가 사용 중인 네트워크 인터페이스를 결정함
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class StatePublisher(Node):
    def __init__(self,
                 topic_name: str,
                 json_topic_name: str,
                 server_ip: str,
                 debounce_frames: int = 15):
        super().__init__('object_detector')
        self.pub_state = self.create_publisher(String, topic_name, 10)
        self.pub_json = self.create_publisher(String, json_topic_name, 10)

        # server/local ip
        self.server_ip = server_ip
        self.local_ip = get_local_ip()

        # 디바운싱 상태
        self.last_state = None  # type: str
        self.pending_state = None  # type: str
        self.pending_count = 0
        self.required = max(1, int(debounce_frames))

    def _publish(self, state_code: str, bits: Dict[str, int]):
        # 1) 평문 상태코드(BWC) 퍼블리시
        msg = String()
        msg.data = state_code  # e.g., "101" (book, apple, umbrella)
        self.pub_state.publish(msg)

        # 2) 최소 JSON 포맷 퍼블리시
        payload = {
            "server_ip": self.server_ip,
            "local_ip": self.local_ip,
            "umbrella": int(bits.get("umbrella", 0)),
            "book": int(bits.get("book", 0)),
            "apple": int(bits.get("apple", 0))
        }
        json_msg = String()
        json_msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_json.publish(json_msg)

        self.last_state = state_code
        self.get_logger().info(f'Published state: {state_code} | JSON: {payload}')

    def maybe_publish(self, current_state: str, bits: Dict[str, int]):
        """
        현재 프레임의 상태 문자열(current_state)이
        self.required 프레임 연속 유지될 때만 퍼블리시.
        """
        # 이미 퍼블리시된 상태와 같다면 후보 초기화(변화 없음)
        if self.last_state == current_state:
            self.pending_state = None
            self.pending_count = 0
            return

        # 퍼블리시된 상태와 다르다면, 후보 상태를 카운팅
        if self.pending_state == current_state:
            self.pending_count += 1
        else:
            self.pending_state = current_state
            self.pending_count = 1

        # 충분히 안정화되면 퍼블리시
        if self.pending_count >= self.required:
            self._publish(current_state, bits)
            # 퍼블리시 후 후보 초기화
            self.pending_state = None
            self.pending_count = 0


def build_bits(detections: List[Dict]) -> Dict[str, int]:
    """Return per-class presence bits."""
    bits = {"book": 0, "apple": 0, "umbrella": 0}
    for d in detections:
        name = d.get("class_name")
        if name in bits:
            bits[name] = 1
    return bits


def build_state_code_from_bits(bits: Dict[str, int]) -> str:
    """Build a 3-bit string in order: book, apple, umbrella."""
    return f'{bits["book"]}{bits["apple"]}{bits["umbrella"]}'


def draw_boxes_and_bars(frame, result, conf_thres: float) -> Tuple[List[Dict], any]:
    """
    Draw boxes + labels + confidence bars for target classes.
    Returns (detections_list, annotated_frame).
    """
    detections: List[Dict] = []
    annotated = frame

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

        # Colors: pass (>=thres) green, else red
        color = (0, 200, 0) if score >= conf_thres else (0, 0, 255)

        # Draw rectangle
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), color, 2)

        # Label text
        label = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1i, max(0, y1i - th - 6)), (x1i + tw + 6, y1i), color, -1)
        cv2.putText(annotated, label, (x1i + 3, y1i - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Confidence bar
        bar_w = max(60, (x2i - x1i))
        bar_h = 6
        bar_x1 = x1i
        bar_y1 = min(y2i + 14, annotated.shape[0] - 10)
        bar_x2 = bar_x1 + bar_w
        bar_y2 = bar_y1 + bar_h

        # background bar
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), -1)
        # filled portion
        fill_w = int(bar_w * max(0.0, min(1.0, score)))
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), color, -1)

        # threshold tick
        tick_x = bar_x1 + int(bar_w * max(0.0, min(1.0, conf_thres)))
        cv2.line(annotated, (tick_x, bar_y1 - 2), (tick_x, bar_y2 + 2), (255, 255, 255), 2)

        detections.append({
            "class_id": cid,
            "class_name": name,
            "score": round(score, 4),
            "bbox_xyxy": [x1i, y1i, x2i, y2i],
        })

    return detections, annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.pt')
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--topic', type=str, default='/object_detector/detected_state')
    parser.add_argument('--json-topic', type=str, default='/object_detector/detected_state_json',
                        help='JSON 메시지를 퍼블리시할 토픽 이름')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto','v4l2','any'])
    parser.add_argument('--fourcc', type=str, default='MJPG')
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--no-window', action='store_true')
    parser.add_argument('--disp-width', type=int, default=960)
    parser.add_argument('--disp-height', type=int, default=540)

    # 서버 IP는 인자로 받고, 로컬 IP는 내부에서 자동 탐색
    parser.add_argument('--server-ip', type=str, default = "192.168.10.110")

    # 디바운싱 프레임 수
    parser.add_argument('--debounce-frames', type=int, default=15,
                        help='연속 프레임 수가 이 값 이상일 때만 상태 변화 publish')
    args = parser.parse_args()

    rclpy.init(args=None)
    node = StatePublisher(
        topic_name=args.topic,
        json_topic_name=args.json_topic,
        server_ip=args.server_ip,
        debounce_frames=args.debounce_frames
    )

    try:
        model = YOLO(args.model)
    except Exception as e:
        node.get_logger().error(f'Failed to load model: {e}')
        rclpy.shutdown()
        sys.exit(1)

    backend = None
    if args.backend == 'v4l2':
        backend = cv2.CAP_V4L2
    elif args.backend == 'any':
        backend = 0

    cap = cv2.VideoCapture(args.camera, backend) if backend is not None else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        node.get_logger().error(f'Cannot open camera index {args.camera}')
        rclpy.shutdown()
        sys.exit(1)

    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win = 'YOLOv8 State (book,apple,umbrella) — debounced'
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, args.disp_width, args.disp_height)

    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok or frame is None:
                node.get_logger().warn('Failed to read frame')
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            r = results[0] if results else None

            detections, overlay = draw_boxes_and_bars(frame.copy(), r, args.conf)

            # 비트 & 상태코드 작성
            bits = build_bits(detections)
            state_now = build_state_code_from_bits(bits)

            # 연속 N프레임 유지 시에만 퍼블리시
            node.maybe_publish(state_now, bits)

            if not args.no_window:
                disp = cv2.resize(overlay, (args.disp_width, args.disp_height), interpolation=cv2.INTER_LINEAR)
                cv2.imshow(win, disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            rclpy.spin_once(node, timeout_sec=0.0)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
