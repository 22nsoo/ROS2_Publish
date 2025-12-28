#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
import time
import sys
import platform
import argparse

# ========== 기본 설정 (필요 시 CLI로 덮어쓰기 가능) ==========
DEFAULT_CAM_INDEX = 0
WIN_NAME  = "ROI Picker"
SAVE_JSON = "roi_points.json"
BASE_W, BASE_H = 1920, 1080  # 1080p 기준 좌표도 함께 저장

HELP_TEXT = [
    "마우스로 꼭짓점들을 시계/반시계 순서로 클릭하여 ROI를 만드세요.",
    "[z] 마지막 점 취소, [c] 모두 지우기",
    "[f] 프레임 고정/해제, [s] 저장 후 종료, [q] 저장 없이 종료"
]

def try_set_props(cap, w, h, fps):
    # 일부 카메라는 설정이 강제되지 않을 수 있음(최대한 시도만)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    # MJPG 포맷을 지원하는 UVC 카메라일 때 프레임 안정화에 도움
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

def make_gst_for_usb(dev_path, w, h, fps):
    # USB UVC 카메라용 GStreamer 파이프라인 (플랫폼 범용)
    # v4l2src → videoconvert → appsink
    return (
        f"v4l2src device={dev_path} ! "
        f"video/x-raw, width={w}, height={h}, framerate={fps}/1 ! "
        f"videoconvert ! appsink drop=true sync=false"
    )

def make_gst_for_csi(w, h, fps, flip=0):
    # Jetson CSI 카메라용 파이프라인 (nvarguscamerasrc 필요)
    # width/height는 센서 모드 지원 해상도와 일치해야 함
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
        "nvvidconv flip-method={flip} ! "
        "video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true sync=false"
    ).format(flip=flip)

def open_camera_flexible(index=0, w=1280, h=720, fps=30, device=None, use_gst=False, csi=False):
    """
    Linux/Jetson에서 안전하게 웹캠/CSI를 여는 함수.
    우선순위: (옵션)GStreamer → V4L2 → CAP_ANY
    Windows에선 DSHOW/MSMF도 fallback로 시도.
    """
    # 1) CSI 카메라 강제: nvarguscamerasrc 파이프라인
    if csi:
        gst = make_gst_for_csi(w, h, fps)
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            try_set_props(cap, w, h, fps)
            return cap
        else:
            raise RuntimeError("CSI 파이프라인(nvarguscamerasrc)을 열 수 없습니다. "
                               "Jetson인지, 카메라 연결/권한을 확인하세요.")

    # dev_path 결정
    dev_path = device if device else f"/dev/video{index}"

    # 2) GStreamer (USB/웹캠) 강제 요청 시
    if use_gst:
        gst = make_gst_for_usb(dev_path, w, h, fps)
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            try_set_props(cap, w, h, fps)
            return cap
        # 실패 시 다음 단계로 넘어감

    # 3) V4L2
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if cap.isOpened():
        try_set_props(cap, w, h, fps)
        return cap

    # 4) CAP_ANY
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if cap.isOpened():
        try_set_props(cap, w, h, fps)
        return cap

    # 5) (Windows일 경우) DSHOW → MSMF (Fallback)
    if platform.system().lower().startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            try_set_props(cap, w, h, fps)
            return cap
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            try_set_props(cap, w, h, fps)
            return cap

    raise RuntimeError(
        f"카메라를 열 수 없습니다. 시도한 경로: {dev_path} / 백엔드: GStreamer, V4L2, CAP_ANY"
    )

def draw_overlay(img, pts):
    vis = img.copy()
    # 안내문
    y = 24
    for line in HELP_TEXT:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        y += 24

    # 점/선
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 5, (255, 180, 0), -1)
        cv2.putText(vis, str(i), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,0), 1, cv2.LINE_AA)

    if len(pts) >= 2:
        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0,150,255), thickness=2)

    # 닫힌 폴리곤 미리보기
    if len(pts) >= 3:
        cv2.polylines(vis, [np.array(pts + [pts[0]], dtype=np.int32)], isClosed=True, color=(255,0,0), thickness=1)

    return vis

def parse_args():
    p = argparse.ArgumentParser(description="ROI Picker (USB/CSI, V4L2/GStreamer)")
    p.add_argument("--index", type=int, default=DEFAULT_CAM_INDEX, help="웹캠 인덱스 (/dev/video{index})")
    p.add_argument("--device", type=str, default=None, help="명시적 디바이스 경로 (예: /dev/video2)")
    # ▼▼ 여기만 변경: 카메라 '열 때' 기본 해상도를 1920x1080으로 요청 ▼▼
    p.add_argument("--width",  type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    # ▲▲ 여기만 변경 ▲▲
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--gst",    action="store_true", help="USB 웹캠을 GStreamer 파이프라인으로 강제 오픈")
    p.add_argument("--csi",    action="store_true", help="Jetson CSI 카메라(nvarguscamerasrc) 사용")
    p.add_argument("--save",   type=str, default=SAVE_JSON, help="저장 파일명(JSON)")
    return p.parse_args()

def main():
    args = parse_args()

    # 창 이름에 장치 정보 표시
    win_name = f"{WIN_NAME} (idx={args.index}{', CSI' if args.csi else ''}{', GST' if args.gst else ''})"

    # 카메라 열기 (여기서만 1920x1080 기본값 적용)
    cap = open_camera_flexible(
        index=args.index,
        w=args.width, h=args.height, fps=args.fps,
        device=args.device, use_gst=args.gst, csi=args.csi
    )

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    pts = []                # 현재 프레임 기준 픽셀 좌표 목록
    frozen = False          # 프레임 고정 여부
    frozen_frame = None

    # 마우스 콜백
    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x), int(y)))

    cv2.setMouseCallback(win_name, on_mouse)

    try:
        while True:
            if not frozen:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
            else:
                frame = frozen_frame.copy()

            vis = draw_overlay(frame, pts)
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('z'):   # undo
                if pts:
                    pts.pop()
            elif key == ord('c'): # clear
                pts.clear()
            elif key == ord('f'): # freeze/unfreeze
                if not frozen:
                    frozen = True
                    frozen_frame = frame.copy()
                else:
                    frozen = False
                    frozen_frame = None
            elif key == ord('s'): # save & exit
                if len(pts) < 3:
                    print("[WARN] 최소 3개의 점이 필요합니다(다각형).")
                    continue

                h, w = frame.shape[:2]
                pts_px = [{"x": int(x), "y": int(y)} for (x, y) in pts]

                # 정규화 (0~1)
                pts_norm = [{"x": float(x)/w, "y": float(y)/h} for (x, y) in pts]

                # 1920x1080 기준 스케일
                sx, sy = BASE_W / float(w), BASE_H / float(h)
                pts_1080p = [{"x": int(round(x * sx)), "y": int(round(y * sy))} for (x, y) in pts]

                payload = {
                    "frame_size": {"width": int(w), "height": int(h)},
                    "points_px": pts_px,
                    "points_norm": pts_norm,
                    "points_1080p": pts_1080p,
                    "base_reference": {"width": BASE_W, "height": BASE_H}
                }

                with open(args.save, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                print("\n=== ROI 저장 완료 ===")
                print(f"- 파일: {args.save}")
                print(f"- 현재 프레임 크기: {w}x{h}")
                print("- 픽셀 좌표(points_px):")
                print(np.array([(p["x"], p["y"]) for p in pts_px], dtype=np.int32))
                print("- 1920x1080 기준(points_1080p):")
                print(np.array([(p["x"], p["y"]) for p in pts_1080p], dtype=np.int32))
                break
            elif key == ord('q') or key == 27:  # q or ESC → exit without save
                print("[INFO] 저장 없이 종료합니다.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
