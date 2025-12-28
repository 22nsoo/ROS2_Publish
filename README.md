# Azure Kinect ROI Tracking with YOLO and ROS2

이 프로젝트는 웹캠을 사용하여 ROI(Region of Interest) 영역 내의 객체를 YOLOv11로 검출하고, MediaPipe Hands를 활용한 손 인식 기능과 함께 ROS2를 통해 상태를 발행하는 시스템입니다.

## 📁 파일 구조

```
.
├── pick.py              # ROI 영역 선택 도구
├── test.py              # YOLO 객체 검출 및 ROS2 퍼블리셔
├── yolo11is.pt          # YOLOv8 모델 파일
└── README.md            # 프로젝트 문서
```

## 🚀 주요 기능

### 1. ROI 선택 도구 (`pick.py`)
- 마우스를 사용하여 카메라 화면에서 ROI 영역을 선택
- USB/CSI 카메라 지원
- Jetson CSI 카메라 지원
- 선택한 ROI를 JSON 파일로 저장 (픽셀 좌표, 정규화 좌표, 1920x1080 기준 좌표)

### 2. 객체 검출 및 ROS2 퍼블리싱 (`test.py`)
- **YOLOv11**: 8가지 객체 클래스 검출
  - basket, book, bowl, cup, dish, laptop, remote, snack
- **MediaPipe Hands**: 손 인식 기능
  - 손이 ROI 영역 안에 있을 때 YOLO 추론 및 ROS2 발행 차단
  - 손 관절 시각화 (랜드마크 + 연결선)
- **ROS2 통신**: 
  - 상태 코드 비트 스트링 발행
  - JSON 형식의 객체 존재 여부 발행
  - 객체 개수 정보 발행
- **Debouncing**: 상태 변화 안정화를 위한 프레임 기반 디바운싱

## 📋 요구사항

### Python 패키지
```bash
pip install opencv-python
pip install ultralytics
pip install torch
pip install numpy
pip install mediapipe
pip install rclpy
```

### 시스템 요구사항
- Python 3.x
- ROS2 (Humble 또는 최신 버전)
- CUDA 지원 GPU (선택사항, CPU 모드도 지원)
- USB 카메라 또는 Azure Kinect 카메라

## 🔧 사용 방법

### 1. ROI 영역 선택

```bash
python pick.py [옵션]
```

**주요 옵션:**
- `--index N`: 웹캠 인덱스 (기본값: 0)
- `--device PATH`: 명시적 디바이스 경로 (예: `/dev/video2`)
- `--width W`: 카메라 해상도 너비 (기본값: 1920)
- `--height H`: 카메라 해상도 높이 (기본값: 1080)
- `--fps FPS`: 프레임레이트 (기본값: 30)
- `--gst`: USB 웹캠을 GStreamer 파이프라인으로 강제 오픈
- `--csi`: Jetson CSI 카메라 사용
- `--save FILE`: 저장 파일명 (기본값: `roi_points.json`)

**조작 방법:**
- 마우스 왼쪽 클릭: ROI 꼭짓점 추가
- `z`: 마지막 점 취소
- `c`: 모든 점 지우기
- `f`: 프레임 고정/해제
- `s`: 저장 후 종료
- `q` 또는 `ESC`: 저장 없이 종료

### 2. 객체 검출 및 ROS2 발행

```bash
python test.py [옵션]
```

**주요 옵션:**
- `--model PATH`: YOLO 모델 파일 경로 (기본값: `yolo11is.pt`)
- `--camera N`: 카메라 인덱스 (기본값: 0)
- `--imgsz SIZE`: YOLO 입력 이미지 크기 (기본값: 640)
- `--conf THRESHOLD`: 신뢰도 임계값 (기본값: 0.35)
- `--topic TOPIC`: ROS2 상태 토픽 (기본값: `/object_detector/detected_state`)
- `--json-topic TOPIC`: ROS2 JSON 토픽 (기본값: `/object_detector/detected_state_json`)
- `--json-topic-counts TOPIC`: ROS2 개수 토픽 (기본값: `/object_detector/detected_counts_json`)
- `--server-ip IP`: 서버 IP 주소 (기본값: `192.168.10.110`)
- `--debounce-frames N`: 디바운스 프레임 수 (기본값: 15)
- `--device DEVICE`: 실행 디바이스 (`auto`, `cpu`, `cuda`, 기본값: `auto`)
- `--half`: FP16 반정밀도 사용 (CUDA만)
- `--no-window`: GUI 창 비활성화

**예시:**
```bash
# 기본 실행
python test.py

# CUDA 사용 및 신뢰도 조정
python test.py --device cuda --conf 0.5

# GUI 없이 실행
python test.py --no-window
```

## 📊 ROS2 토픽

### 1. `/object_detector/detected_state` (String)
- 객체 존재 여부를 비트 스트링으로 발행
- 형식: `"01001000"` (8자리 비트)
- 순서: cup, book, bowl, dish, laptop, remote, snack, basket

### 2. `/object_detector/detected_state_json` (String)
- JSON 형식의 객체 존재 여부
```json
{
  "FromUrl": "192.168.1.100",
  "ToUrl": "192.168.10.110",
  "cup": 1,
  "book": 0,
  "bowl": 1,
  ...
}
```

### 3. `/object_detector/detected_counts_json` (String)
- 객체 개수 정보
```json
{
  "FromUrl": "192.168.1.100",
  "ToUrl": "192.168.10.110",
  "num": 8,
  "cup": [2, 1],
  "book": [1, 2],
  ...
}
```

## 🎯 ROI 설정

ROI는 `test.py` 파일 내에서 하드코딩되어 있습니다:

```python
roi_pts_1080p = np.array([
    [1547,  212],
    [   1,  253],
    [   2, 1079],
    [1692, 1078],
], dtype=np.int32)
```

이 좌표는 1920x1080 해상도 기준이며, 실제 카메라 해상도에 따라 자동으로 스케일링됩니다.

`pick.py`를 사용하여 새로운 ROI를 선택하고 JSON 파일로 저장한 후, `test.py`에서 해당 좌표를 사용하도록 수정할 수 있습니다.

## 🔍 객체 클래스

| ID | 클래스명 | 초기 개수 |
|----|---------|----------|
| 1  | basket  | 1        |
| 2  | book    | 2        |
| 3  | bowl    | 1        |
| 4  | cup     | 1        |
| 5  | dish    | 1        |
| 6  | laptop  | 1        |
| 7  | remote  | 1        |
| 8  | snack   | 1        |

## 🛠️ 기술 스택

- **컴퓨터 비전**: OpenCV, YOLOv11 (Ultralytics)
- **손 인식**: MediaPipe Hands
- **딥러닝 프레임워크**: PyTorch
- **로봇 통신**: ROS2 (rclpy)

## 📝 참고사항

- MediaPipe Hands는 성능 최적화를 위해 매 N프레임(`HAND_INTERVAL=2`)마다만 실행됩니다.
- 손이 ROI 영역 안에 있을 때는 YOLO 추론과 ROS2 발행이 차단되지만, 손 관절은 계속 화면에 표시됩니다.
- ROI 영역 밖의 객체는 검출되지만 ROS2 발행에는 포함되지 않습니다.
- 상태 변화는 디바운싱을 통해 안정화되어, 일시적인 오검출을 방지합니다.

## 🐛 문제 해결

### ROS2 토픽이 보이지 않는 경우
- ROS2 환경이 올바르게 설정되었는지 확인
- `ros2 topic list` 명령으로 토픽 확인
- 네트워크 설정 확인 (FromUrl, ToUrl)

### GPU가 인식되지 않는 경우
- `--device cpu` 옵션으로 CPU 모드 사용
- CUDA 설치 및 PyTorch CUDA 버전 확인
