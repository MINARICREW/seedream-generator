import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import urllib.request
from datetime import datetime

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

class FaceCropper:
    def __init__(self, margin=0.5, target_size=1024, debug=False, confidence_threshold=0.7, crop_method="balanced"):
        """
        정사각형 크롭을 위한 설정
        
        Args:
            margin: 전체 마진 비율 (기본 0.5 = 50%)
            target_size: 출력 이미지 크기 (기본 1024x1024)
            debug: 디버그 모드 (검출 영역 시각화)
            confidence_threshold: 얼굴 검출 신뢰도 임계값 (기본 0.7)
            crop_method: 크롭 방식 ("balanced", "balanced_v2", "adaptive")
        """
        self.margin = margin
        self.target_size = target_size
        self.debug = debug
        self.confidence_threshold = confidence_threshold
        self.crop_method = crop_method
        
        # DNN 모델 파일 경로
        self.model_dir = Path.home() / ".face_cropper" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.prototxt_path = self.model_dir / "deploy.prototxt"
        self.model_path = self.model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        
        # 모델 다운로드 및 초기화
        self._download_models()
        self._init_detector()
        
        print("✅ DNN 기반 얼굴 검출기를 사용합니다.")
        print(f"📐 설정값: 크롭방식={crop_method}, 마진={margin*100}%, 신뢰도 임계값={confidence_threshold}")
        if debug:
            print("🔍 디버그 모드 활성화 - 검출 영역을 시각화합니다.")
    
    def _download_models(self):
        """DNN 모델 파일 다운로드"""
        # Prototxt 파일
        if not self.prototxt_path.exists():
            print("📥 DNN 모델 설정 파일 다운로드 중...")
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            urllib.request.urlretrieve(prototxt_url, self.prototxt_path)
            print("✅ Prototxt 다운로드 완료")
        
        # Caffemodel 파일
        if not self.model_path.exists():
            print("📥 DNN 모델 파일 다운로드 중 (약 5.4MB)...")
            model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            urllib.request.urlretrieve(model_url, self.model_path)
            print("✅ 모델 다운로드 완료")
    
    def _init_detector(self):
        """DNN 검출기 초기화"""
        try:
            self.detector = cv2.dnn.readNetFromCaffe(
                str(self.prototxt_path), 
                str(self.model_path)
            )
            
            # GPU 사용 가능 여부 확인 및 설정
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("🚀 GPU 가속 활성화됨")
            else:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("💻 CPU 모드로 실행")
                
        except Exception as e:
            print(f"❌ DNN 모델 로드 실패: {e}")
            print("Haar Cascade로 대체합니다.")
            self.detector = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_face_dnn(self, image):
        """DNN으로 얼굴 검출"""
        if self.detector is None:
            return self.detect_face_haar(image)
        
        h, w = image.shape[:2]
        
        # 이미지 전처리
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        print(f"    🔍 DNN으로 얼굴 검출 중...")
        
        # 검출 수행
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        best_face = None
        best_confidence = 0
        
        # 검출 결과 처리
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # 바운딩 박스 좌표 추출
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # 유효성 검사
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # 너무 작은 검출 제외
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_face = [x1, y1, x2 - x1, y2 - y1]
        
        if best_face:
            print(f"    ✅ 얼굴 검출 성공! (신뢰도: {best_confidence:.3f})")
            return {'box': best_face, 'confidence': best_confidence}
        else:
            print(f"    ❌ 얼굴을 찾을 수 없습니다.")
            return None
    
    def detect_face_haar(self, image):
        """Haar Cascade로 얼굴 검출 (폴백)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴 선택
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        return {'box': [x, y, w, h], 'confidence': 1.0}
    
    def draw_debug_rectangles(self, image, face_bbox, crop_bbox, confidence=None):
        """디버그용 사각형 그리기"""
        debug_image = image.copy()
        
        # 얼굴 검출 영역 (빨간색)
        x, y, w, h = face_bbox
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        if confidence:
            label = f"Face ({confidence:.2f})"
        else:
            label = "Face Detection"
        
        cv2.putText(debug_image, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # 최종 크롭 영역 (초록색) - 좌표를 정수로 변환
        x1, y1, x2, y2 = [int(coord) for coord in crop_bbox]
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(debug_image, "Final Crop", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 범례 추가
        legend_y = 50
        cv2.putText(debug_image, "Red: Face Detection (DNN)", (50, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(debug_image, "Green: Final Crop Area", (50, legend_y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return debug_image
    
    def calculate_balanced_crop(self, face_center_x, face_center_y, face_size, img_width, img_height):
        """이미지 경계를 고려한 균형잡힌 크롭 영역 계산 (기본)"""
        # 원하는 정사각형 크기 (얼굴 크기 + 마진)
        desired_size = int(face_size * (1 + 2 * self.margin))
        
        # 이미지 내에서 가능한 최대 정사각형 크기
        max_possible_size = min(img_width, img_height, desired_size)
        
        # 얼굴이 크롭 영역 중앙에 오도록 초기 위치 설정
        half_size = max_possible_size // 2
        x1 = face_center_x - half_size
        y1 = face_center_y - half_size
        x2 = x1 + max_possible_size
        y2 = y1 + max_possible_size
        
        # 경계를 벗어나는 경우 조정
        if x1 < 0:
            x1 = 0
            x2 = max_possible_size
        elif x2 > img_width:
            x2 = img_width
            x1 = img_width - max_possible_size
            
        if y1 < 0:
            y1 = 0
            y2 = max_possible_size
        elif y2 > img_height:
            y2 = img_height
            y1 = img_height - max_possible_size
        
        # 최종 경계 체크
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        # 정사각형 보장
        final_size = min(x2 - x1, y2 - y1)
        
        # 얼굴이 최대한 중앙에 오도록 재조정
        if x2 - x1 > final_size:
            excess = (x2 - x1 - final_size) // 2
            x1 += excess
            x2 = x1 + final_size
        if y2 - y1 > final_size:
            excess = (y2 - y1 - final_size) // 2
            y1 += excess
            y2 = y1 + final_size
            
        return x1, y1, x2, y2
    
    def calculate_balanced_crop_v2(self, face_center_x, face_center_y, face_size, img_width, img_height, bbox):
        """더 정밀한 여백 계산을 포함한 크롭 (v2)"""
        x, y, w, h = bbox
        
        # 각 방향별 여유 공간 계산
        left_space = x
        right_space = img_width - (x + w)
        top_space = y
        bottom_space = img_height - (y + h)
        
        print(f"    여백: 좌={left_space}, 우={right_space}, 상={top_space}, 하={bottom_space}")
        
        # 원하는 마진 (픽셀)
        desired_margin_px = face_size * self.margin
        
        # 가능한 마진 계산
        possible_left = min(desired_margin_px, left_space)
        possible_right = min(desired_margin_px, right_space)
        possible_top = min(desired_margin_px, top_space)
        possible_bottom = min(desired_margin_px, bottom_space)
        
        # 정사각형 유지를 위해 최소 마진 선택
        min_horizontal = min(possible_left, possible_right)
        min_vertical = min(possible_top, possible_bottom)
        final_margin = min(min_horizontal, min_vertical)
        
        # 최소 마진 보장 (얼굴 크기의 10%)
        min_margin = face_size * 0.1
        if final_margin < min_margin:
            print(f"    ⚠️ 마진 부족: {final_margin/face_size:.1%} < {self.margin:.1%}")
            final_margin = min_margin
        
        # 실제 적용된 마진 비율 출력
        actual_margin_ratio = final_margin / face_size
        print(f"    📏 마진 조정: 목표 {self.margin:.1%} → 실제 {actual_margin_ratio:.1%}")
        
        # 크롭 영역 계산 - 정수로 변환
        crop_size = int(face_size + 2 * final_margin)
        half_size = crop_size // 2
        
        x1 = int(face_center_x - half_size)
        y1 = int(face_center_y - half_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # 경계 체크
        if x1 < 0:
            x1 = 0
            x2 = crop_size
        elif x2 > img_width:
            x2 = img_width
            x1 = img_width - crop_size
            
        if y1 < 0:
            y1 = 0
            y2 = crop_size
        elif y2 > img_height:
            y2 = img_height
            y1 = img_height - crop_size
        
        # 최종 경계 체크
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        # 정사각형이 아닌 경우 조정
        width = x2 - x1
        height = y2 - y1
        if width != height:
            final_size = min(width, height)
            # 중앙에 맞춰서 조정
            if width > final_size:
                diff = (width - final_size) // 2
                x1 += diff
                x2 = x1 + final_size
            if height > final_size:
                diff = (height - final_size) // 2
                y1 += diff
                y2 = y1 + final_size
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def calculate_adaptive_crop(self, face_bbox, img_width, img_height):
        """동적 마진 조정을 통한 적응형 크롭"""
        x, y, w, h = face_bbox
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        face_size = max(w, h)
        
        # 이상적인 크롭 크기 (얼굴의 2.5배)
        ideal_size = int(face_size * (1 + 2 * self.margin))
        
        # 실제 가능한 크기
        max_width = img_width
        max_height = img_height
        
        # 종횡비를 유지하면서 크기 조정
        if ideal_size > min(max_width, max_height):
            # 이미지가 작으면 가능한 최대 정사각형
            crop_size = min(max_width, max_height)
            
            # 얼굴이 크롭 영역 내에서 적절한 위치에 오도록
            available_margin = (crop_size - face_size) / 2
            actual_margin_ratio = available_margin / face_size
            
            print(f"    🔄 동적 마진 적용: 목표 {self.margin:.1%} → 실제 {actual_margin_ratio:.1%}")
            
            # 경고 메시지
            if actual_margin_ratio < self.margin * 0.5:
                print(f"    ⚠️ 마진이 목표의 50% 미만입니다. 더 넓은 여백으로 촬영하세요.")
        else:
            crop_size = ideal_size
            print(f"    ✅ 충분한 여백 확보")
        
        # 크롭 영역 계산
        half_size = crop_size // 2
        x1 = face_center_x - half_size
        y1 = face_center_y - half_size
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # 경계 조정
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > img_width:
            x1 -= (x2 - img_width)
            x2 = img_width
        if y2 > img_height:
            y1 -= (y2 - img_height)
            y2 = img_height
        
        # 최종 경계 체크
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        # 정사각형 보장
        final_size = min(x2 - x1, y2 - y1)
        if x2 - x1 > final_size:
            diff = (x2 - x1 - final_size) // 2
            x1 += diff
            x2 = x1 + final_size
        if y2 - y1 > final_size:
            diff = (y2 - y1 - final_size) // 2
            y1 += diff
            y2 = y1 + final_size
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def resize_to_target_size(self, image):
        """이미지를 1024x1024로 리사이징 (패딩 또는 다운스케일링, 확대 없음)"""
        h, w = image.shape[:2]
        
        print(f"    원본 크기: {w}x{h}")
        
        # 이미 target_size와 같으면 그대로 반환
        if h == self.target_size and w == self.target_size:
            print(f"    이미 {self.target_size}x{self.target_size} 크기입니다.")
            return image
        
        # 큰 이미지인 경우: 다운스케일링
        if h > self.target_size or w > self.target_size:
            print(f"    큰 이미지 -> 다운스케일링")
            
            # 비율 유지하면서 target_size에 맞게 축소
            scale = min(self.target_size / w, self.target_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 리사이즈
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"    축소된 크기: {new_w}x{new_h}")
            
            # 정확히 target_size가 아니면 패딩 추가
            if new_w != self.target_size or new_h != self.target_size:
                # 패딩 계산
                pad_w = (self.target_size - new_w) // 2
                pad_h = (self.target_size - new_h) // 2
                pad_w_extra = (self.target_size - new_w) % 2
                pad_h_extra = (self.target_size - new_h) % 2
                
                # 흰색 패딩 추가
                final_image = cv2.copyMakeBorder(
                    resized,
                    pad_h, pad_h + pad_h_extra,
                    pad_w, pad_w + pad_w_extra,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]  # 흰색
                )
                print(f"    패딩 추가: {pad_w}+{pad_w_extra}(좌우), {pad_h}+{pad_h_extra}(상하)")
            else:
                final_image = resized
                
        # 작은 이미지인 경우: 원본 크기 유지하고 패딩만
        else:
            print(f"    작은 이미지 -> 원본 크기 유지 + 패딩")
            
            # 확대 없이 원본 크기 그대로 사용
            resized = image
            new_w, new_h = w, h
            print(f"    원본 크기 유지: {new_w}x{new_h}")
            
            # 패딩 계산
            pad_w = (self.target_size - new_w) // 2
            pad_h = (self.target_size - new_h) // 2
            pad_w_extra = (self.target_size - new_w) % 2
            pad_h_extra = (self.target_size - new_h) % 2
            
            # 흰색 패딩 추가
            final_image = cv2.copyMakeBorder(
                resized,
                pad_h, pad_h + pad_h_extra,
                pad_w, pad_w + pad_w_extra,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]  # 흰색
            )
            print(f"    패딩 추가: {pad_w}+{pad_w_extra}(좌우), {pad_h}+{pad_h_extra}(상하)")
        
        print(f"    최종 크기: {final_image.shape[1]}x{final_image.shape[0]}")
        return final_image

    def load_image(self, image_path):
        """이미지 로드 (HEIC 포함)"""
        image_path = str(image_path)
        
        # HEIC 파일 처리
        if image_path.lower().endswith(('.heic', '.heif')):
            if not HEIF_AVAILABLE:
                print(f"HEIC 지원을 위해 pillow-heif를 설치해주세요: pip install pillow-heif")
                return None
                
            try:
                # PIL로 HEIC 읽기
                pil_image = Image.open(image_path)
                # RGB로 변환 (RGBA일 수 있음)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                # numpy 배열로 변환
                image_array = np.array(pil_image)
                # OpenCV 형식으로 변환 (RGB -> BGR)
                image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                return image
            except Exception as e:
                print(f"HEIC 파일 읽기 실패 {image_path}: {e}")
                return None
        else:
            # 일반 이미지 파일
            return cv2.imread(image_path)
    
    def save_image(self, image, output_path):
        """이미지 저장 (HEIC -> JPG 변환)"""
        output_path = Path(output_path)
        
        # HEIC 입력인 경우 JPG로 저장
        if output_path.suffix.lower() in ['.heic', '.heif']:
            output_path = output_path.with_suffix('.jpg')
            
        cv2.imwrite(str(output_path), image)
        return output_path
    
    def crop_face_from_image(self, image_path, output_path=None, debug_output_path=None):
        """단일 이미지에서 얼굴 크롭 - 정사각형으로 크롭"""
        try:
            # 이미지 읽기 (HEIC 포함)
            image = self.load_image(image_path)
            if image is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                return None
                
            print(f"얼굴 검출 시도 중...")
            
            # DNN으로 얼굴 검출
            detection = self.detect_face_dnn(image)
            
            if detection is None:
                print(f"얼굴 검출 실패: {image_path}")
                print(f"원본 이미지를 리사이징합니다...")
                
                # 원본 이미지를 정사각형으로 크롭 후 리사이징
                h, w, _ = image.shape
                size = min(h, w)
                center_x, center_y = w // 2, h // 2
                
                x1 = center_x - size // 2
                y1 = center_y - size // 2
                x2 = x1 + size
                y2 = y1 + size
                
                # 경계 체크
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                cropped = image[y1:y2, x1:x2]
                resized_image = self.resize_to_target_size(cropped)
                
                # 출력 경로 설정
                if output_path is None:
                    input_path = Path(image_path)
                    output_path = input_path.parent / f"original_{input_path.stem}.jpg"
                
                # 리사이징된 이미지 저장
                output_path = self.save_image(resized_image, output_path)
                print(f"원본 리사이징 완료: {output_path}")
                return "original_resized"
                
            # 바운딩 박스 추출
            h, w, _ = image.shape
            bbox = detection['box']  # [x, y, width, height]
            confidence = detection.get('confidence', 1.0)
            
            x = bbox[0]
            y = bbox[1]
            width = bbox[2]
            height = bbox[3]
            
            # 음수 좌표 보정
            x = max(0, x)
            y = max(0, y)
            
            print(f"    검출된 얼굴 위치: x={x}, y={y}, w={width}, h={height}")
            face_bbox = [x, y, width, height]
            
            # 얼굴 중심점 계산
            face_center_x = x + width // 2
            face_center_y = y + height // 2
            
            # 얼굴 크기 기준으로 정사각형 크기 결정
            face_size = max(width, height)
            
            # 선택된 크롭 방식에 따라 분기
            if self.crop_method == "balanced":
                x1, y1, x2, y2 = self.calculate_balanced_crop(
                    face_center_x, face_center_y, face_size, w, h
                )
            elif self.crop_method == "balanced_v2":
                x1, y1, x2, y2 = self.calculate_balanced_crop_v2(
                    face_center_x, face_center_y, face_size, w, h, [x, y, width, height]
                )
            elif self.crop_method == "adaptive":
                x1, y1, x2, y2 = self.calculate_adaptive_crop(
                    [x, y, width, height], w, h
                )
            else:
                # 기본값
                x1, y1, x2, y2 = self.calculate_balanced_crop(
                    face_center_x, face_center_y, face_size, w, h
                )
            
            # 좌표를 정수로 변환
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            print(f"    크롭 방식: {self.crop_method}")
            print(f"    얼굴 중심: ({face_center_x}, {face_center_y})")
            print(f"    얼굴 크기: {face_size}")
            print(f"    최종 크롭 영역: ({x1}, {y1}) ~ ({x2}, {y2})")
            print(f"    크롭 크기: {x2-x1} x {y2-y1}")
            crop_bbox = [x1, y1, x2, y2]
            
            # 디버그 모드일 경우 검출 영역 시각화
            if self.debug and debug_output_path:
                debug_image = self.draw_debug_rectangles(image, face_bbox, crop_bbox, confidence)
                debug_path = self.save_image(debug_image, debug_output_path)
                print(f"    📸 디버그 이미지 저장: {debug_path}")
            
            # 크롭 영역이 유효한지 확인
            if x2 <= x1 or y2 <= y1:
                print(f"    ❌ 유효하지 않은 크롭 영역입니다. 원본 이미지를 리사이징합니다.")
                resized_image = self.resize_to_target_size(image)
                
                if output_path is None:
                    input_path = Path(image_path)
                    output_path = input_path.parent / f"original_{input_path.stem}.jpg"
                
                output_path = self.save_image(resized_image, output_path)
                print(f"원본 리사이징 완료: {output_path}")
                return "original_resized"
            
            # 얼굴 크롭
            cropped_face = image[y1:y2, x1:x2]
            print(f"    크롭 완료 -> 리사이징 시작")
            
            # 크롭된 이미지를 1024x1024로 리사이징
            resized_face = self.resize_to_target_size(cropped_face)
            
            # 출력 경로 설정
            if output_path is None:
                input_path = Path(image_path)
                output_path = input_path.parent / f"cropped_{input_path.stem}.jpg"
            
            # 최종 이미지 저장
            final_output_path = self.save_image(resized_face, output_path)
            print(f"크롭 + 리사이징 완료: {final_output_path}")
            
            return resized_face
            
        except Exception as e:
            print(f"    ❌ 처리 중 오류 발생: {e}")
            print(f"    원본 이미지를 리사이징합니다...")
            
            try:
                # 오류 발생 시 원본 이미지 리사이징 시도
                image = self.load_image(image_path)
                if image is not None:
                    resized_image = self.resize_to_target_size(image)
                    
                    if output_path is None:
                        input_path = Path(image_path)
                        output_path = input_path.parent / f"error_{input_path.stem}.jpg"
                    
                    output_path = self.save_image(resized_image, output_path)
                    print(f"    원본 리사이징 완료: {output_path}")
                    return "error_resized"
            except:
                pass
            
            return None
    
    def batch_crop_faces(self, input_folder, output_folder=None):
        """폴더 내 모든 이미지에서 얼굴 크롭"""
        input_path = Path(input_folder)
        
        if output_folder is None:
            output_path = input_path.parent / "cropped_faces"
        else:
            output_path = Path(output_folder)
            
        # 입력 폴더명 추출
        person_folder_name = input_path.name
        
        # 현재 시간을 YYYYMMDD_HHMMSS 형식으로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 폴더명 + 설정 정보 형식으로 생성
        session_folder = output_path / f"{timestamp}_{person_folder_name}_margin{self.margin}_{self.crop_method}_dnn"
        
        # 디버그 폴더 생성
        debug_folder = None
        if self.debug:
            debug_folder = session_folder / "debug"
            debug_folder.mkdir(parents=True, exist_ok=True)
        
        # 폴더 생성
        session_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"처리 대상: {person_folder_name}")
        print(f"결과 저장 폴더: {session_folder}")
        if self.debug:
            print(f"디버그 이미지 폴더: {debug_folder}")
        print(f"목표 해상도: {self.target_size}x{self.target_size}")
        print(f"크롭 방식: {self.crop_method}")
        print(f"마진: {self.margin*100}%")
        print(f"얼굴 검출: DNN (SSD MobileNet)")
        
        # 지원하는 이미지 확장자 (HEIC 포함)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif']
        
        # 폴더 내 이미지 파일 찾기
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"이미지 파일을 찾을 수 없습니다: {input_folder}")
            return
            
        print(f"총 {len(image_files)}개의 이미지를 처리합니다...")
        
        success_count = 0
        original_count = 0
        error_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 처리 중: {image_file.name}")
            
            output_file = session_folder / image_file.name
            debug_file = None
            if self.debug:
                debug_file = debug_folder / f"debug_{image_file.stem}.jpg"
            
            result = self.crop_face_from_image(image_file, output_file, debug_file)
            
            if result is not None:
                if isinstance(result, str):
                    if result == "original_resized":
                        original_count += 1
                    elif result == "error_resized":
                        error_count += 1
                else:
                    success_count += 1
            else:
                error_count += 1
                    
        print(f"\n" + "="*50)
        print(f"처리 완료! ({person_folder_name})")
        print(f"얼굴 크롭 성공: {success_count}개")
        print(f"원본 리사이징: {original_count}개")
        if error_count > 0:
            print(f"오류 처리: {error_count}개")
        print(f"전체 처리: {success_count + original_count + error_count}/{len(image_files)}개")
        print(f"모든 결과물은 {self.target_size}x{self.target_size} 해상도입니다")
        print(f"결과 저장 위치: {session_folder}")
        if self.debug:
            print(f"디버그 이미지: {debug_folder}")
        print(f"="*50)
        
        return session_folder

def get_available_folders(input_base_folder="input_images"):
    """input_images 폴더 내 사용 가능한 인물 폴더들을 보여줌"""
    base_path = Path(input_base_folder)
    
    if not base_path.exists():
        print(f"'{input_base_folder}' 폴더를 찾을 수 없습니다.")
        return []
    
    # 하위 폴더들만 찾기
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"'{input_base_folder}' 폴더에 하위 폴더가 없습니다.")
        return []
    
    print(f"\n사용 가능한 인물 폴더들:")
    print("-" * 30)
    for i, folder in enumerate(folders, 1):
        # 각 폴더의 이미지 개수 체크 (HEIC 포함)
        image_count = len([f for f in folder.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif']])
        print(f"{i}. {folder.name} ({image_count}개 이미지)")
    
    return folders

def select_folders_interactive(folders):
    """사용자가 폴더를 선택할 수 있게 해주는 함수"""
    if not folders:
        return []
    
    print(f"\n📁 폴더 선택 옵션:")
    print(f"1. 개별 선택: 숫자를 콤마로 구분 (예: 1,3,5)")
    print(f"2. 범위 선택: 하이픈 사용 (예: 1-3)")
    print(f"3. 전체 선택: 'all' 입력")
    print(f"4. 취소: 'q' 또는 'quit' 입력")
    
    while True:
        try:
            print(f"\n선택하세요: ", end="")
            user_input = input().strip().lower()
            
            if user_input in ['q', 'quit']:
                print("선택이 취소되었습니다.")
                return []
            
            if user_input == 'all':
                print(f"전체 {len(folders)}개 폴더가 선택되었습니다.")
                return folders
            
            selected_folders = []
            
            # 콤마로 구분된 입력 처리
            for part in user_input.split(','):
                part = part.strip()
                
                # 범위 선택 (예: 1-3)
                if '-' in part:
                    try:
                        start, end = map(int, part.split('-'))
                        for i in range(start, end + 1):
                            if 1 <= i <= len(folders):
                                if folders[i-1] not in selected_folders:
                                    selected_folders.append(folders[i-1])
                    except ValueError:
                        print(f"잘못된 범위 형식: {part}")
                        continue
                
                # 개별 선택
                else:
                    try:
                        choice = int(part)
                        if 1 <= choice <= len(folders):
                            if folders[choice-1] not in selected_folders:
                                selected_folders.append(folders[choice-1])
                        else:
                            print(f"범위를 벗어난 숫자: {choice}")
                    except ValueError:
                        print(f"잘못된 입력: {part}")
                        continue
            
            if selected_folders:
                print(f"\n선택된 폴더들:")
                for i, folder in enumerate(selected_folders, 1):
                    print(f"  {i}. {folder.name}")
                
                print(f"\n이대로 진행하시겠습니까? (y/n): ", end="")
                confirm = input().strip().lower()
                if confirm in ['y', 'yes', '']:
                    return selected_folders
                else:
                    print("다시 선택해주세요.")
            else:
                print("선택된 폴더가 없습니다. 다시 입력해주세요.")
                
        except KeyboardInterrupt:
            print("\n작업이 취소되었습니다.")
            return []

def get_crop_method():
    """사용자로부터 크롭 방식을 선택받는 함수"""
    print("\n🎯 크롭 방식을 선택하세요:")
    print("1. Balanced Crop (기본) - 정사각형 유지, 균형잡힌 크롭")
    print("2. Balanced Crop v2 - 더 정밀한 여백 계산")
    print("3. Adaptive Crop - 동적 마진 조정")
    
    while True:
        choice = input("\n선택 (1-3, 기본값 1): ").strip()
        
        if choice == "" or choice == "1":
            return "balanced"
        elif choice == "2":
            return "balanced_v2"
        elif choice == "3":
            return "adaptive"
        else:
            print("1, 2, 3 중에서 선택해주세요.")

def get_cropper_settings():
    """사용자로부터 크롭 설정을 입력받는 함수"""
    print("\n🎨 크롭 설정을 입력하세요 (Enter로 기본값 사용):")
    
    # 크롭 방식 선택
    crop_method = get_crop_method()
    
    # 마진 설정
    while True:
        margin_input = input("마진 (0.1~1, 기본값 0.5): ").strip()
        if margin_input == "":
            margin = 0.5
            break
        try:
            margin = float(margin_input)
            if 0.1 <= margin <= 1.0:
                break
            else:
                print("0.1에서 1.0 사이의 값을 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")
    
    # 신뢰도 임계값 설정
    while True:
        conf_input = input("얼굴 검출 신뢰도 (0.5~0.9, 기본값 0.7): ").strip()
        if conf_input == "":
            confidence = 0.7
            break
        try:
            confidence = float(conf_input)
            if 0.5 <= confidence <= 0.9:
                break
            else:
                print("0.5에서 0.9 사이의 값을 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")
    
    # 디버그 모드 설정
    debug_input = input("디버그 모드 활성화? (검출 영역 시각화) (y/N): ").strip().lower()
    debug = debug_input in ['y', 'yes']
    
    print(f"\n설정값: 크롭방식={crop_method}, 마진={margin*100}%, 신뢰도={confidence}, 디버그={'ON' if debug else 'OFF'}")
    return crop_method, margin, confidence, debug

def main():
    """메인 실행 함수"""
    print("="*70)
    print("   🎭 얼굴 크롭 + 1024x1024 리사이징 스크립트 (DNN 버전)")
    print("="*70)
    print()
    
    # 사용 가능한 폴더 확인
    folders = get_available_folders()
    if not folders:
        print("처리할 폴더가 없습니다.")
        return
    
    # 사용자가 폴더 선택
    selected_folders = select_folders_interactive(folders)
    if not selected_folders:
        print("선택된 폴더가 없습니다.")
        return
    
    # 크롭 설정 입력받기
    crop_method, margin, confidence, debug = get_cropper_settings()
    
    print(f"\n🚀 얼굴 크롭 + 리사이징을 시작합니다...")
    print(f"처리할 폴더 수: {len(selected_folders)}개")
    print(f"목표 해상도: 1024x1024")
    print(f"처리 방식:")
    print(f"  - 작은 이미지: 원본 크기 유지 + 흰색 패딩으로 1024x1024 맞춤")
    print(f"  - 큰 이미지: 다운스케일링으로 1024x1024 맞춤")
    print(f"  ⭐ 확대 없음 - 원본 화질 보존!")
    print(f"  🎯 정사각형 크롭으로 비율 왜곡 없음!")
    print(f"  🔍 DNN 기반: 높은 정확도의 얼굴 검출!")
    
    if crop_method == "balanced":
        print(f"  ⚖️  균형잡힌 크롭: 얼굴이 가장자리에 있어도 적절한 마진 유지!")
    elif crop_method == "balanced_v2":
        print(f"  📏 정밀한 여백 계산: 각 방향별 여백을 분석하여 최적화!")
    elif crop_method == "adaptive":
        print(f"  🔄 동적 마진 조정: 이미지 크기에 따라 자동으로 마진 조절!")
    
    if debug:
        print(f"  📸 디버그 모드: 검출 영역 시각화!")
    print()
    
    # FaceCropper
    cropper = FaceCropper(
        margin=margin,
        target_size=1024,
        debug=debug,
        confidence_threshold=confidence,
        crop_method=crop_method
    )
    
    # 선택된 폴더들을 순차 처리
    output_folder = "cropped_images"
    total_processed = 0
    
    for i, folder in enumerate(selected_folders, 1):
        print(f"\n{'='*20} [{i}/{len(selected_folders)}] {'='*20}")
        print(f"현재 처리 중: {folder.name}")
        print(f"{'='*50}")
        
        result_folder = cropper.batch_crop_faces(str(folder), output_folder)
        if result_folder:
            total_processed += 1
        
        if i < len(selected_folders):
            print(f"\n⏳ 다음 폴더 처리까지 잠시 대기...")
            print("-" * 50)
    
    print(f"\n🎉 전체 작업 완료!")
    print(f"총 처리된 폴더: {total_processed}/{len(selected_folders)}개")
    print(f"모든 결과물은 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    # 필요한 패키지 설치 안내
    try:
        import cv2
        from PIL import Image
        import numpy as np
    except ImportError:
        print("필요한 패키지를 설치해주세요:")
        print("pip install opencv-python pillow numpy pillow-heif")
        exit(1)
    
    print("✅ DNN 기반 얼굴 검출 라이브러리가 준비되었습니다.")
    
    # HEIC 지원 체크
    if not HEIF_AVAILABLE:
        print("⚠️  HEIC 파일 지원을 위해 추가 패키지를 설치하는 것을 권장합니다:")
        print("pip install pillow-heif")
        print("(설치하지 않으면 HEIC 파일을 처리할 수 없습니다)\n")
    else:
        print("✅ HEIC 파일 지원이 활성화되었습니다.\n")
    
    main()
