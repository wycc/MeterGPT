"""
PaddleOCR 包裝器
提供統一的 OCR 介面，支援多種文字識別場景
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import time
from pathlib import Path

from ..models.messages import OCRResult, BoundingBox, ROI
from ..utils.logger import get_logger, log_execution_time


class PaddleOCRWrapper:
    """PaddleOCR 包裝器"""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en', 
                 use_gpu: bool = False, confidence_threshold: float = 0.7,
                 det_model_dir: Optional[str] = None,
                 rec_model_dir: Optional[str] = None,
                 cls_model_dir: Optional[str] = None):
        """
        初始化 PaddleOCR 包裝器
        
        Args:
            use_angle_cls: 是否使用角度分類器
            lang: 語言設定
            use_gpu: 是否使用 GPU
            confidence_threshold: 信心度閾值
            det_model_dir: 檢測模型路徑
            rec_model_dir: 識別模型路徑
            cls_model_dir: 分類模型路徑
        """
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        
        self.ocr_engine = None
        self.logger = get_logger("PaddleOCRWrapper")
        
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """初始化 OCR 引擎"""
        try:
            from paddleocr import PaddleOCR
            
            # 建立 OCR 引擎參數
            ocr_params = {
                'use_angle_cls': self.use_angle_cls,
                'lang': self.lang,
                'use_gpu': self.use_gpu,
                'show_log': False
            }
            
            # 添加自定義模型路徑
            if self.det_model_dir:
                ocr_params['det_model_dir'] = self.det_model_dir
            if self.rec_model_dir:
                ocr_params['rec_model_dir'] = self.rec_model_dir
            if self.cls_model_dir:
                ocr_params['cls_model_dir'] = self.cls_model_dir
            
            self.ocr_engine = PaddleOCR(**ocr_params)
            self.logger.info("PaddleOCR 引擎初始化成功")
            
        except ImportError:
            self.logger.error("PaddleOCR 未安裝，請執行: pip install paddlepaddle paddleocr")
            raise
        except Exception as e:
            self.logger.error(f"PaddleOCR 初始化失敗: {e}")
            raise
    
    @log_execution_time()
    def recognize_text(self, image: np.ndarray, roi_id: str = "default", 
                      frame_id: str = "unknown") -> OCRResult:
        """
        識別文字
        
        Args:
            image: 輸入影像
            roi_id: ROI 識別碼
            frame_id: 影像幀識別碼
            
        Returns:
            OCRResult: OCR 識別結果
        """
        start_time = time.time()
        
        try:
            if self.ocr_engine is None:
                raise RuntimeError("OCR 引擎未初始化")
            
            # 執行 OCR 識別
            results = self.ocr_engine.ocr(image, cls=self.use_angle_cls)
            
            # 解析結果
            recognized_text = ""
            bounding_boxes = []
            total_confidence = 0.0
            valid_results = 0
            
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        bbox_coords = line[0]
                        text_info = line[1]
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            # 過濾低信心度結果
                            if confidence >= self.confidence_threshold:
                                recognized_text += text + " "
                                
                                # 建立邊界框
                                bbox_array = np.array(bbox_coords)
                                x_min, y_min = bbox_array.min(axis=0)
                                x_max, y_max = bbox_array.max(axis=0)
                                
                                bbox = BoundingBox(
                                    x=float(x_min),
                                    y=float(y_min),
                                    width=float(x_max - x_min),
                                    height=float(y_max - y_min),
                                    confidence=float(confidence)
                                )
                                bounding_boxes.append(bbox)
                                
                                total_confidence += confidence
                                valid_results += 1
            
            # 計算平均信心度
            avg_confidence = total_confidence / valid_results if valid_results > 0 else 0.0
            
            # 清理識別文字
            recognized_text = recognized_text.strip()
            
            processing_time = time.time() - start_time
            
            ocr_result = OCRResult(
                frame_id=frame_id,
                roi_id=roi_id,
                recognized_text=recognized_text,
                confidence=avg_confidence,
                bounding_boxes=bounding_boxes,
                processing_time=processing_time,
                ocr_engine="PaddleOCR"
            )
            
            self.logger.info(
                f"OCR 識別完成: '{recognized_text}'",
                frame_id=frame_id,
                roi_id=roi_id,
                confidence=avg_confidence
            )
            
            return ocr_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"OCR 識別失敗: {e}", frame_id=frame_id, roi_id=roi_id)
            
            # 返回空結果
            return OCRResult(
                frame_id=frame_id,
                roi_id=roi_id,
                recognized_text="",
                confidence=0.0,
                bounding_boxes=[],
                processing_time=processing_time,
                ocr_engine="PaddleOCR"
            )
    
    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        影像前處理
        
        Args:
            image: 原始影像
            enhance: 是否進行增強處理
            
        Returns:
            np.ndarray: 處理後的影像
        """
        try:
            processed_image = image.copy()
            
            if enhance:
                # 轉換為灰階
                if len(processed_image.shape) == 3:
                    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = processed_image
                
                # 直方圖均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # 降噪
                denoised = cv2.fastNlMeansDenoising(enhanced)
                
                # 銳化
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(denoised, -1, kernel)
                
                processed_image = sharpened
            
            return processed_image
            
        except Exception as e:
            self.logger.error(f"影像前處理失敗: {e}")
            return image


class SevenSegmentOCR:
    """七段顯示器 OCR (基於規則的方法)"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        """
        初始化七段顯示器 OCR
        
        Args:
            confidence_threshold: 信心度閾值
        """
        self.confidence_threshold = confidence_threshold
        self.logger = get_logger("SevenSegmentOCR")
        
        # 七段顯示器數字模板
        self.digit_templates = self._create_digit_templates()
    
    def _create_digit_templates(self) -> Dict[int, np.ndarray]:
        """建立數字模板"""
        templates = {}
        
        # 定義七段顯示器的段位
        # 每個數字用7位二進制表示，對應7個段位
        digit_patterns = {
            0: [1, 1, 1, 1, 1, 1, 0],  # 0
            1: [0, 1, 1, 0, 0, 0, 0],  # 1
            2: [1, 1, 0, 1, 1, 0, 1],  # 2
            3: [1, 1, 1, 1, 0, 0, 1],  # 3
            4: [0, 1, 1, 0, 0, 1, 1],  # 4
            5: [1, 0, 1, 1, 0, 1, 1],  # 5
            6: [1, 0, 1, 1, 1, 1, 1],  # 6
            7: [1, 1, 1, 0, 0, 0, 0],  # 7
            8: [1, 1, 1, 1, 1, 1, 1],  # 8
            9: [1, 1, 1, 1, 0, 1, 1],  # 9
        }
        
        # 這裡可以根據實際需求建立更詳細的模板
        for digit, pattern in digit_patterns.items():
            templates[digit] = np.array(pattern)
        
        return templates
    
    @log_execution_time()
    def recognize_seven_segment(self, image: np.ndarray, roi_id: str = "default",
                               frame_id: str = "unknown") -> OCRResult:
        """
        識別七段顯示器數字
        
        Args:
            image: 輸入影像
            roi_id: ROI 識別碼
            frame_id: 影像幀識別碼
            
        Returns:
            OCRResult: OCR 識別結果
        """
        start_time = time.time()
        
        try:
            # 影像前處理
            processed_image = self._preprocess_seven_segment(image)
            
            # 分割數字
            digit_regions = self._segment_digits(processed_image)
            
            # 識別每個數字
            recognized_digits = []
            bounding_boxes = []
            confidences = []
            
            for i, region in enumerate(digit_regions):
                digit, confidence, bbox = self._recognize_single_digit(region, i)
                if digit is not None:
                    recognized_digits.append(str(digit))
                    confidences.append(confidence)
                    bounding_boxes.append(bbox)
            
            # 組合結果
            recognized_text = "".join(recognized_digits)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            ocr_result = OCRResult(
                frame_id=frame_id,
                roi_id=roi_id,
                recognized_text=recognized_text,
                confidence=avg_confidence,
                bounding_boxes=bounding_boxes,
                processing_time=processing_time,
                ocr_engine="SevenSegmentOCR"
            )
            
            self.logger.info(
                f"七段顯示器識別完成: '{recognized_text}'",
                frame_id=frame_id,
                roi_id=roi_id,
                confidence=avg_confidence
            )
            
            return ocr_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"七段顯示器識別失敗: {e}", frame_id=frame_id, roi_id=roi_id)
            
            return OCRResult(
                frame_id=frame_id,
                roi_id=roi_id,
                recognized_text="",
                confidence=0.0,
                bounding_boxes=[],
                processing_time=processing_time,
                ocr_engine="SevenSegmentOCR"
            )
    
    def _preprocess_seven_segment(self, image: np.ndarray) -> np.ndarray:
        """七段顯示器專用前處理"""
        # 轉換為灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形態學操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _segment_digits(self, image: np.ndarray) -> List[np.ndarray]:
        """分割數字區域"""
        # 尋找輪廓
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 過濾和排序輪廓
        digit_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 過濾小區域
                digit_contours.append(contour)
        
        # 按 x 座標排序
        digit_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
        
        # 提取數字區域
        digit_regions = []
        for contour in digit_contours:
            x, y, w, h = cv2.boundingRect(contour)
            digit_region = image[y:y+h, x:x+w]
            digit_regions.append(digit_region)
        
        return digit_regions
    
    def _recognize_single_digit(self, digit_image: np.ndarray, index: int) -> Tuple[Optional[int], float, BoundingBox]:
        """識別單個數字"""
        try:
            # 調整尺寸到標準大小
            resized = cv2.resize(digit_image, (28, 48))
            
            # 提取特徵 (簡化版本)
            features = self._extract_seven_segment_features(resized)
            
            # 與模板匹配
            best_digit = None
            best_confidence = 0.0
            
            for digit, template in self.digit_templates.items():
                similarity = self._calculate_similarity(features, template)
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_digit = digit
            
            # 建立邊界框
            bbox = BoundingBox(
                x=0.0, y=0.0,
                width=float(digit_image.shape[1]),
                height=float(digit_image.shape[0]),
                confidence=best_confidence
            )
            
            return best_digit, best_confidence, bbox
            
        except Exception as e:
            self.logger.error(f"單個數字識別失敗: {e}")
            return None, 0.0, BoundingBox(x=0, y=0, width=0, height=0, confidence=0)
    
    def _extract_seven_segment_features(self, image: np.ndarray) -> np.ndarray:
        """提取七段顯示器特徵"""
        # 簡化的特徵提取，實際應用中需要更精確的方法
        h, w = image.shape
        
        # 定義7個段位的區域
        segments = np.zeros(7)
        
        # 上橫段
        segments[0] = np.mean(image[0:h//6, w//4:3*w//4])
        # 右上豎段
        segments[1] = np.mean(image[0:h//2, 3*w//4:w])
        # 右下豎段
        segments[2] = np.mean(image[h//2:h, 3*w//4:w])
        # 下橫段
        segments[3] = np.mean(image[5*h//6:h, w//4:3*w//4])
        # 左下豎段
        segments[4] = np.mean(image[h//2:h, 0:w//4])
        # 左上豎段
        segments[5] = np.mean(image[0:h//2, 0:w//4])
        # 中橫段
        segments[6] = np.mean(image[2*h//5:3*h//5, w//4:3*w//4])
        
        # 二值化特徵
        return (segments > 128).astype(int)
    
    def _calculate_similarity(self, features: np.ndarray, template: np.ndarray) -> float:
        """計算特徵相似度"""
        return np.sum(features == template) / len(template)


class OCRManager:
    """OCR 管理器"""
    
    def __init__(self):
        """初始化 OCR 管理器"""
        self.paddle_ocr = None
        self.seven_segment_ocr = None
        self.logger = get_logger("OCRManager")
    
    def initialize_paddle_ocr(self, **kwargs) -> PaddleOCRWrapper:
        """初始化 PaddleOCR"""
        try:
            self.paddle_ocr = PaddleOCRWrapper(**kwargs)
            self.logger.info("PaddleOCR 初始化成功")
            return self.paddle_ocr
        except Exception as e:
            self.logger.error(f"PaddleOCR 初始化失敗: {e}")
            raise
    
    def initialize_seven_segment_ocr(self, **kwargs) -> SevenSegmentOCR:
        """初始化七段顯示器 OCR"""
        try:
            self.seven_segment_ocr = SevenSegmentOCR(**kwargs)
            self.logger.info("七段顯示器 OCR 初始化成功")
            return self.seven_segment_ocr
        except Exception as e:
            self.logger.error(f"七段顯示器 OCR 初始化失敗: {e}")
            raise
    
    def recognize_roi(self, roi: ROI, frame_id: str) -> OCRResult:
        """
        識別 ROI 中的文字
        
        Args:
            roi: 感興趣區域
            frame_id: 影像幀識別碼
            
        Returns:
            OCRResult: OCR 識別結果
        """
        try:
            # 解碼 ROI 影像資料
            roi_image = cv2.imdecode(
                np.frombuffer(roi.roi_data, np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            # 根據預期格式選擇 OCR 引擎
            if roi.expected_format.lower() in ['seven_segment', '7segment']:
                if self.seven_segment_ocr is None:
                    self.initialize_seven_segment_ocr()
                return self.seven_segment_ocr.recognize_seven_segment(
                    roi_image, roi.roi_id, frame_id
                )
            else:
                if self.paddle_ocr is None:
                    self.initialize_paddle_ocr()
                return self.paddle_ocr.recognize_text(
                    roi_image, roi.roi_id, frame_id
                )
                
        except Exception as e:
            self.logger.error(f"ROI 識別失敗: {e}", frame_id=frame_id, roi_id=roi.roi_id)
            return OCRResult(
                frame_id=frame_id,
                roi_id=roi.roi_id,
                recognized_text="",
                confidence=0.0,
                bounding_boxes=[],
                processing_time=0.0,
                ocr_engine="Unknown"
            )


# 全域 OCR 管理器實例
ocr_manager = OCRManager()