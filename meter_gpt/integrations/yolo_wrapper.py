"""
YOLO 模型包裝器
提供統一的 YOLO 模型介面，支援儀器偵測和角點偵測
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import torch
from pathlib import Path
import time

from ..models.messages import BoundingBox, CornerPoints, DetectionResult, InstrumentType
from ..utils.logger import get_logger, log_execution_time


class YOLOWrapper:
    """YOLO 模型包裝器基類"""
    
    def __init__(self, model_path: str, device: str = "cpu", 
                 confidence_threshold: float = 0.5, input_size: Tuple[int, int] = (640, 640)):
        """
        初始化 YOLO 包裝器
        
        Args:
            model_path: 模型檔案路徑
            device: 運算設備 (cpu/cuda)
            confidence_threshold: 信心度閾值
            input_size: 輸入尺寸
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.model = None
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        self._load_model()
    
    def _load_model(self):
        """載入模型"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"模型檔案不存在: {self.model_path}")
            
            # 嘗試載入 YOLOv8 模型
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.logger.info(f"成功載入 YOLOv8 模型: {self.model_path}")
            except ImportError:
                # 如果沒有 ultralytics，嘗試使用 torch.hub
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=self.model_path, device=self.device)
                self.logger.info(f"成功載入 YOLOv5 模型: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"載入模型失敗: {e}")
            raise
    
    @log_execution_time()
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        執行預測
        
        Args:
            image: 輸入影像
            
        Returns:
            List[Dict]: 預測結果列表
        """
        if self.model is None:
            raise RuntimeError("模型未載入")
        
        try:
            # 執行推理
            results = self.model(image, conf=self.confidence_threshold, 
                               imgsz=self.input_size)
            
            # 解析結果
            predictions = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        predictions.append({
                            'bbox': box,
                            'confidence': float(conf),
                            'class_id': cls,
                            'class_name': self.model.names[cls] if hasattr(self.model, 'names') else str(cls)
                        })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"預測失敗: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        影像前處理
        
        Args:
            image: 原始影像
            
        Returns:
            np.ndarray: 處理後的影像
        """
        # 調整尺寸
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)
        
        # 正規化
        image = image.astype(np.float32) / 255.0
        
        return image


class InstrumentDetector(YOLOWrapper):
    """儀器偵測器"""
    
    def __init__(self, model_path: str, device: str = "cpu", 
                 confidence_threshold: float = 0.5):
        """
        初始化儀器偵測器
        
        Args:
            model_path: 模型檔案路徑
            device: 運算設備
            confidence_threshold: 信心度閾值
        """
        super().__init__(model_path, device, confidence_threshold)
        
        # 儀器類型映射
        self.instrument_type_mapping = {
            'digital_display': InstrumentType.DIGITAL_DISPLAY,
            'seven_segment': InstrumentType.SEVEN_SEGMENT,
            'analog_gauge': InstrumentType.ANALOG_GAUGE,
            'lcd_screen': InstrumentType.LCD_SCREEN
        }
    
    def detect_instruments(self, image: np.ndarray, frame_id: str) -> List[DetectionResult]:
        """
        偵測儀器
        
        Args:
            image: 輸入影像
            frame_id: 影像幀 ID
            
        Returns:
            List[DetectionResult]: 偵測結果列表
        """
        start_time = time.time()
        
        try:
            predictions = self.predict(image)
            results = []
            
            for pred in predictions:
                # 解析邊界框
                bbox = pred['bbox']
                bounding_box = BoundingBox(
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    width=float(bbox[2] - bbox[0]),
                    height=float(bbox[3] - bbox[1]),
                    confidence=pred['confidence']
                )
                
                # 判斷儀器類型
                class_name = pred['class_name'].lower()
                instrument_type = self.instrument_type_mapping.get(
                    class_name, InstrumentType.UNKNOWN
                )
                
                # 建立偵測結果
                detection_result = DetectionResult(
                    frame_id=frame_id,
                    instrument_type=instrument_type,
                    bounding_box=bounding_box,
                    confidence=pred['confidence'],
                    processing_time=time.time() - start_time
                )
                
                results.append(detection_result)
            
            self.logger.info(f"偵測到 {len(results)} 個儀器", frame_id=frame_id)
            return results
            
        except Exception as e:
            self.logger.error(f"儀器偵測失敗: {e}", frame_id=frame_id)
            raise


class CornerDetector(YOLOWrapper):
    """角點偵測器"""
    
    def __init__(self, model_path: str, device: str = "cpu", 
                 confidence_threshold: float = 0.7):
        """
        初始化角點偵測器
        
        Args:
            model_path: 模型檔案路徑
            device: 運算設備
            confidence_threshold: 信心度閾值
        """
        super().__init__(model_path, device, confidence_threshold)
    
    def detect_corners(self, image: np.ndarray, roi_bbox: BoundingBox) -> Optional[CornerPoints]:
        """
        偵測螢幕角點
        
        Args:
            image: 輸入影像
            roi_bbox: 感興趣區域邊界框
            
        Returns:
            Optional[CornerPoints]: 角點座標，如果偵測失敗則返回 None
        """
        try:
            # 裁切 ROI
            x, y, w, h = int(roi_bbox.x), int(roi_bbox.y), int(roi_bbox.width), int(roi_bbox.height)
            roi_image = image[y:y+h, x:x+w]
            
            # 執行角點偵測
            predictions = self.predict(roi_image)
            
            if not predictions:
                return None
            
            # 找到信心度最高的預測
            best_pred = max(predictions, key=lambda p: p['confidence'])
            
            if best_pred['confidence'] < self.confidence_threshold:
                return None
            
            # 解析角點座標 (假設模型輸出4個角點)
            bbox = best_pred['bbox']
            
            # 將相對座標轉換為絕對座標
            corners = CornerPoints(
                top_left=(float(bbox[0] + x), float(bbox[1] + y)),
                top_right=(float(bbox[2] + x), float(bbox[1] + y)),
                bottom_left=(float(bbox[0] + x), float(bbox[3] + y)),
                bottom_right=(float(bbox[2] + x), float(bbox[3] + y)),
                confidence=best_pred['confidence']
            )
            
            return corners
            
        except Exception as e:
            self.logger.error(f"角點偵測失敗: {e}")
            return None
    
    def detect_corners_traditional(self, image: np.ndarray, roi_bbox: BoundingBox) -> Optional[CornerPoints]:
        """
        使用傳統方法偵測角點 (備援方法)
        
        Args:
            image: 輸入影像
            roi_bbox: 感興趣區域邊界框
            
        Returns:
            Optional[CornerPoints]: 角點座標
        """
        try:
            # 裁切 ROI
            x, y, w, h = int(roi_bbox.x), int(roi_bbox.y), int(roi_bbox.width), int(roi_bbox.height)
            roi_image = image[y:y+h, x:x+w]
            
            # 轉換為灰階
            if len(roi_image.shape) == 3:
                gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_image
            
            # 使用 Harris 角點偵測
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=4, qualityLevel=0.01, minDistance=10
            )
            
            if corners is None or len(corners) < 4:
                return None
            
            # 排序角點 (左上、右上、左下、右下)
            corners = corners.reshape(-1, 2)
            corners = self._sort_corners(corners)
            
            # 轉換為絕對座標
            corner_points = CornerPoints(
                top_left=(float(corners[0][0] + x), float(corners[0][1] + y)),
                top_right=(float(corners[1][0] + x), float(corners[1][1] + y)),
                bottom_left=(float(corners[2][0] + x), float(corners[2][1] + y)),
                bottom_right=(float(corners[3][0] + x), float(corners[3][1] + y)),
                confidence=0.8  # 傳統方法的預設信心度
            )
            
            return corner_points
            
        except Exception as e:
            self.logger.error(f"傳統角點偵測失敗: {e}")
            return None
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        排序角點座標
        
        Args:
            corners: 角點座標陣列
            
        Returns:
            np.ndarray: 排序後的角點 [左上, 右上, 左下, 右下]
        """
        # 計算重心
        center = np.mean(corners, axis=0)
        
        # 根據相對位置排序
        sorted_corners = np.zeros((4, 2))
        
        for corner in corners:
            if corner[0] < center[0] and corner[1] < center[1]:  # 左上
                sorted_corners[0] = corner
            elif corner[0] > center[0] and corner[1] < center[1]:  # 右上
                sorted_corners[1] = corner
            elif corner[0] < center[0] and corner[1] > center[1]:  # 左下
                sorted_corners[2] = corner
            else:  # 右下
                sorted_corners[3] = corner
        
        return sorted_corners


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.models: Dict[str, YOLOWrapper] = {}
        self.logger = get_logger("ModelManager")
    
    def load_model(self, model_name: str, model_path: str, model_type: str,
                   device: str = "cpu", confidence_threshold: float = 0.5) -> YOLOWrapper:
        """
        載入模型
        
        Args:
            model_name: 模型名稱
            model_path: 模型路徑
            model_type: 模型類型 (instrument_detection/corner_detection)
            device: 運算設備
            confidence_threshold: 信心度閾值
            
        Returns:
            YOLOWrapper: 模型包裝器
        """
        try:
            if model_type == "instrument_detection":
                model = InstrumentDetector(model_path, device, confidence_threshold)
            elif model_type == "corner_detection":
                model = CornerDetector(model_path, device, confidence_threshold)
            else:
                model = YOLOWrapper(model_path, device, confidence_threshold)
            
            self.models[model_name] = model
            self.logger.info(f"成功載入模型: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"載入模型失敗 {model_name}: {e}")
            raise
    
    def get_model(self, model_name: str) -> Optional[YOLOWrapper]:
        """
        取得模型
        
        Args:
            model_name: 模型名稱
            
        Returns:
            Optional[YOLOWrapper]: 模型包裝器
        """
        return self.models.get(model_name)
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸載模型
        
        Args:
            model_name: 模型名稱
            
        Returns:
            bool: 是否成功卸載
        """
        if model_name in self.models:
            del self.models[model_name]
            self.logger.info(f"成功卸載模型: {model_name}")
            return True
        return False
    
    def list_models(self) -> List[str]:
        """
        列出已載入的模型
        
        Returns:
            List[str]: 模型名稱列表
        """
        return list(self.models.keys())


# 全域模型管理器實例
model_manager = ModelManager()