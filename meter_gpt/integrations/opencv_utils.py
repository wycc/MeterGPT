"""
OpenCV 工具函數
提供影像處理、品質評估和幾何變換等功能
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import math
from scipy import ndimage

from ..models.messages import QualityMetrics, CornerPoints, BoundingBox, ROI
from ..utils.logger import get_logger, log_execution_time


class ImageProcessor:
    """影像處理器"""
    
    def __init__(self):
        """初始化影像處理器"""
        self.logger = get_logger("ImageProcessor")
    
    @log_execution_time()
    def perspective_correction(self, image: np.ndarray, corners: CornerPoints) -> np.ndarray:
        """
        透視校正
        
        Args:
            image: 輸入影像
            corners: 四個角點座標
            
        Returns:
            np.ndarray: 校正後的影像
        """
        try:
            # 定義源點
            src_points = np.float32([
                corners.top_left,
                corners.top_right,
                corners.bottom_left,
                corners.bottom_right
            ])
            
            # 計算目標矩形尺寸
            width = max(
                np.linalg.norm(np.array(corners.top_right) - np.array(corners.top_left)),
                np.linalg.norm(np.array(corners.bottom_right) - np.array(corners.bottom_left))
            )
            height = max(
                np.linalg.norm(np.array(corners.bottom_left) - np.array(corners.top_left)),
                np.linalg.norm(np.array(corners.bottom_right) - np.array(corners.top_right))
            )
            
            # 定義目標點
            dst_points = np.float32([
                [0, 0],
                [width, 0],
                [0, height],
                [width, height]
            ])
            
            # 計算透視變換矩陣
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 執行透視變換
            corrected_image = cv2.warpPerspective(
                image, transform_matrix, (int(width), int(height))
            )
            
            self.logger.debug(f"透視校正完成，輸出尺寸: {int(width)}x{int(height)}")
            return corrected_image
            
        except Exception as e:
            self.logger.error(f"透視校正失敗: {e}")
            return image
    
    @log_execution_time()
    def extract_rois(self, image: np.ndarray, template_config: Dict[str, Any]) -> List[ROI]:
        """
        根據模板配置提取 ROI
        
        Args:
            image: 輸入影像
            template_config: 模板配置
            
        Returns:
            List[ROI]: ROI 列表
        """
        try:
            rois = []
            roi_configs = template_config.get('rois', [])
            
            for i, roi_config in enumerate(roi_configs):
                # 解析 ROI 配置
                field_name = roi_config.get('field_name', f'roi_{i}')
                bbox_config = roi_config.get('bbox', {})
                expected_format = roi_config.get('expected_format', 'text')
                
                # 建立邊界框
                bbox = BoundingBox(
                    x=bbox_config.get('x', 0),
                    y=bbox_config.get('y', 0),
                    width=bbox_config.get('width', 100),
                    height=bbox_config.get('height', 50),
                    confidence=1.0
                )
                
                # 提取 ROI 影像
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                roi_image = image[y:y+h, x:x+w]
                
                # 編碼 ROI 影像
                _, encoded_image = cv2.imencode('.jpg', roi_image)
                roi_data = encoded_image.tobytes()
                
                # 建立 ROI 物件
                roi = ROI(
                    roi_id=f"roi_{i}_{field_name}",
                    field_name=field_name,
                    bounding_box=bbox,
                    roi_data=roi_data,
                    expected_format=expected_format
                )
                
                rois.append(roi)
            
            self.logger.info(f"提取了 {len(rois)} 個 ROI")
            return rois
            
        except Exception as e:
            self.logger.error(f"ROI 提取失敗: {e}")
            return []
    
    def enhance_image(self, image: np.ndarray, enhancement_type: str = "auto") -> np.ndarray:
        """
        影像增強
        
        Args:
            image: 輸入影像
            enhancement_type: 增強類型 (auto/hdr/denoise/sharpen)
            
        Returns:
            np.ndarray: 增強後的影像
        """
        try:
            enhanced_image = image.copy()
            
            if enhancement_type == "auto" or enhancement_type == "hdr":
                # HDR 風格增強
                enhanced_image = self._hdr_enhancement(enhanced_image)
            
            if enhancement_type == "auto" or enhancement_type == "denoise":
                # 降噪
                enhanced_image = self._denoise_image(enhanced_image)
            
            if enhancement_type == "auto" or enhancement_type == "sharpen":
                # 銳化
                enhanced_image = self._sharpen_image(enhanced_image)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.error(f"影像增強失敗: {e}")
            return image
    
    def _hdr_enhancement(self, image: np.ndarray) -> np.ndarray:
        """HDR 風格增強"""
        # 轉換為 LAB 色彩空間
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 對 L 通道進行 CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # 合併通道
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """影像降噪"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """影像銳化"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)


class QualityAssessor:
    """影像品質評估器"""
    
    def __init__(self):
        """初始化品質評估器"""
        self.logger = get_logger("QualityAssessor")
    
    @log_execution_time()
    def assess_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        評估影像品質
        
        Args:
            image: 輸入影像
            
        Returns:
            QualityMetrics: 品質指標
        """
        try:
            # 計算各項品質指標
            sharpness_score = self._calculate_sharpness(image)
            brightness_score = self._calculate_brightness(image)
            contrast_score = self._calculate_contrast(image)
            occlusion_ratio = self._calculate_occlusion(image)
            distortion_score = self._calculate_distortion(image)
            
            # 計算整體品質分數
            overall_score = self._calculate_overall_score(
                sharpness_score, brightness_score, contrast_score,
                occlusion_ratio, distortion_score
            )
            
            quality_metrics = QualityMetrics(
                sharpness_score=sharpness_score,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                occlusion_ratio=occlusion_ratio,
                distortion_score=distortion_score,
                overall_score=overall_score
            )
            
            self.logger.debug(f"品質評估完成，整體分數: {overall_score:.3f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"品質評估失敗: {e}")
            # 返回預設的低品質指標
            return QualityMetrics(
                sharpness_score=0.0,
                brightness_score=0.0,
                contrast_score=0.0,
                occlusion_ratio=1.0,
                distortion_score=0.0,
                overall_score=0.0
            )
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """計算清晰度分數"""
        try:
            # 轉換為灰階
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 使用 Laplacian 算子計算清晰度
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 正規化到 0-1 範圍
            sharpness = min(laplacian_var / 1000.0, 1.0)
            
            return float(sharpness)
            
        except Exception:
            return 0.0
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """計算亮度分數"""
        try:
            # 轉換為灰階
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 計算平均亮度
            mean_brightness = np.mean(gray) / 255.0
            
            # 理想亮度範圍是 0.3-0.7
            if 0.3 <= mean_brightness <= 0.7:
                brightness_score = 1.0
            elif mean_brightness < 0.3:
                brightness_score = mean_brightness / 0.3
            else:
                brightness_score = (1.0 - mean_brightness) / 0.3
            
            return float(max(0.0, brightness_score))
            
        except Exception:
            return 0.0
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """計算對比度分數"""
        try:
            # 轉換為灰階
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 計算標準差作為對比度指標
            contrast = np.std(gray) / 255.0
            
            # 正規化到 0-1 範圍
            contrast_score = min(contrast * 4.0, 1.0)
            
            return float(contrast_score)
            
        except Exception:
            return 0.0
    
    def _calculate_occlusion(self, image: np.ndarray) -> float:
        """計算遮擋比例"""
        try:
            # 轉換為灰階
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 使用閾值分割找出可能的遮擋區域
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            
            # 計算黑色像素比例
            black_pixels = np.sum(binary == 0)
            total_pixels = binary.size
            
            occlusion_ratio = black_pixels / total_pixels
            
            return float(occlusion_ratio)
            
        except Exception:
            return 1.0
    
    def _calculate_distortion(self, image: np.ndarray) -> float:
        """計算失真分數"""
        try:
            # 簡化的失真檢測，基於邊緣檢測
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 邊緣檢測
            edges = cv2.Canny(gray, 50, 150)
            
            # 計算邊緣密度
            edge_density = np.sum(edges > 0) / edges.size
            
            # 正規化失真分數 (邊緣密度越高，失真越小)
            distortion_score = min(edge_density * 10.0, 1.0)
            
            return float(distortion_score)
            
        except Exception:
            return 0.0
    
    def _calculate_overall_score(self, sharpness: float, brightness: float,
                               contrast: float, occlusion: float, distortion: float) -> float:
        """計算整體品質分數"""
        # 權重設定
        weights = {
            'sharpness': 0.3,
            'brightness': 0.2,
            'contrast': 0.2,
            'occlusion': 0.2,  # 遮擋比例越低越好
            'distortion': 0.1
        }
        
        # 計算加權平均 (遮擋比例需要反轉)
        overall_score = (
            weights['sharpness'] * sharpness +
            weights['brightness'] * brightness +
            weights['contrast'] * contrast +
            weights['occlusion'] * (1.0 - occlusion) +
            weights['distortion'] * distortion
        )
        
        return float(max(0.0, min(1.0, overall_score)))


class GeometryUtils:
    """幾何工具函數"""
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """計算兩點間距離"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """計算三點形成的角度"""
        # 向量 1->2 和 1->3
        v1 = (point2[0] - point1[0], point2[1] - point1[1])
        v2 = (point3[0] - point1[0], point3[1] - point1[1])
        
        # 計算角度
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # 限制範圍
        
        return math.acos(cos_angle) * 180.0 / math.pi
    
    @staticmethod
    def is_rectangle(corners: CornerPoints, tolerance: float = 10.0) -> bool:
        """檢查四個角點是否形成矩形"""
        points = [
            corners.top_left,
            corners.top_right,
            corners.bottom_right,
            corners.bottom_left
        ]
        
        # 檢查四個角度是否接近 90 度
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            angle = GeometryUtils.calculate_angle(p2, p1, p3)
            if abs(angle - 90.0) > tolerance:
                return False
        
        return True
    
    @staticmethod
    def calculate_rectangle_area(corners: CornerPoints) -> float:
        """計算矩形面積"""
        # 計算寬度和高度
        width = GeometryUtils.calculate_distance(corners.top_left, corners.top_right)
        height = GeometryUtils.calculate_distance(corners.top_left, corners.bottom_left)
        
        return width * height


class StreamProcessor:
    """串流處理器"""
    
    def __init__(self, buffer_size: int = 10):
        """
        初始化串流處理器
        
        Args:
            buffer_size: 緩衝區大小
        """
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.logger = get_logger("StreamProcessor")
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """
        添加影像幀到緩衝區
        
        Args:
            frame: 影像幀
            timestamp: 時間戳記
        """
        self.frame_buffer.append((frame, timestamp))
        
        # 保持緩衝區大小
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """取得最新的影像幀"""
        if self.frame_buffer:
            return self.frame_buffer[-1]
        return None
    
    def get_stable_frame(self, stability_threshold: float = 0.1) -> Optional[np.ndarray]:
        """
        取得穩定的影像幀
        
        Args:
            stability_threshold: 穩定性閾值
            
        Returns:
            Optional[np.ndarray]: 穩定的影像幀
        """
        if len(self.frame_buffer) < 3:
            return None
        
        try:
            # 比較最近的幾幀
            recent_frames = [frame for frame, _ in self.frame_buffer[-3:]]
            
            # 計算幀間差異
            diff1 = cv2.absdiff(recent_frames[0], recent_frames[1])
            diff2 = cv2.absdiff(recent_frames[1], recent_frames[2])
            
            # 計算平均差異
            avg_diff1 = np.mean(diff1) / 255.0
            avg_diff2 = np.mean(diff2) / 255.0
            
            # 如果差異小於閾值，認為是穩定的
            if avg_diff1 < stability_threshold and avg_diff2 < stability_threshold:
                return recent_frames[-1]
            
            return None
            
        except Exception as e:
            self.logger.error(f"穩定幀檢測失敗: {e}")
            return None


# 工具函數實例
image_processor = ImageProcessor()
quality_assessor = QualityAssessor()
geometry_utils = GeometryUtils()