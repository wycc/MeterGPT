"""
DetectionAgent 代理人
負責儀器偵測和角點偵測，包括透視校正和 ROI 提取
基於 MetaGPT 的真正代理人協作模式
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import asyncio

from metagpt.roles import Role
from metagpt.actions import Action
from metagpt.schema import Message

from ..models.messages import (
    StreamFrame, DetectionResult, CornerPoints, ROI, BoundingBox, InstrumentType
)
from ..core.config import get_config, MeterGPTConfig
from ..utils.logger import get_logger, log_agent_action
from ..integrations.yolo_wrapper import model_manager
from ..integrations.opencv_utils import image_processor, geometry_utils


class InstrumentDetectionAction(Action):
    """儀器偵測動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("InstrumentDetectionAction")
        self.config = get_config()
        self.instrument_detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """初始化儀器偵測器"""
        try:
            if self.config and self.config.detection_model:
                model_config = self.config.detection_model
                # 這裡應該載入實際的 YOLO 模型
                # 暫時使用模擬偵測器
                self.instrument_detector = MockInstrumentDetector(model_config)
                self.logger.info("儀器偵測器初始化成功")
            else:
                self.logger.warning("沒有找到儀器偵測模型配置")
                self.instrument_detector = MockInstrumentDetector()
        except Exception as e:
            self.logger.error(f"儀器偵測器初始化失敗: {e}")
            self.instrument_detector = MockInstrumentDetector()
    
    async def run(self, stream_frame: StreamFrame) -> DetectionResult:
        """
        執行儀器偵測
        
        Args:
            stream_frame: 串流影像幀
            
        Returns:
            DetectionResult: 偵測結果
        """
        try:
            if not self.instrument_detector:
                raise RuntimeError("儀器偵測器未初始化")
            
            # 解碼影像
            frame_array = cv2.imdecode(
                np.frombuffer(stream_frame.frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if frame_array is None:
                raise ValueError("無法解碼影像資料")
            
            # 影像前處理
            enhanced_frame = self._enhance_image(frame_array)
            
            # 執行儀器偵測
            detection_result = await self.instrument_detector.detect_instruments(
                enhanced_frame, stream_frame.frame_id
            )
            
            self.logger.info(
                f"儀器偵測完成 - 類型: {detection_result.instrument_type}, "
                f"信心度: {detection_result.confidence:.3f}",
                frame_id=stream_frame.frame_id
            )
            
            return detection_result
            
        except Exception as e:
            self.logger.error(
                f"儀器偵測失敗: {e}",
                frame_id=stream_frame.frame_id
            )
            # 返回失敗的偵測結果
            return DetectionResult(
                frame_id=stream_frame.frame_id,
                instrument_type=InstrumentType.UNKNOWN,
                bounding_box=BoundingBox(x=0, y=0, width=0, height=0, confidence=0.0),
                confidence=0.0,
                processing_time=0.0
            )
    
    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """影像增強"""
        try:
            # 轉換為灰階
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 直方圖均衡化
            enhanced = cv2.equalizeHist(gray)
            
            # 高斯模糊去噪
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # 轉回彩色
            enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced_color
            
        except Exception as e:
            self.logger.error(f"影像增強失敗: {e}")
            return frame


class CornerDetectionAction(Action):
    """角點偵測動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("CornerDetectionAction")
        self.config = get_config()
    
    async def run(self, frame: np.ndarray, bounding_box: BoundingBox) -> Optional[CornerPoints]:
        """
        執行角點偵測
        
        Args:
            frame: 影像幀
            bounding_box: 儀器邊界框
            
        Returns:
            Optional[CornerPoints]: 角點座標
        """
        try:
            # 提取 ROI
            x, y, w, h = int(bounding_box.x), int(bounding_box.y), int(bounding_box.width), int(bounding_box.height)
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                return None
            
            # 轉換為灰階
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 使用 Harris 角點偵測
            corners = cv2.goodFeaturesToTrack(
                gray_roi,
                maxCorners=4,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=3
            )
            
            if corners is None or len(corners) < 4:
                # 使用邊緣偵測作為備選方案
                return self._detect_corners_by_edges(gray_roi, x, y)
            
            # 轉換座標到原始影像座標系
            corners = corners.reshape(-1, 2)
            corners[:, 0] += x
            corners[:, 1] += y
            
            # 排序角點 (左上、右上、左下、右下)
            sorted_corners = self._sort_corners(corners)
            
            corner_points = CornerPoints(
                top_left=tuple(sorted_corners[0]),
                top_right=tuple(sorted_corners[1]),
                bottom_left=tuple(sorted_corners[2]),
                bottom_right=tuple(sorted_corners[3]),
                confidence=0.8  # 簡化的信心度計算
            )
            
            return corner_points
            
        except Exception as e:
            self.logger.error(f"角點偵測失敗: {e}")
            return None
    
    def _detect_corners_by_edges(self, gray_roi: np.ndarray, offset_x: int, offset_y: int) -> Optional[CornerPoints]:
        """使用邊緣偵測來找角點"""
        try:
            # 邊緣偵測
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # 找輪廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # 找最大輪廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 多邊形近似
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) >= 4:
                # 取前四個點
                corners = approx[:4].reshape(-1, 2)
                corners[:, 0] += offset_x
                corners[:, 1] += offset_y
                
                sorted_corners = self._sort_corners(corners)
                
                return CornerPoints(
                    top_left=tuple(sorted_corners[0]),
                    top_right=tuple(sorted_corners[1]),
                    bottom_left=tuple(sorted_corners[2]),
                    bottom_right=tuple(sorted_corners[3]),
                    confidence=0.6
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"邊緣角點偵測失敗: {e}")
            return None
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """排序角點為左上、右上、左下、右下"""
        # 計算重心
        center = np.mean(corners, axis=0)
        
        # 根據相對位置排序
        sorted_corners = np.zeros((4, 2))
        
        for i, corner in enumerate(corners):
            if corner[0] < center[0] and corner[1] < center[1]:  # 左上
                sorted_corners[0] = corner
            elif corner[0] >= center[0] and corner[1] < center[1]:  # 右上
                sorted_corners[1] = corner
            elif corner[0] < center[0] and corner[1] >= center[1]:  # 左下
                sorted_corners[2] = corner
            else:  # 右下
                sorted_corners[3] = corner
        
        return sorted_corners


class DetectionResultAction(Action):
    """偵測結果發布動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("DetectionResultAction")
    
    async def run(self, detection_result: DetectionResult) -> Message:
        """
        發布偵測結果
        
        Args:
            detection_result: 偵測結果
            
        Returns:
            Message: 偵測結果訊息
        """
        try:
            message = Message(
                content={
                    "type": "detection_result",
                    "frame_id": detection_result.frame_id,
                    "detection_result": detection_result.dict(),
                    "instrument_type": detection_result.instrument_type,
                    "confidence": detection_result.confidence,
                    "timestamp": datetime.now().isoformat()
                },
                role="DetectionAgent",
                cause_by=type(self)
            )
            
            self.logger.info(
                f"發布偵測結果: {detection_result.frame_id}, "
                f"類型: {detection_result.instrument_type}, "
                f"信心度: {detection_result.confidence:.3f}"
            )
            
            return message
            
        except Exception as e:
            self.logger.error(f"偵測結果發布失敗: {e}")
            raise


class DetectionAgent(Role):
    """偵測代理人 - 基於 MetaGPT 的協作模式"""
    
    def __init__(self, config: Optional[MeterGPTConfig] = None):
        """
        初始化偵測代理人
        
        Args:
            config: 系統配置
        """
        super().__init__(
            name="DetectionAgent",
            profile="儀器偵測代理人",
            goal="偵測影像中的儀器並提取關鍵區域",
            constraints="確保偵測準確性和處理效率"
        )
        
        self.config = config or get_config()
        self.logger = get_logger("DetectionAgent")
        
        # 設置動作
        self._set_actions([
            InstrumentDetectionAction(),
            CornerDetectionAction(),
            DetectionResultAction()
        ])
        
        # 監聽品質報告和偵測請求
        self._watch([
            "quality_report",
            "detection_request"
        ])
        
        # 統計資訊
        self.detection_count = 0
        self.successful_detections = 0
    
    async def _act(self) -> Message:
        """
        執行偵測動作
        
        Returns:
            Message: 偵測結果訊息
        """
        try:
            # 取得需要偵測的訊息
            messages = self.rc.memory.get_by_actions([
                "quality_report",
                "detection_request"
            ])
            
            if not messages:
                return Message(
                    content={"type": "no_action", "status": "waiting"},
                    role=self.profile
                )
            
            # 處理最新的訊息
            latest_message = messages[-1]
            
            # 檢查品質是否可接受
            if latest_message.content.get("type") == "quality_report":
                if not latest_message.content.get("is_acceptable", False):
                    return Message(
                        content={
                            "type": "detection_skipped",
                            "reason": "影像品質不可接受"
                        },
                        role=self.profile
                    )
            
            # 取得串流幀資料
            stream_frame_data = latest_message.content.get("stream_frame")
            if not stream_frame_data:
                return Message(
                    content={"type": "error", "error": "沒有找到串流幀資料"},
                    role=self.profile
                )
            
            # 重建 StreamFrame 物件
            stream_frame = StreamFrame(**stream_frame_data)
            
            # 執行儀器偵測
            detection_action = InstrumentDetectionAction()
            detection_result = await detection_action.run(stream_frame)
            
            # 更新統計
            self.detection_count += 1
            if detection_result.confidence > 0.5:  # 閾值應該從配置讀取
                self.successful_detections += 1
            
            # 發布偵測結果
            result_action = DetectionResultAction()
            result_message = await result_action.run(detection_result)
            
            # 發布到環境
            await self.rc.env.publish_message(result_message)
            
            return Message(
                content={
                    "type": "detection_complete",
                    "frame_id": stream_frame.frame_id,
                    "instrument_type": detection_result.instrument_type,
                    "confidence": detection_result.confidence
                },
                role=self.profile
            )
            
        except Exception as e:
            self.logger.error(f"偵測動作失敗: {e}")
            return Message(
                content={"type": "error", "error": str(e)},
                role=self.profile
            )
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """取得偵測統計資訊"""
        success_rate = (
            self.successful_detections / self.detection_count 
            if self.detection_count > 0 else 0.0
        )
        
        return {
            "total_detections": self.detection_count,
            "successful_detections": self.successful_detections,
            "success_rate": success_rate,
            "failed_detections": self.detection_count - self.successful_detections
        }
    
    def reset_statistics(self):
        """重置統計資訊"""
        self.detection_count = 0
        self.successful_detections = 0
        self.logger.info("偵測統計已重置")


class MockInstrumentDetector:
    """模擬儀器偵測器"""
    
    def __init__(self, model_config=None):
        self.model_config = model_config
        self.logger = get_logger("MockInstrumentDetector")
    
    async def detect_instruments(self, frame: np.ndarray, frame_id: str) -> DetectionResult:
        """模擬儀器偵測"""
        try:
            # 模擬處理時間
            await asyncio.sleep(0.1)
            
            # 模擬偵測結果
            height, width = frame.shape[:2]
            
            # 假設在影像中央找到一個數位顯示器
            center_x, center_y = width // 2, height // 2
            bbox_width, bbox_height = min(200, width // 2), min(100, height // 2)
            
            bounding_box = BoundingBox(
                x=center_x - bbox_width // 2,
                y=center_y - bbox_height // 2,
                width=bbox_width,
                height=bbox_height,
                confidence=0.85
            )
            
            # 模擬角點偵測
            corner_points = CornerPoints(
                top_left=(bounding_box.x, bounding_box.y),
                top_right=(bounding_box.x + bounding_box.width, bounding_box.y),
                bottom_left=(bounding_box.x, bounding_box.y + bounding_box.height),
                bottom_right=(bounding_box.x + bounding_box.width, bounding_box.y + bounding_box.height),
                confidence=0.8
            )
            
            detection_result = DetectionResult(
                frame_id=frame_id,
                instrument_type=InstrumentType.DIGITAL_DISPLAY,
                bounding_box=bounding_box,
                corner_points=corner_points,
                confidence=0.85,
                processing_time=0.1
            )
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"模擬偵測失敗: {e}")
            return DetectionResult(
                frame_id=frame_id,
                instrument_type=InstrumentType.UNKNOWN,
                bounding_box=BoundingBox(x=0, y=0, width=0, height=0, confidence=0.0),
                confidence=0.0,
                processing_time=0.0
            )