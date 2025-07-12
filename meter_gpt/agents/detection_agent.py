"""
DetectionAgent 代理人
負責儀器偵測和角點偵測，包括透視校正和 ROI 提取
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import asyncio

from metagpt.agent import Agent
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message

from ..models.messages import (
    StreamFrame, DetectionResult, CornerPoints, ROI, BoundingBox, InstrumentType
)
from ..core.config import get_config
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
                self.instrument_detector = model_manager.load_model(
                    model_name="instrument_detector",
                    model_path=model_config.model_path,
                    model_type="instrument_detection",
                    device=model_config.device,
                    confidence_threshold=model_config.confidence_threshold
                )
                self.logger.info("儀器偵測器初始化成功")
            else:
                self.logger.warning("沒有找到儀器偵測模型配置")
        except Exception as e:
            self.logger.error(f"儀器偵測器初始化失敗: {e}")
    
    async def run(self, stream_frame: StreamFrame) -> List[DetectionResult]:
        """
        執行儀器偵測
        
        Args:
            stream_frame: 串流影像幀
            
        Returns:
            List[DetectionResult]: 偵測結果列表
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
            enhanced_frame = image_processor.enhance_image(frame_array, "auto")
            
            # 執行儀器偵測
            detection_results = self.instrument_detector.detect_instruments(
                enhanced_frame, stream_frame.frame_id
            )
            
            self.logger.info(
                f"偵測到 {len(detection_results)} 個儀器",
                frame_id=stream_frame.frame_id
            )
            
            return detection_results
            
        except Exception as e:
            self.logger.error(
                f"儀器偵測失敗: {e}",
                frame_id=stream_frame.frame_id
            )
            return []


class CornerDetectionAction(Action):
    """角點偵測動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("CornerDetectionAction")
        self.config = get_config()
        self.corner_detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """初始化角點偵測器"""
        try:
            if self.config and self.config.corner_detection_model:
                model_config = self.config.corner_detection_model
                self.corner_detector = model_manager.load_model(
                    model_name="corner_detector",
                    model_path=model_config.model_path,
                    model_type="corner_detection",
                    device=model_config.device,
                    confidence_threshold=model_config.confidence_threshold
                )
                self.logger.info("角點偵測器初始化成功")
            else:
                self.logger.warning("沒有找到角點偵測模型配置")
        except Exception as e:
            self.logger.error(f"角點偵測器初始化失敗: {e}")
    
    async def run(self, image: np.ndarray, detection_result: DetectionResult) -> Optional[CornerPoints]:
        """
        執行角點偵測
        
        Args:
            image: 輸入影像
            detection_result: 儀器偵測結果
            
        Returns:
            Optional[CornerPoints]: 角點座標
        """
        try:
            if not self.corner_detector:
                # 使用傳統方法作為備援
                return self.corner_detector.detect_corners_traditional(
                    image, detection_result.bounding_box
                )
            
            # 使用深度學習模型偵測角點
            corner_points = self.corner_detector.detect_corners(
                image, detection_result.bounding_box
            )
            
            if corner_points:
                # 驗證角點是否形成合理的矩形
                if geometry_utils.is_rectangle(corner_points, tolerance=15.0):
                    self.logger.debug(
                        f"成功偵測到角點",
                        frame_id=detection_result.frame_id,
                        confidence=corner_points.confidence
                    )
                    return corner_points
                else:
                    self.logger.warning(
                        f"偵測到的角點不形成矩形，使用傳統方法重試",
                        frame_id=detection_result.frame_id
                    )
                    return self.corner_detector.detect_corners_traditional(
                        image, detection_result.bounding_box
                    )
            else:
                # 使用傳統方法作為備援
                return self.corner_detector.detect_corners_traditional(
                    image, detection_result.bounding_box
                )
                
        except Exception as e:
            self.logger.error(
                f"角點偵測失敗: {e}",
                frame_id=detection_result.frame_id
            )
            return None


class ROIExtractionAction(Action):
    """ROI 提取動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("ROIExtractionAction")
        self.config = get_config()
    
    async def run(self, image: np.ndarray, detection_result: DetectionResult,
                 corner_points: Optional[CornerPoints]) -> List[ROI]:
        """
        提取感興趣區域
        
        Args:
            image: 輸入影像
            detection_result: 儀器偵測結果
            corner_points: 角點座標
            
        Returns:
            List[ROI]: ROI 列表
        """
        try:
            # 如果有角點，先進行透視校正
            if corner_points:
                corrected_image = image_processor.perspective_correction(image, corner_points)
                self.logger.debug(
                    f"透視校正完成",
                    frame_id=detection_result.frame_id
                )
            else:
                # 直接裁切儀器區域
                bbox = detection_result.bounding_box
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                corrected_image = image[y:y+h, x:x+w]
                self.logger.debug(
                    f"使用直接裁切",
                    frame_id=detection_result.frame_id
                )
            
            # 根據儀器類型取得模板配置
            template_config = self._get_template_config(detection_result.instrument_type)
            
            # 提取 ROI
            rois = image_processor.extract_rois(corrected_image, template_config)
            
            self.logger.info(
                f"提取了 {len(rois)} 個 ROI",
                frame_id=detection_result.frame_id,
                instrument_type=detection_result.instrument_type.value
            )
            
            return rois
            
        except Exception as e:
            self.logger.error(
                f"ROI 提取失敗: {e}",
                frame_id=detection_result.frame_id
            )
            return []
    
    def _get_template_config(self, instrument_type: InstrumentType) -> Dict[str, Any]:
        """
        取得儀器模板配置
        
        Args:
            instrument_type: 儀器類型
            
        Returns:
            Dict[str, Any]: 模板配置
        """
        try:
            if self.config and self.config.instrument_templates:
                template_key = instrument_type.value
                template_config = self.config.instrument_templates.get(template_key)
                
                if template_config:
                    return template_config
            
            # 返回預設模板配置
            return self._get_default_template_config(instrument_type)
            
        except Exception as e:
            self.logger.error(f"取得模板配置失敗: {e}")
            return self._get_default_template_config(instrument_type)
    
    def _get_default_template_config(self, instrument_type: InstrumentType) -> Dict[str, Any]:
        """取得預設模板配置"""
        default_configs = {
            InstrumentType.DIGITAL_DISPLAY: {
                'rois': [
                    {
                        'field_name': 'main_display',
                        'bbox': {'x': 10, 'y': 10, 'width': 200, 'height': 80},
                        'expected_format': 'text'
                    }
                ]
            },
            InstrumentType.SEVEN_SEGMENT: {
                'rois': [
                    {
                        'field_name': 'seven_segment_display',
                        'bbox': {'x': 5, 'y': 5, 'width': 150, 'height': 60},
                        'expected_format': 'seven_segment'
                    }
                ]
            },
            InstrumentType.LCD_SCREEN: {
                'rois': [
                    {
                        'field_name': 'lcd_display',
                        'bbox': {'x': 15, 'y': 15, 'width': 180, 'height': 70},
                        'expected_format': 'text'
                    }
                ]
            },
            InstrumentType.ANALOG_GAUGE: {
                'rois': [
                    {
                        'field_name': 'gauge_reading',
                        'bbox': {'x': 20, 'y': 20, 'width': 160, 'height': 160},
                        'expected_format': 'analog'
                    }
                ]
            }
        }
        
        return default_configs.get(instrument_type, {'rois': []})


class DetectionAgent(Role):
    """偵測代理人"""
    
    def __init__(self, name: str = "DetectionAgent", **kwargs):
        """
        初始化偵測代理人
        
        Args:
            name: 代理人名稱
        """
        super().__init__(name=name, **kwargs)
        
        # 設置動作
        self._init_actions([
            InstrumentDetectionAction(),
            CornerDetectionAction(),
            ROIExtractionAction()
        ])
        
        self.logger = get_logger("DetectionAgent")
        self.config = get_config()
        
        # 偵測歷史記錄
        self.detection_history: Dict[str, List[DetectionResult]] = {}
        self.max_history_size = 50
    
    @log_agent_action("DetectionAgent")
    async def _act(self) -> Message:
        """執行代理人動作"""
        try:
            # 從訊息中取得串流幀
            stream_frames = []
            for msg in self.rc.memory.get():
                if hasattr(msg, 'content') and 'stream_frame' in str(msg.content):
                    # 這裡應該解析訊息內容取得 StreamFrame
                    # 為了示範，我們跳過實際解析
                    pass
            
            if not stream_frames:
                return Message(
                    content="沒有收到串流幀資料",
                    role=self.profile,
                    cause_by=InstrumentDetectionAction
                )
            
            # 處理第一個串流幀
            stream_frame = stream_frames[0]
            detection_results = await self.detect_and_extract(stream_frame)
            
            # 建立回應訊息
            message_content = {
                'action': 'detection_and_extraction',
                'frame_id': stream_frame.frame_id,
                'detection_count': len(detection_results),
                'timestamp': datetime.now().isoformat()
            }
            
            return Message(
                content=str(message_content),
                role=self.profile,
                cause_by=InstrumentDetectionAction
            )
            
        except Exception as e:
            self.logger.error(f"代理人動作執行失敗: {e}")
            return Message(
                content=f"Error: {str(e)}",
                role=self.profile,
                cause_by=InstrumentDetectionAction
            )
    
    async def detect_and_extract(self, stream_frame: StreamFrame) -> List[Dict[str, Any]]:
        """
        執行完整的偵測和提取流程
        
        Args:
            stream_frame: 串流影像幀
            
        Returns:
            List[Dict[str, Any]]: 偵測和提取結果
        """
        try:
            results = []
            
            # 解碼影像
            frame_array = cv2.imdecode(
                np.frombuffer(stream_frame.frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if frame_array is None:
                raise ValueError("無法解碼影像資料")
            
            # 1. 儀器偵測
            instrument_action = InstrumentDetectionAction()
            detection_results = await instrument_action.run(stream_frame)
            
            # 更新偵測歷史
            self._update_detection_history(detection_results)
            
            # 2. 對每個偵測到的儀器進行角點偵測和 ROI 提取
            corner_action = CornerDetectionAction()
            roi_action = ROIExtractionAction()
            
            for detection_result in detection_results:
                try:
                    # 角點偵測
                    corner_points = await corner_action.run(frame_array, detection_result)
                    
                    # ROI 提取
                    rois = await roi_action.run(frame_array, detection_result, corner_points)
                    
                    # 組合結果
                    result = {
                        'detection_result': detection_result,
                        'corner_points': corner_points,
                        'rois': rois,
                        'processing_success': True
                    }
                    
                    results.append(result)
                    
                    self.logger.log_detection_result(
                        detection_result.frame_id,
                        detection_result.instrument_type.value,
                        detection_result.confidence
                    )
                    
                except Exception as e:
                    self.logger.error(
                        f"處理儀器失敗: {e}",
                        frame_id=stream_frame.frame_id,
                        instrument_type=detection_result.instrument_type.value
                    )
                    
                    # 添加失敗結果
                    result = {
                        'detection_result': detection_result,
                        'corner_points': None,
                        'rois': [],
                        'processing_success': False,
                        'error': str(e)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"偵測和提取流程失敗: {e}")
            raise
    
    def _update_detection_history(self, detection_results: List[DetectionResult]):
        """更新偵測歷史記錄"""
        try:
            for result in detection_results:
                camera_id = result.frame_id.split('_')[0] if '_' in result.frame_id else 'unknown'
                
                if camera_id not in self.detection_history:
                    self.detection_history[camera_id] = []
                
                self.detection_history[camera_id].append(result)
                
                # 限制歷史記錄大小
                if len(self.detection_history[camera_id]) > self.max_history_size:
                    self.detection_history[camera_id].pop(0)
                    
        except Exception as e:
            self.logger.error(f"更新偵測歷史失敗: {e}")
    
    def get_detection_statistics(self, camera_id: str) -> Dict[str, Any]:
        """
        取得偵測統計資訊
        
        Args:
            camera_id: 攝影機 ID
            
        Returns:
            Dict[str, Any]: 統計資訊
        """
        try:
            history = self.detection_history.get(camera_id, [])
            if not history:
                return {'total_detections': 0, 'instrument_types': {}, 'average_confidence': 0.0}
            
            # 統計儀器類型
            instrument_counts = {}
            total_confidence = 0.0
            
            for result in history:
                instrument_type = result.instrument_type.value
                instrument_counts[instrument_type] = instrument_counts.get(instrument_type, 0) + 1
                total_confidence += result.confidence
            
            return {
                'total_detections': len(history),
                'instrument_types': instrument_counts,
                'average_confidence': total_confidence / len(history),
                'latest_detection': history[-1].timestamp.isoformat() if history else None
            }
            
        except Exception as e:
            self.logger.error(f"取得偵測統計失敗: {e}")
            return {'error': str(e)}
    
    async def get_detection_summary(self) -> Dict[str, Any]:
        """取得偵測摘要"""
        try:
            summary = {
                'total_cameras': len(self.detection_history),
                'camera_statistics': {},
                'overall_detection_rate': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            total_detections = 0
            
            for camera_id, history in self.detection_history.items():
                stats = self.get_detection_statistics(camera_id)
                summary['camera_statistics'][camera_id] = stats
                total_detections += stats['total_detections']
            
            # 計算整體偵測率 (簡化計算)
            if len(self.detection_history) > 0:
                summary['overall_detection_rate'] = total_detections / len(self.detection_history)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"取得偵測摘要失敗: {e}")
            return {'error': str(e)}


# 建立全域 DetectionAgent 實例的工廠函數
def create_detection_agent() -> DetectionAgent:
    """建立 DetectionAgent 實例"""
    return DetectionAgent()