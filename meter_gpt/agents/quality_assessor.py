"""
QualityAssessor 代理人
負責評估影像品質，計算「健康分數」，為後續處理提供品質依據
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio

from metagpt.agent import Agent
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message

from ..models.messages import (
    StreamFrame, QualityReport, QualityMetrics, ProcessingStatus
)
from ..core.config import get_config
from ..utils.logger import get_logger, log_agent_action
from ..integrations.opencv_utils import quality_assessor


class QualityAssessmentAction(Action):
    """品質評估動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("QualityAssessmentAction")
        self.config = get_config()
    
    async def run(self, stream_frame: StreamFrame) -> QualityReport:
        """
        執行品質評估
        
        Args:
            stream_frame: 串流影像幀
            
        Returns:
            QualityReport: 品質評估報告
        """
        try:
            # 解碼影像
            frame_array = cv2.imdecode(
                np.frombuffer(stream_frame.frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if frame_array is None:
                raise ValueError("無法解碼影像資料")
            
            # 執行品質評估
            quality_metrics = quality_assessor.assess_quality(frame_array)
            
            # 取得品質配置
            quality_config = self.config.quality if self.config else None
            overall_threshold = quality_config.overall_threshold if quality_config else 0.6
            
            # 判斷是否達到可接受品質
            is_acceptable = quality_metrics.overall_score >= overall_threshold
            
            # 生成改善建議
            recommendations = self._generate_recommendations(quality_metrics, quality_config)
            
            # 建立品質報告
            quality_report = QualityReport(
                frame_id=stream_frame.frame_id,
                camera_id=stream_frame.camera_info.camera_id,
                metrics=quality_metrics,
                is_acceptable=is_acceptable,
                recommendations=recommendations
            )
            
            self.logger.log_quality_assessment(
                stream_frame.frame_id,
                stream_frame.camera_info.camera_id,
                quality_metrics.overall_score,
                is_acceptable
            )
            
            return quality_report
            
        except Exception as e:
            self.logger.error(
                f"品質評估失敗: {e}",
                frame_id=stream_frame.frame_id,
                camera_id=stream_frame.camera_info.camera_id
            )
            
            # 返回預設的低品質報告
            return QualityReport(
                frame_id=stream_frame.frame_id,
                camera_id=stream_frame.camera_info.camera_id,
                metrics=QualityMetrics(
                    sharpness_score=0.0,
                    brightness_score=0.0,
                    contrast_score=0.0,
                    occlusion_ratio=1.0,
                    distortion_score=0.0,
                    overall_score=0.0
                ),
                is_acceptable=False,
                recommendations=["影像品質評估失敗，請檢查攝影機連接"]
            )
    
    def _generate_recommendations(self, metrics: QualityMetrics, 
                                config: Optional[Any]) -> List[str]:
        """
        生成改善建議
        
        Args:
            metrics: 品質指標
            config: 品質配置
            
        Returns:
            List[str]: 改善建議列表
        """
        recommendations = []
        
        try:
            # 清晰度建議
            sharpness_threshold = config.sharpness_threshold if config else 0.3
            if metrics.sharpness_score < sharpness_threshold:
                recommendations.append("影像清晰度不足，建議調整攝影機焦距或清潔鏡頭")
            
            # 亮度建議
            brightness_range = config.brightness_range if config else (0.2, 0.8)
            if metrics.brightness_score < 0.5:
                if metrics.brightness_score < brightness_range[0]:
                    recommendations.append("影像過暗，建議增加照明或調整攝影機曝光設定")
                elif metrics.brightness_score > brightness_range[1]:
                    recommendations.append("影像過亮，建議減少照明或調整攝影機曝光設定")
            
            # 對比度建議
            contrast_threshold = config.contrast_threshold if config else 0.3
            if metrics.contrast_score < contrast_threshold:
                recommendations.append("影像對比度不足，建議調整攝影機對比度設定或改善照明條件")
            
            # 遮擋建議
            occlusion_threshold = config.occlusion_threshold if config else 0.3
            if metrics.occlusion_ratio > occlusion_threshold:
                recommendations.append("檢測到影像遮擋，建議清除攝影機視野中的障礙物")
            
            # 失真建議
            if metrics.distortion_score < 0.5:
                recommendations.append("檢測到影像失真，建議調整攝影機角度或位置")
            
            # 整體品質建議
            if metrics.overall_score < 0.4:
                recommendations.append("整體影像品質較差，建議考慮切換到備援攝影機")
            elif metrics.overall_score < 0.6:
                recommendations.append("影像品質需要改善，建議進行攝影機維護")
            
        except Exception as e:
            self.logger.error(f"生成建議失敗: {e}")
            recommendations.append("品質分析異常，請檢查系統狀態")
        
        return recommendations


class BatchQualityAssessmentAction(Action):
    """批次品質評估動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("BatchQualityAssessmentAction")
    
    async def run(self, stream_frames: List[StreamFrame]) -> List[QualityReport]:
        """
        批次執行品質評估
        
        Args:
            stream_frames: 串流影像幀列表
            
        Returns:
            List[QualityReport]: 品質評估報告列表
        """
        try:
            assessment_action = QualityAssessmentAction()
            
            # 並行處理多個影像幀
            tasks = [
                assessment_action.run(frame) 
                for frame in stream_frames
            ]
            
            quality_reports = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 過濾異常結果
            valid_reports = []
            for report in quality_reports:
                if isinstance(report, QualityReport):
                    valid_reports.append(report)
                else:
                    self.logger.error(f"品質評估異常: {report}")
            
            return valid_reports
            
        except Exception as e:
            self.logger.error(f"批次品質評估失敗: {e}")
            return []


class QualityAssessor(Role):
    """品質評估代理人"""
    
    def __init__(self, name: str = "QualityAssessor", **kwargs):
        """
        初始化品質評估代理人
        
        Args:
            name: 代理人名稱
        """
        super().__init__(name=name, **kwargs)
        
        # 設置動作
        self._init_actions([
            QualityAssessmentAction(),
            BatchQualityAssessmentAction()
        ])
        
        self.logger = get_logger("QualityAssessor")
        self.config = get_config()
        
        # 品質歷史記錄
        self.quality_history: Dict[str, List[QualityReport]] = {}
        self.max_history_size = 100
    
    @log_agent_action("QualityAssessor")
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
                    cause_by=QualityAssessmentAction
                )
            
            # 執行品質評估
            action = BatchQualityAssessmentAction()
            quality_reports = await action.run(stream_frames)
            
            # 更新品質歷史
            self._update_quality_history(quality_reports)
            
            # 建立回應訊息
            message_content = {
                'action': 'quality_assessment',
                'reports_count': len(quality_reports),
                'acceptable_count': len([r for r in quality_reports if r.is_acceptable]),
                'average_score': self._calculate_average_score(quality_reports),
                'timestamp': datetime.now().isoformat()
            }
            
            return Message(
                content=str(message_content),
                role=self.profile,
                cause_by=BatchQualityAssessmentAction
            )
            
        except Exception as e:
            self.logger.error(f"代理人動作執行失敗: {e}")
            return Message(
                content=f"Error: {str(e)}",
                role=self.profile,
                cause_by=QualityAssessmentAction
            )
    
    async def assess_single_frame(self, stream_frame: StreamFrame) -> QualityReport:
        """
        評估單一影像幀
        
        Args:
            stream_frame: 串流影像幀
            
        Returns:
            QualityReport: 品質評估報告
        """
        try:
            action = QualityAssessmentAction()
            quality_report = await action.run(stream_frame)
            
            # 更新品質歷史
            self._update_single_quality_history(quality_report)
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"單一幀品質評估失敗: {e}")
            raise
    
    async def assess_multiple_frames(self, stream_frames: List[StreamFrame]) -> List[QualityReport]:
        """
        評估多個影像幀
        
        Args:
            stream_frames: 串流影像幀列表
            
        Returns:
            List[QualityReport]: 品質評估報告列表
        """
        try:
            action = BatchQualityAssessmentAction()
            quality_reports = await action.run(stream_frames)
            
            # 更新品質歷史
            self._update_quality_history(quality_reports)
            
            return quality_reports
            
        except Exception as e:
            self.logger.error(f"多幀品質評估失敗: {e}")
            raise
    
    def _update_quality_history(self, quality_reports: List[QualityReport]):
        """更新品質歷史記錄"""
        try:
            for report in quality_reports:
                self._update_single_quality_history(report)
        except Exception as e:
            self.logger.error(f"更新品質歷史失敗: {e}")
    
    def _update_single_quality_history(self, quality_report: QualityReport):
        """更新單一品質歷史記錄"""
        try:
            camera_id = quality_report.camera_id
            
            if camera_id not in self.quality_history:
                self.quality_history[camera_id] = []
            
            self.quality_history[camera_id].append(quality_report)
            
            # 限制歷史記錄大小
            if len(self.quality_history[camera_id]) > self.max_history_size:
                self.quality_history[camera_id].pop(0)
                
        except Exception as e:
            self.logger.error(f"更新單一品質歷史失敗: {e}")
    
    def _calculate_average_score(self, quality_reports: List[QualityReport]) -> float:
        """計算平均品質分數"""
        if not quality_reports:
            return 0.0
        
        total_score = sum(report.metrics.overall_score for report in quality_reports)
        return total_score / len(quality_reports)
    
    def get_camera_quality_trend(self, camera_id: str, window_size: int = 10) -> Dict[str, float]:
        """
        取得攝影機品質趨勢
        
        Args:
            camera_id: 攝影機 ID
            window_size: 視窗大小
            
        Returns:
            Dict[str, float]: 品質趨勢指標
        """
        try:
            history = self.quality_history.get(camera_id, [])
            if len(history) < 2:
                return {'trend': 0.0, 'stability': 0.0, 'current_score': 0.0}
            
            # 取得最近的記錄
            recent_history = history[-window_size:]
            scores = [report.metrics.overall_score for report in recent_history]
            
            # 計算趨勢 (線性回歸斜率)
            if len(scores) >= 2:
                x = list(range(len(scores)))
                n = len(scores)
                sum_x = sum(x)
                sum_y = sum(scores)
                sum_xy = sum(x[i] * scores[i] for i in range(n))
                sum_x2 = sum(x[i] ** 2 for i in range(n))
                
                trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            else:
                trend = 0.0
            
            # 計算穩定性 (標準差的倒數)
            if len(scores) > 1:
                mean_score = sum(scores) / len(scores)
                variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
                stability = 1.0 / (1.0 + variance)
            else:
                stability = 1.0
            
            return {
                'trend': trend,
                'stability': stability,
                'current_score': scores[-1] if scores else 0.0,
                'average_score': sum(scores) / len(scores) if scores else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"取得品質趨勢失敗: {e}")
            return {'trend': 0.0, 'stability': 0.0, 'current_score': 0.0}
    
    def get_best_quality_camera(self, camera_ids: List[str]) -> Optional[str]:
        """
        取得品質最佳的攝影機
        
        Args:
            camera_ids: 攝影機 ID 列表
            
        Returns:
            Optional[str]: 品質最佳的攝影機 ID
        """
        try:
            best_camera = None
            best_score = -1.0
            
            for camera_id in camera_ids:
                trend_data = self.get_camera_quality_trend(camera_id)
                current_score = trend_data['current_score']
                
                if current_score > best_score:
                    best_score = current_score
                    best_camera = camera_id
            
            return best_camera
            
        except Exception as e:
            self.logger.error(f"取得最佳品質攝影機失敗: {e}")
            return None
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """取得品質摘要"""
        try:
            summary = {
                'total_cameras': len(self.quality_history),
                'camera_quality': {},
                'overall_health': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            total_score = 0.0
            camera_count = 0
            
            for camera_id, history in self.quality_history.items():
                if history:
                    latest_report = history[-1]
                    trend_data = self.get_camera_quality_trend(camera_id)
                    
                    summary['camera_quality'][camera_id] = {
                        'current_score': latest_report.metrics.overall_score,
                        'is_acceptable': latest_report.is_acceptable,
                        'trend': trend_data['trend'],
                        'stability': trend_data['stability'],
                        'last_assessment': latest_report.timestamp.isoformat()
                    }
                    
                    total_score += latest_report.metrics.overall_score
                    camera_count += 1
            
            # 計算整體健康度
            if camera_count > 0:
                summary['overall_health'] = total_score / camera_count
            
            return summary
            
        except Exception as e:
            self.logger.error(f"取得品質摘要失敗: {e}")
            return {'error': str(e)}


# 建立全域 QualityAssessor 實例的工廠函數
def create_quality_assessor() -> QualityAssessor:
    """建立 QualityAssessor 實例"""
    return QualityAssessor()