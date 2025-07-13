"""
QualityAssessor 代理人
負責評估影像品質，計算「健康分數」，為後續處理提供品質依據
基於 MetaGPT 的真正代理人協作模式
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio

from metagpt.roles import Role
from metagpt.actions import Action
from metagpt.schema import Message

from ..models.messages import (
    StreamFrame, QualityReport, QualityMetrics, ProcessingStatus
)
from ..core.config import get_config, MeterGPTConfig
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
            
            # 計算品質指標
            metrics = await self._calculate_quality_metrics(frame_array)
            
            # 判斷是否可接受
            is_acceptable = self._is_quality_acceptable(metrics)
            
            # 生成改善建議
            recommendations = self._generate_recommendations(metrics)
            
            quality_report = QualityReport(
                frame_id=stream_frame.frame_id,
                camera_id=stream_frame.camera_info.camera_id,
                metrics=metrics,
                is_acceptable=is_acceptable,
                recommendations=recommendations
            )
            
            self.logger.info(
                f"品質評估完成 - 整體分數: {metrics.overall_score:.3f}, "
                f"可接受: {is_acceptable}",
                frame_id=stream_frame.frame_id
            )
            
            return quality_report
            
        except Exception as e:
            self.logger.error(
                f"品質評估失敗: {e}",
                frame_id=stream_frame.frame_id
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
                recommendations=["影像處理失敗，請檢查攝影機"]
            )
    
    async def _calculate_quality_metrics(self, frame: np.ndarray) -> QualityMetrics:
        """計算品質指標"""
        try:
            # 轉換為灰階
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 計算清晰度 (使用 Laplacian 變異數)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000.0, 1.0)  # 正規化
            
            # 計算亮度
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 最佳亮度在 0.5 附近
            
            # 計算對比度
            contrast = np.std(gray) / 255.0
            contrast_score = min(contrast * 4, 1.0)  # 正規化
            
            # 計算遮擋比例 (簡化版本)
            # 檢測過暗或過亮的區域
            dark_pixels = np.sum(gray < 30)
            bright_pixels = np.sum(gray > 225)
            total_pixels = gray.shape[0] * gray.shape[1]
            occlusion_ratio = (dark_pixels + bright_pixels) / total_pixels
            
            # 計算失真分數 (簡化版本)
            # 使用邊緣檢測來評估失真
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            distortion_score = min(edge_density * 10, 1.0)
            
            # 計算整體分數
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # 各指標權重
            scores = [
                sharpness_score,
                brightness_score,
                contrast_score,
                1.0 - occlusion_ratio,  # 遮擋比例越低越好
                distortion_score
            ]
            overall_score = sum(w * s for w, s in zip(weights, scores))
            
            return QualityMetrics(
                sharpness_score=sharpness_score,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                occlusion_ratio=occlusion_ratio,
                distortion_score=distortion_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            self.logger.error(f"品質指標計算失敗: {e}")
            # 返回預設低品質指標
            return QualityMetrics(
                sharpness_score=0.0,
                brightness_score=0.0,
                contrast_score=0.0,
                occlusion_ratio=1.0,
                distortion_score=0.0,
                overall_score=0.0
            )
    
    def _is_quality_acceptable(self, metrics: QualityMetrics) -> bool:
        """判斷品質是否可接受"""
        try:
            if self.config and self.config.quality:
                threshold = self.config.quality.overall_threshold
            else:
                threshold = 0.6  # 預設閾值
            
            return metrics.overall_score >= threshold
            
        except Exception as e:
            self.logger.error(f"品質判斷失敗: {e}")
            return False
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """生成改善建議"""
        recommendations = []
        
        try:
            if metrics.sharpness_score < 0.3:
                recommendations.append("影像模糊，建議調整攝影機焦距")
            
            if metrics.brightness_score < 0.5:
                recommendations.append("亮度不佳，建議調整光源或曝光設定")
            
            if metrics.contrast_score < 0.3:
                recommendations.append("對比度不足，建議調整攝影機設定")
            
            if metrics.occlusion_ratio > 0.3:
                recommendations.append("存在遮擋或過曝區域，建議調整攝影機角度")
            
            if metrics.distortion_score < 0.2:
                recommendations.append("影像失真嚴重，建議檢查攝影機鏡頭")
            
            if not recommendations:
                recommendations.append("影像品質良好")
            
        except Exception as e:
            self.logger.error(f"建議生成失敗: {e}")
            recommendations = ["品質評估異常，請檢查系統"]
        
        return recommendations


class QualityReportAction(Action):
    """品質報告發布動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("QualityReportAction")
    
    async def run(self, quality_report: QualityReport) -> Message:
        """
        發布品質報告
        
        Args:
            quality_report: 品質評估報告
            
        Returns:
            Message: 品質報告訊息
        """
        try:
            message = Message(
                content={
                    "type": "quality_report",
                    "frame_id": quality_report.frame_id,
                    "camera_id": quality_report.camera_id,
                    "quality_report": quality_report.dict(),
                    "is_acceptable": quality_report.is_acceptable,
                    "overall_score": quality_report.metrics.overall_score,
                    "timestamp": datetime.now().isoformat()
                },
                role="QualityAssessor",
                cause_by=type(self)
            )
            
            self.logger.info(
                f"發布品質報告: {quality_report.frame_id}, "
                f"分數: {quality_report.metrics.overall_score:.3f}",
                camera_id=quality_report.camera_id
            )
            
            return message
            
        except Exception as e:
            self.logger.error(f"品質報告發布失敗: {e}")
            raise


class QualityAssessor(Role):
    """品質評估代理人 - 基於 MetaGPT 的協作模式"""
    
    def __init__(self, config: Optional[MeterGPTConfig] = None):
        """
        初始化品質評估代理人
        
        Args:
            config: 系統配置
        """
        super().__init__(
            name="QualityAssessor",
            profile="品質評估代理人",
            goal="評估影像品質並提供改善建議",
            constraints="確保只有高品質影像進入後續處理流程"
        )
        
        self.config = config or get_config()
        self.logger = get_logger("QualityAssessor")
        
        # 設置動作
        self._set_actions([
            QualityAssessmentAction(),
            QualityReportAction()
        ])
        
        # 監聽串流幀和品質評估請求
        self._watch([
            "stream_frame",
            "quality_assessment_request"
        ])
        
        # 統計資訊
        self.assessment_count = 0
        self.acceptable_count = 0
    
    async def _act(self) -> Message:
        """
        執行品質評估動作
        
        Returns:
            Message: 評估結果訊息
        """
        try:
            # 取得需要評估的串流幀訊息
            messages = self.rc.memory.get_by_actions([
                "stream_frame",
                "quality_assessment_request"
            ])
            
            if not messages:
                return Message(
                    content={"type": "no_action", "status": "waiting"},
                    role=self.profile
                )
            
            # 處理最新的訊息
            latest_message = messages[-1]
            stream_frame_data = latest_message.content.get("stream_frame")
            
            if not stream_frame_data:
                return Message(
                    content={"type": "error", "error": "沒有找到串流幀資料"},
                    role=self.profile
                )
            
            # 重建 StreamFrame 物件
            stream_frame = StreamFrame(**stream_frame_data)
            
            # 執行品質評估
            assessment_action = QualityAssessmentAction()
            quality_report = await assessment_action.run(stream_frame)
            
            # 更新統計
            self.assessment_count += 1
            if quality_report.is_acceptable:
                self.acceptable_count += 1
            
            # 發布品質報告
            report_action = QualityReportAction()
            report_message = await report_action.run(quality_report)
            
            # 發布到環境
            await self.rc.env.publish_message(report_message)
            
            return Message(
                content={
                    "type": "quality_assessment_complete",
                    "frame_id": stream_frame.frame_id,
                    "is_acceptable": quality_report.is_acceptable,
                    "overall_score": quality_report.metrics.overall_score
                },
                role=self.profile
            )
            
        except Exception as e:
            self.logger.error(f"品質評估動作失敗: {e}")
            return Message(
                content={"type": "error", "error": str(e)},
                role=self.profile
            )
    
    def get_assessment_statistics(self) -> Dict[str, Any]:
        """取得評估統計資訊"""
        acceptance_rate = (
            self.acceptable_count / self.assessment_count 
            if self.assessment_count > 0 else 0.0
        )
        
        return {
            "total_assessments": self.assessment_count,
            "acceptable_count": self.acceptable_count,
            "acceptance_rate": acceptance_rate,
            "rejection_count": self.assessment_count - self.acceptable_count
        }
    
    def reset_statistics(self):
        """重置統計資訊"""
        self.assessment_count = 0
        self.acceptable_count = 0
        self.logger.info("品質評估統計已重置")