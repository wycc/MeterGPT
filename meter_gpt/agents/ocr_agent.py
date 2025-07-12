"""
OCRAgent 代理人
負責執行 OCR 識別，支援多種文字識別引擎和格式
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
    ROI, OCRResult, ProcessingStatus, FailureType
)
from ..core.config import get_config
from ..utils.logger import get_logger, log_agent_action
from ..integrations.paddle_ocr_wrapper import ocr_manager


class OCRProcessingAction(Action):
    """OCR 處理動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("OCRProcessingAction")
        self.config = get_config()
        self._initialize_ocr_engines()
    
    def _initialize_ocr_engines(self):
        """初始化 OCR 引擎"""
        try:
            if self.config and self.config.ocr:
                ocr_config = self.config.ocr
                
                # 初始化 PaddleOCR
                ocr_manager.initialize_paddle_ocr(
                    use_angle_cls=True,
                    lang=ocr_config.language,
                    use_gpu=ocr_config.use_gpu,
                    confidence_threshold=ocr_config.confidence_threshold,
                    det_model_dir=ocr_config.det_model_dir,
                    rec_model_dir=ocr_config.rec_model_dir,
                    cls_model_dir=ocr_config.cls_model_dir
                )
                
                # 初始化七段顯示器 OCR
                ocr_manager.initialize_seven_segment_ocr(
                    confidence_threshold=ocr_config.confidence_threshold
                )
                
                self.logger.info("OCR 引擎初始化成功")
            else:
                self.logger.warning("沒有找到 OCR 配置，使用預設設定")
                ocr_manager.initialize_paddle_ocr()
                ocr_manager.initialize_seven_segment_ocr()
                
        except Exception as e:
            self.logger.error(f"OCR 引擎初始化失敗: {e}")
    
    async def run(self, roi: ROI, frame_id: str) -> OCRResult:
        """
        執行 OCR 識別
        
        Args:
            roi: 感興趣區域
            frame_id: 影像幀識別碼
            
        Returns:
            OCRResult: OCR 識別結果
        """
        try:
            # 使用 OCR 管理器進行識別
            ocr_result = ocr_manager.recognize_roi(roi, frame_id)
            
            # 後處理識別結果
            processed_result = self._post_process_result(ocr_result, roi)
            
            self.logger.log_ocr_result(
                frame_id,
                roi.roi_id,
                processed_result.recognized_text,
                processed_result.confidence
            )
            
            return processed_result
            
        except Exception as e:
            self.logger.error(
                f"OCR 識別失敗: {e}",
                frame_id=frame_id,
                roi_id=roi.roi_id
            )
            
            # 返回空結果
            return OCRResult(
                frame_id=frame_id,
                roi_id=roi.roi_id,
                recognized_text="",
                confidence=0.0,
                bounding_boxes=[],
                processing_time=0.0,
                ocr_engine="Unknown"
            )
    
    def _post_process_result(self, ocr_result: OCRResult, roi: ROI) -> OCRResult:
        """
        後處理 OCR 結果
        
        Args:
            ocr_result: 原始 OCR 結果
            roi: ROI 資訊
            
        Returns:
            OCRResult: 處理後的 OCR 結果
        """
        try:
            processed_text = ocr_result.recognized_text
            
            # 根據預期格式進行後處理
            if roi.expected_format.lower() in ['number', 'numeric']:
                processed_text = self._extract_numbers(processed_text)
            elif roi.expected_format.lower() in ['seven_segment', '7segment']:
                processed_text = self._clean_seven_segment_result(processed_text)
            elif roi.expected_format.lower() == 'text':
                processed_text = self._clean_text_result(processed_text)
            
            # 更新結果
            ocr_result.recognized_text = processed_text
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"OCR 結果後處理失敗: {e}")
            return ocr_result
    
    def _extract_numbers(self, text: str) -> str:
        """提取數字"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return ' '.join(numbers) if numbers else ""
    
    def _clean_seven_segment_result(self, text: str) -> str:
        """清理七段顯示器結果"""
        # 移除非數字字符，保留小數點和負號
        import re
        cleaned = re.sub(r'[^0-9.\-]', '', text)
        return cleaned
    
    def _clean_text_result(self, text: str) -> str:
        """清理文字結果"""
        # 移除多餘的空白和特殊字符
        cleaned = ' '.join(text.split())
        return cleaned.strip()


class BatchOCRProcessingAction(Action):
    """批次 OCR 處理動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("BatchOCRProcessingAction")
    
    async def run(self, rois: List[ROI], frame_id: str) -> List[OCRResult]:
        """
        批次執行 OCR 識別
        
        Args:
            rois: ROI 列表
            frame_id: 影像幀識別碼
            
        Returns:
            List[OCRResult]: OCR 識別結果列表
        """
        try:
            ocr_action = OCRProcessingAction()
            
            # 並行處理多個 ROI
            tasks = [
                ocr_action.run(roi, frame_id)
                for roi in rois
            ]
            
            ocr_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 過濾異常結果
            valid_results = []
            for result in ocr_results:
                if isinstance(result, OCRResult):
                    valid_results.append(result)
                else:
                    self.logger.error(f"OCR 處理異常: {result}")
            
            return valid_results
            
        except Exception as e:
            self.logger.error(f"批次 OCR 處理失敗: {e}")
            return []


class OCRQualityCheckAction(Action):
    """OCR 品質檢查動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("OCRQualityCheckAction")
        self.config = get_config()
    
    async def run(self, ocr_results: List[OCRResult]) -> Dict[str, Any]:
        """
        檢查 OCR 結果品質
        
        Args:
            ocr_results: OCR 結果列表
            
        Returns:
            Dict[str, Any]: 品質檢查結果
        """
        try:
            quality_report = {
                'total_results': len(ocr_results),
                'high_confidence_count': 0,
                'low_confidence_count': 0,
                'empty_results_count': 0,
                'average_confidence': 0.0,
                'quality_issues': [],
                'recommendations': []
            }
            
            if not ocr_results:
                quality_report['recommendations'].append("沒有 OCR 結果可供檢查")
                return quality_report
            
            # 取得信心度閾值
            confidence_threshold = 0.7
            if self.config and self.config.ocr:
                confidence_threshold = self.config.ocr.confidence_threshold
            
            total_confidence = 0.0
            
            for result in ocr_results:
                total_confidence += result.confidence
                
                if result.confidence >= confidence_threshold:
                    quality_report['high_confidence_count'] += 1
                else:
                    quality_report['low_confidence_count'] += 1
                
                if not result.recognized_text.strip():
                    quality_report['empty_results_count'] += 1
                
                # 檢查特定品質問題
                self._check_result_quality(result, quality_report)
            
            # 計算平均信心度
            quality_report['average_confidence'] = total_confidence / len(ocr_results)
            
            # 生成建議
            self._generate_quality_recommendations(quality_report)
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"OCR 品質檢查失敗: {e}")
            return {'error': str(e)}
    
    def _check_result_quality(self, result: OCRResult, quality_report: Dict[str, Any]):
        """檢查單個結果的品質問題"""
        try:
            # 檢查處理時間
            if result.processing_time > 5.0:  # 超過 5 秒
                quality_report['quality_issues'].append(
                    f"ROI {result.roi_id} 處理時間過長: {result.processing_time:.2f}s"
                )
            
            # 檢查識別文字長度
            if len(result.recognized_text) > 100:
                quality_report['quality_issues'].append(
                    f"ROI {result.roi_id} 識別文字過長，可能包含雜訊"
                )
            
            # 檢查邊界框數量
            if len(result.bounding_boxes) > 20:
                quality_report['quality_issues'].append(
                    f"ROI {result.roi_id} 偵測到過多文字區域，可能存在雜訊"
                )
                
        except Exception as e:
            self.logger.error(f"品質檢查失敗: {e}")
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]):
        """生成品質改善建議"""
        try:
            recommendations = quality_report['recommendations']
            
            # 低信心度建議
            if quality_report['low_confidence_count'] > quality_report['high_confidence_count']:
                recommendations.append("大部分 OCR 結果信心度較低，建議改善影像品質或調整 OCR 參數")
            
            # 空結果建議
            if quality_report['empty_results_count'] > 0:
                recommendations.append(f"有 {quality_report['empty_results_count']} 個空結果，建議檢查 ROI 設定")
            
            # 平均信心度建議
            if quality_report['average_confidence'] < 0.5:
                recommendations.append("整體 OCR 信心度較低，建議檢查影像預處理流程")
            
            # 品質問題建議
            if quality_report['quality_issues']:
                recommendations.append("發現品質問題，建議檢查具體問題並進行調整")
                
        except Exception as e:
            self.logger.error(f"生成建議失敗: {e}")


class OCRAgent(Role):
    """OCR 代理人"""
    
    def __init__(self, name: str = "OCRAgent", **kwargs):
        """
        初始化 OCR 代理人
        
        Args:
            name: 代理人名稱
        """
        super().__init__(name=name, **kwargs)
        
        # 設置動作
        self._init_actions([
            OCRProcessingAction(),
            BatchOCRProcessingAction(),
            OCRQualityCheckAction()
        ])
        
        self.logger = get_logger("OCRAgent")
        self.config = get_config()
        
        # OCR 歷史記錄
        self.ocr_history: Dict[str, List[OCRResult]] = {}
        self.max_history_size = 100
        
        # 效能統計
        self.performance_stats = {
            'total_processed': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0
        }
    
    @log_agent_action("OCRAgent")
    async def _act(self) -> Message:
        """執行代理人動作"""
        try:
            # 從訊息中取得 ROI 列表
            rois = []
            frame_id = "unknown"
            
            for msg in self.rc.memory.get():
                if hasattr(msg, 'content') and 'rois' in str(msg.content):
                    # 這裡應該解析訊息內容取得 ROI 列表
                    # 為了示範，我們跳過實際解析
                    pass
            
            if not rois:
                return Message(
                    content="沒有收到 ROI 資料",
                    role=self.profile,
                    cause_by=OCRProcessingAction
                )
            
            # 執行批次 OCR 處理
            batch_action = BatchOCRProcessingAction()
            ocr_results = await batch_action.run(rois, frame_id)
            
            # 品質檢查
            quality_action = OCRQualityCheckAction()
            quality_report = await quality_action.run(ocr_results)
            
            # 更新歷史記錄和統計
            self._update_history_and_stats(ocr_results, frame_id)
            
            # 建立回應訊息
            message_content = {
                'action': 'ocr_processing',
                'frame_id': frame_id,
                'results_count': len(ocr_results),
                'quality_report': quality_report,
                'timestamp': datetime.now().isoformat()
            }
            
            return Message(
                content=str(message_content),
                role=self.profile,
                cause_by=BatchOCRProcessingAction
            )
            
        except Exception as e:
            self.logger.error(f"代理人動作執行失敗: {e}")
            return Message(
                content=f"Error: {str(e)}",
                role=self.profile,
                cause_by=OCRProcessingAction
            )
    
    async def process_rois(self, rois: List[ROI], frame_id: str) -> List[OCRResult]:
        """
        處理 ROI 列表
        
        Args:
            rois: ROI 列表
            frame_id: 影像幀識別碼
            
        Returns:
            List[OCRResult]: OCR 結果列表
        """
        try:
            batch_action = BatchOCRProcessingAction()
            ocr_results = await batch_action.run(rois, frame_id)
            
            # 更新歷史記錄和統計
            self._update_history_and_stats(ocr_results, frame_id)
            
            return ocr_results
            
        except Exception as e:
            self.logger.error(f"處理 ROI 失敗: {e}")
            raise
    
    async def process_single_roi(self, roi: ROI, frame_id: str) -> OCRResult:
        """
        處理單個 ROI
        
        Args:
            roi: ROI
            frame_id: 影像幀識別碼
            
        Returns:
            OCRResult: OCR 結果
        """
        try:
            ocr_action = OCRProcessingAction()
            ocr_result = await ocr_action.run(roi, frame_id)
            
            # 更新歷史記錄和統計
            self._update_history_and_stats([ocr_result], frame_id)
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"處理單個 ROI 失敗: {e}")
            raise
    
    def _update_history_and_stats(self, ocr_results: List[OCRResult], frame_id: str):
        """更新歷史記錄和統計資訊"""
        try:
            # 更新歷史記錄
            camera_id = frame_id.split('_')[0] if '_' in frame_id else 'unknown'
            
            if camera_id not in self.ocr_history:
                self.ocr_history[camera_id] = []
            
            self.ocr_history[camera_id].extend(ocr_results)
            
            # 限制歷史記錄大小
            if len(self.ocr_history[camera_id]) > self.max_history_size:
                excess = len(self.ocr_history[camera_id]) - self.max_history_size
                self.ocr_history[camera_id] = self.ocr_history[camera_id][excess:]
            
            # 更新統計資訊
            self.performance_stats['total_processed'] += len(ocr_results)
            
            total_confidence = 0.0
            total_processing_time = 0.0
            
            for result in ocr_results:
                if result.recognized_text.strip():
                    self.performance_stats['successful_recognitions'] += 1
                else:
                    self.performance_stats['failed_recognitions'] += 1
                
                total_confidence += result.confidence
                total_processing_time += result.processing_time
            
            # 更新平均值
            if len(ocr_results) > 0:
                current_avg_confidence = total_confidence / len(ocr_results)
                current_avg_time = total_processing_time / len(ocr_results)
                
                # 使用移動平均
                alpha = 0.1  # 學習率
                self.performance_stats['average_confidence'] = (
                    (1 - alpha) * self.performance_stats['average_confidence'] +
                    alpha * current_avg_confidence
                )
                self.performance_stats['average_processing_time'] = (
                    (1 - alpha) * self.performance_stats['average_processing_time'] +
                    alpha * current_avg_time
                )
                
        except Exception as e:
            self.logger.error(f"更新歷史記錄和統計失敗: {e}")
    
    def get_ocr_statistics(self, camera_id: str) -> Dict[str, Any]:
        """
        取得 OCR 統計資訊
        
        Args:
            camera_id: 攝影機 ID
            
        Returns:
            Dict[str, Any]: 統計資訊
        """
        try:
            history = self.ocr_history.get(camera_id, [])
            if not history:
                return {'total_ocr_results': 0, 'success_rate': 0.0, 'average_confidence': 0.0}
            
            successful_count = len([r for r in history if r.recognized_text.strip()])
            total_confidence = sum(r.confidence for r in history)
            
            return {
                'total_ocr_results': len(history),
                'successful_results': successful_count,
                'success_rate': successful_count / len(history),
                'average_confidence': total_confidence / len(history),
                'latest_ocr': history[-1].timestamp.isoformat() if history else None
            }
            
        except Exception as e:
            self.logger.error(f"取得 OCR 統計失敗: {e}")
            return {'error': str(e)}
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """取得效能摘要"""
        try:
            summary = {
                'performance_stats': self.performance_stats.copy(),
                'camera_statistics': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # 計算成功率
            total_processed = self.performance_stats['total_processed']
            if total_processed > 0:
                summary['performance_stats']['success_rate'] = (
                    self.performance_stats['successful_recognitions'] / total_processed
                )
            else:
                summary['performance_stats']['success_rate'] = 0.0
            
            # 各攝影機統計
            for camera_id in self.ocr_history.keys():
                summary['camera_statistics'][camera_id] = self.get_ocr_statistics(camera_id)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"取得效能摘要失敗: {e}")
            return {'error': str(e)}


# 建立全域 OCRAgent 實例的工廠函數
def create_ocr_agent() -> OCRAgent:
    """建立 OCRAgent 實例"""
    return OCRAgent()