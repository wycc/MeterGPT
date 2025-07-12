"""
FallbackAgent 代理人
負責備援決策邏輯，當主要流程失敗時提供智慧備援方案
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from metagpt.agent import Agent
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message

from ..models.messages import (
    FallbackDecision, FallbackAction, FailureType, QualityReport, 
    ValidationResult, OCRResult, VLMRequest, VLMResponse, CameraInfo
)
from ..core.config import get_config
from ..utils.logger import get_logger, log_agent_action


class FallbackTrigger(Enum):
    """備援觸發條件"""
    LOW_QUALITY = "low_quality"
    DETECTION_FAILED = "detection_failed"
    OCR_FAILED = "ocr_failed"
    VALIDATION_FAILED = "validation_failed"
    TIMEOUT = "timeout"
    MANUAL_REQUEST = "manual_request"


class FallbackDecisionAction(Action):
    """備援決策動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("FallbackDecisionAction")
        self.config = get_config()
    
    async def run(self, trigger: FallbackTrigger, context: Dict[str, Any]) -> FallbackDecision:
        """
        執行備援決策
        
        Args:
            trigger: 觸發條件
            context: 上下文資訊
            
        Returns:
            FallbackDecision: 備援決策
        """
        try:
            frame_id = context.get('frame_id', 'unknown')
            
            # 分析觸發原因
            failure_type = self._map_trigger_to_failure_type(trigger)
            
            # 根據觸發條件和上下文決定備援策略
            recommended_action, alternative_camera_id = await self._analyze_fallback_strategy(
                trigger, context
            )
            
            # 取得配置參數
            fallback_config = self.config.fallback if self.config else None
            confidence_threshold = fallback_config.camera_switch_threshold if fallback_config else 0.4
            max_retries = fallback_config.max_retries if fallback_config else 3
            
            # 建立備援決策
            fallback_decision = FallbackDecision(
                frame_id=frame_id,
                trigger_reason=failure_type,
                recommended_action=recommended_action,
                alternative_camera_id=alternative_camera_id,
                confidence_threshold=confidence_threshold,
                max_retries=max_retries,
                priority=self._calculate_priority(trigger, context)
            )
            
            self.logger.log_fallback_decision(
                frame_id,
                failure_type,
                recommended_action.value
            )
            
            return fallback_decision
            
        except Exception as e:
            self.logger.error(f"備援決策失敗: {e}")
            # 返回預設的人工審核決策
            return FallbackDecision(
                frame_id=context.get('frame_id', 'unknown'),
                trigger_reason=FailureType.UNKNOWN_ERROR,
                recommended_action=FallbackAction.MANUAL_REVIEW,
                confidence_threshold=0.1,
                max_retries=0,
                priority=1
            )
    
    def _map_trigger_to_failure_type(self, trigger: FallbackTrigger) -> FailureType:
        """將觸發條件映射到失敗類型"""
        mapping = {
            FallbackTrigger.LOW_QUALITY: FailureType.LOW_QUALITY,
            FallbackTrigger.DETECTION_FAILED: FailureType.DETECTION_FAILED,
            FallbackTrigger.OCR_FAILED: FailureType.OCR_FAILED,
            FallbackTrigger.VALIDATION_FAILED: FailureType.VALIDATION_FAILED,
            FallbackTrigger.TIMEOUT: FailureType.TIMEOUT,
            FallbackTrigger.MANUAL_REQUEST: FailureType.UNKNOWN_ERROR
        }
        return mapping.get(trigger, FailureType.UNKNOWN_ERROR)
    
    async def _analyze_fallback_strategy(self, trigger: FallbackTrigger, 
                                       context: Dict[str, Any]) -> Tuple[FallbackAction, Optional[str]]:
        """分析備援策略"""
        try:
            current_camera_id = context.get('camera_id')
            available_cameras = context.get('available_cameras', [])
            quality_scores = context.get('quality_scores', {})
            retry_count = context.get('retry_count', 0)
            
            fallback_config = self.config.fallback if self.config else None
            max_retries = fallback_config.max_retries if fallback_config else 3
            
            # 策略決策邏輯
            if trigger == FallbackTrigger.LOW_QUALITY:
                return await self._handle_low_quality(
                    current_camera_id, available_cameras, quality_scores
                )
            
            elif trigger == FallbackTrigger.DETECTION_FAILED:
                if retry_count < max_retries:
                    return FallbackAction.RETRY_OCR, None
                else:
                    return await self._handle_detection_failure(
                        current_camera_id, available_cameras, quality_scores
                    )
            
            elif trigger == FallbackTrigger.OCR_FAILED:
                return await self._handle_ocr_failure(
                    current_camera_id, available_cameras, quality_scores, retry_count, max_retries
                )
            
            elif trigger == FallbackTrigger.VALIDATION_FAILED:
                return await self._handle_validation_failure(
                    current_camera_id, available_cameras, quality_scores
                )
            
            elif trigger == FallbackTrigger.TIMEOUT:
                return FallbackAction.MANUAL_REVIEW, None
            
            else:
                return FallbackAction.MANUAL_REVIEW, None
                
        except Exception as e:
            self.logger.error(f"分析備援策略失敗: {e}")
            return FallbackAction.MANUAL_REVIEW, None
    
    async def _handle_low_quality(self, current_camera_id: str, 
                                available_cameras: List[str], 
                                quality_scores: Dict[str, float]) -> Tuple[FallbackAction, Optional[str]]:
        """處理低品質情況"""
        try:
            # 尋找品質更好的攝影機
            best_camera = self._find_best_quality_camera(
                available_cameras, quality_scores, current_camera_id
            )
            
            if best_camera:
                return FallbackAction.SWITCH_CAMERA, best_camera
            
            # 如果沒有更好的攝影機，嘗試 PTZ 調整
            fallback_config = self.config.fallback if self.config else None
            if fallback_config and fallback_config.ptz_adjustment_enabled:
                return FallbackAction.ADJUST_PTZ, current_camera_id
            
            # 最後使用 VLM
            return FallbackAction.USE_VLM, None
            
        except Exception as e:
            self.logger.error(f"處理低品質失敗: {e}")
            return FallbackAction.MANUAL_REVIEW, None
    
    async def _handle_detection_failure(self, current_camera_id: str,
                                      available_cameras: List[str],
                                      quality_scores: Dict[str, float]) -> Tuple[FallbackAction, Optional[str]]:
        """處理偵測失敗情況"""
        try:
            # 優先切換攝影機
            best_camera = self._find_best_quality_camera(
                available_cameras, quality_scores, current_camera_id
            )
            
            if best_camera:
                return FallbackAction.SWITCH_CAMERA, best_camera
            
            # 使用 VLM 作為備援
            return FallbackAction.USE_VLM, None
            
        except Exception as e:
            self.logger.error(f"處理偵測失敗失敗: {e}")
            return FallbackAction.MANUAL_REVIEW, None
    
    async def _handle_ocr_failure(self, current_camera_id: str,
                                available_cameras: List[str],
                                quality_scores: Dict[str, float],
                                retry_count: int, max_retries: int) -> Tuple[FallbackAction, Optional[str]]:
        """處理 OCR 失敗情況"""
        try:
            # 如果重試次數未達上限，先重試
            if retry_count < max_retries:
                return FallbackAction.RETRY_OCR, None
            
            # 嘗試切換攝影機
            best_camera = self._find_best_quality_camera(
                available_cameras, quality_scores, current_camera_id
            )
            
            if best_camera:
                return FallbackAction.SWITCH_CAMERA, best_camera
            
            # 使用 VLM
            return FallbackAction.USE_VLM, None
            
        except Exception as e:
            self.logger.error(f"處理 OCR 失敗失敗: {e}")
            return FallbackAction.MANUAL_REVIEW, None
    
    async def _handle_validation_failure(self, current_camera_id: str,
                                       available_cameras: List[str],
                                       quality_scores: Dict[str, float]) -> Tuple[FallbackAction, Optional[str]]:
        """處理驗證失敗情況"""
        try:
            # 驗證失敗通常表示讀值不合理，優先使用 VLM
            return FallbackAction.USE_VLM, None
            
        except Exception as e:
            self.logger.error(f"處理驗證失敗失敗: {e}")
            return FallbackAction.MANUAL_REVIEW, None
    
    def _find_best_quality_camera(self, available_cameras: List[str],
                                quality_scores: Dict[str, float],
                                exclude_camera: str = None) -> Optional[str]:
        """尋找品質最佳的攝影機"""
        try:
            fallback_config = self.config.fallback if self.config else None
            threshold = fallback_config.camera_switch_threshold if fallback_config else 0.4
            
            best_camera = None
            best_score = threshold
            
            for camera_id in available_cameras:
                if camera_id == exclude_camera:
                    continue
                
                score = quality_scores.get(camera_id, 0.0)
                if score > best_score:
                    best_score = score
                    best_camera = camera_id
            
            return best_camera
            
        except Exception as e:
            self.logger.error(f"尋找最佳品質攝影機失敗: {e}")
            return None
    
    def _calculate_priority(self, trigger: FallbackTrigger, context: Dict[str, Any]) -> int:
        """計算優先級"""
        try:
            # 根據觸發條件設定優先級
            priority_mapping = {
                FallbackTrigger.TIMEOUT: 1,  # 最高優先級
                FallbackTrigger.DETECTION_FAILED: 2,
                FallbackTrigger.OCR_FAILED: 3,
                FallbackTrigger.VALIDATION_FAILED: 4,
                FallbackTrigger.LOW_QUALITY: 5,
                FallbackTrigger.MANUAL_REQUEST: 6
            }
            
            return priority_mapping.get(trigger, 10)
            
        except Exception as e:
            self.logger.error(f"計算優先級失敗: {e}")
            return 10


class VLMFallbackAction(Action):
    """VLM 備援動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("VLMFallbackAction")
        self.config = get_config()
    
    async def run(self, image_data: bytes, frame_id: str, 
                 prompt: str = None) -> VLMResponse:
        """
        執行 VLM 備援識別
        
        Args:
            image_data: 影像資料
            frame_id: 影像幀識別碼
            prompt: 提示詞
            
        Returns:
            VLMResponse: VLM 回應
        """
        try:
            # 建立 VLM 請求
            request_id = f"vlm_{frame_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if not prompt:
                prompt = "請識別這個儀器上顯示的數字或文字。只回答看到的數值，不要包含其他說明。"
            
            vlm_config = self.config.vlm if self.config else None
            max_tokens = vlm_config.max_tokens if vlm_config else 500
            temperature = vlm_config.temperature if vlm_config else 0.1
            
            vlm_request = VLMRequest(
                request_id=request_id,
                frame_id=frame_id,
                image_data=image_data,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 模擬 VLM 處理 (實際應用中應該調用真實的 VLM API)
            vlm_response = await self._simulate_vlm_processing(vlm_request)
            
            self.logger.info(
                f"VLM 備援識別完成",
                frame_id=frame_id,
                confidence=vlm_response.confidence
            )
            
            return vlm_response
            
        except Exception as e:
            self.logger.error(f"VLM 備援識別失敗: {e}")
            return VLMResponse(
                request_id=f"error_{frame_id}",
                response_text="",
                confidence=0.0,
                processing_time=0.0,
                model_name="unknown",
                token_usage={}
            )
    
    async def _simulate_vlm_processing(self, request: VLMRequest) -> VLMResponse:
        """模擬 VLM 處理 (實際應用中應該替換為真實的 VLM API 調用)"""
        import time
        import random
        
        start_time = time.time()
        
        # 模擬處理延遲
        await asyncio.sleep(0.5)
        
        # 模擬回應
        simulated_responses = [
            "123.45", "67.89", "0.00", "ERROR", "---", "88.88"
        ]
        
        response_text = random.choice(simulated_responses)
        confidence = random.uniform(0.7, 0.95)
        processing_time = time.time() - start_time
        
        return VLMResponse(
            request_id=request.request_id,
            response_text=response_text,
            confidence=confidence,
            processing_time=processing_time,
            model_name="gpt-4-vision-preview",
            token_usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        )


class PTZControlAction(Action):
    """PTZ 控制動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("PTZControlAction")
        self.config = get_config()
    
    async def run(self, camera_id: str, adjustment_type: str = "scan") -> Dict[str, Any]:
        """
        執行 PTZ 控制
        
        Args:
            camera_id: 攝影機 ID
            adjustment_type: 調整類型 (scan/focus/position)
            
        Returns:
            Dict[str, Any]: 控制結果
        """
        try:
            fallback_config = self.config.fallback if self.config else None
            
            if not fallback_config or not fallback_config.ptz_adjustment_enabled:
                return {
                    "success": False,
                    "message": "PTZ 調整未啟用"
                }
            
            # 模擬 PTZ 控制
            result = await self._simulate_ptz_control(camera_id, adjustment_type, fallback_config)
            
            self.logger.info(
                f"PTZ 控制完成",
                camera_id=camera_id,
                adjustment_type=adjustment_type,
                success=result["success"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"PTZ 控制失敗: {e}")
            return {
                "success": False,
                "message": f"PTZ 控制異常: {str(e)}"
            }
    
    async def _simulate_ptz_control(self, camera_id: str, adjustment_type: str, 
                                  config: Any) -> Dict[str, Any]:
        """模擬 PTZ 控制"""
        import random
        
        # 模擬控制延遲
        await asyncio.sleep(1.0)
        
        # 模擬成功率
        success = random.random() > 0.2  # 80% 成功率
        
        if success:
            return {
                "success": True,
                "message": f"PTZ {adjustment_type} 調整成功",
                "new_position": {
                    "pan": random.uniform(-30, 30),
                    "tilt": random.uniform(-15, 15),
                    "zoom": random.uniform(1.0, 3.0)
                }
            }
        else:
            return {
                "success": False,
                "message": f"PTZ {adjustment_type} 調整失敗"
            }


class FallbackAgent(Role):
    """備援代理人"""
    
    def __init__(self, name: str = "FallbackAgent", **kwargs):
        """
        初始化備援代理人
        
        Args:
            name: 代理人名稱
        """
        super().__init__(name=name, **kwargs)
        
        # 設置動作
        self._init_actions([
            FallbackDecisionAction(),
            VLMFallbackAction(),
            PTZControlAction()
        ])
        
        self.logger = get_logger("FallbackAgent")
        self.config = get_config()
        
        # 備援歷史記錄
        self.fallback_history: List[FallbackDecision] = []
        self.vlm_history: List[VLMResponse] = []
        self.max_history_size = 100
    
    @log_agent_action("FallbackAgent")
    async def _act(self) -> Message:
        """執行代理人動作"""
        try:
            # 從訊息中取得觸發資訊
            trigger = FallbackTrigger.MANUAL_REQUEST
            context = {}
            
            for msg in self.rc.memory.get():
                if hasattr(msg, 'content') and 'fallback_trigger' in str(msg.content):
                    # 這裡應該解析訊息內容取得觸發資訊
                    # 為了示範，我們跳過實際解析
                    pass
            
            # 執行備援決策
            fallback_decision = await self.make_fallback_decision(trigger, context)
            
            # 建立回應訊息
            message_content = {
                'action': 'fallback_decision',
                'trigger': trigger.value,
                'recommended_action': fallback_decision.recommended_action.value,
                'timestamp': datetime.now().isoformat()
            }
            
            return Message(
                content=str(message_content),
                role=self.profile,
                cause_by=FallbackDecisionAction
            )
            
        except Exception as e:
            self.logger.error(f"代理人動作執行失敗: {e}")
            return Message(
                content=f"Error: {str(e)}",
                role=self.profile,
                cause_by=FallbackDecisionAction
            )
    
    async def make_fallback_decision(self, trigger: FallbackTrigger, 
                                   context: Dict[str, Any]) -> FallbackDecision:
        """
        制定備援決策
        
        Args:
            trigger: 觸發條件
            context: 上下文資訊
            
        Returns:
            FallbackDecision: 備援決策
        """
        try:
            decision_action = FallbackDecisionAction()
            fallback_decision = await decision_action.run(trigger, context)
            
            # 記錄決策歷史
            self.fallback_history.append(fallback_decision)
            if len(self.fallback_history) > self.max_history_size:
                self.fallback_history.pop(0)
            
            return fallback_decision
            
        except Exception as e:
            self.logger.error(f"制定備援決策失敗: {e}")
            raise
    
    async def execute_vlm_fallback(self, image_data: bytes, frame_id: str,
                                 prompt: str = None) -> VLMResponse:
        """
        執行 VLM 備援
        
        Args:
            image_data: 影像資料
            frame_id: 影像幀識別碼
            prompt: 提示詞
            
        Returns:
            VLMResponse: VLM 回應
        """
        try:
            vlm_action = VLMFallbackAction()
            vlm_response = await vlm_action.run(image_data, frame_id, prompt)
            
            # 記錄 VLM 歷史
            self.vlm_history.append(vlm_response)
            if len(self.vlm_history) > self.max_history_size:
                self.vlm_history.pop(0)
            
            return vlm_response
            
        except Exception as e:
            self.logger.error(f"執行 VLM 備援失敗: {e}")
            raise
    
    async def execute_ptz_adjustment(self, camera_id: str, 
                                   adjustment_type: str = "scan") -> Dict[str, Any]:
        """
        執行 PTZ 調整
        
        Args:
            camera_id: 攝影機 ID
            adjustment_type: 調整類型
            
        Returns:
            Dict[str, Any]: 調整結果
        """
        try:
            ptz_action = PTZControlAction()
            result = await ptz_action.run(camera_id, adjustment_type)
            
            return result
            
        except Exception as e:
            self.logger.error(f"執行 PTZ 調整失敗: {e}")
            raise
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """取得備援統計資訊"""
        try:
            if not self.fallback_history:
                return {
                    'total_fallbacks': 0,
                    'action_distribution': {},
                    'trigger_distribution': {}
                }
            
            # 統計動作分布
            action_counts = {}
            trigger_counts = {}
            
            for decision in self.fallback_history:
                action = decision.recommended_action.value
                trigger = decision.trigger_reason.value
                
                action_counts[action] = action_counts.get(action, 0) + 1
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
            
            return {
                'total_fallbacks': len(self.fallback_history),
                'action_distribution': action_counts,
                'trigger_distribution': trigger_counts,
                'vlm_usage_count': len(self.vlm_history),
                'latest_fallback': self.fallback_history[-1].timestamp.isoformat() if self.fallback_history else None
            }
            
        except Exception as e:
            self.logger.error(f"取得備援統計失敗: {e}")
            return {'error': str(e)}
    
    async def get_fallback_summary(self) -> Dict[str, Any]:
        """取得備援摘要"""
        try:
            stats = self.get_fallback_statistics()
            
            summary = {
                'statistics': stats,
                'vlm_performance': self._analyze_vlm_performance(),
                'recommendations': self._generate_fallback_recommendations(),
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"取得備援摘要失敗: {e}")
            return {'error': str(e)}
    
    def _analyze_vlm_performance(self) -> Dict[str, Any]:
        """分析 VLM 效能"""
        try:
            if not self.vlm_history:
                return {'no_data': True}
            
            total_confidence = sum(r.confidence for r in self.vlm_history)
            total_time = sum(r.processing_time for r in self.vlm_history)
            
            return {
                'total_requests': len(self.vlm_history),
                'average_confidence': total_confidence / len(self.vlm_history),
                'average_processing_time': total_time / len(self.vlm_history),
                'high_confidence_rate': len([r for r in self.vlm_history if r.confidence > 0.8]) / len(self.vlm_history)
            }
            
        except Exception as e:
            self.logger.error(f"分析 VLM 效能失敗: {e}")
            return {'error': str(e)}
    
    def _generate_fallback_recommendations(self) -> List[str]:
        """生成備援建議"""
        try:
            recommendations = []
            stats = self.get_fallback_statistics()
            
            # 分析備援頻率
            if stats['total_fallbacks'] > 50:
                recommendations.append("備援觸發頻率較高，建議檢查主要流程的穩定性")
            
            # 分析觸發原因
            trigger_dist = stats.get('trigger_distribution', {})
            if trigger_dist.get('low_quality', 0) > 20:
                recommendations.append("影像品質問題頻繁，建議檢查攝影機設定和環境照明")
            
            if trigger_dist.get('ocr_failed', 0) > 15:
                recommendations.append("OCR 失敗頻繁，建議調整 OCR 參數或改善影像前處理")
            
            # 分析 VLM 使用
            vlm_perf = self._analyze_vlm_performance()
            if not vlm_perf.get('no_data', False):
                if vlm_perf.get('average_confidence', 0) < 0.7:
                    recommendations.append("VLM 平均信心度較低，建議優化提示詞或更換模型")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"生成備援建議失敗: {e}")
            return ["備援分析異常，請檢查系統狀態"]


# 建立全域 FallbackAgent 實例的工廠函數
def create_fallback_agent() -> FallbackAgent:
    """建立 FallbackAgent 實例"""
    return FallbackAgent()