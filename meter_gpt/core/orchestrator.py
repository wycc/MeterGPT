"""
MeterGPT 系統核心協調器 (Orchestrator)
負責管理所有代理人的協作流程，實作完整的儀器讀值 SOP 流程
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from enum import Enum
import logging
from contextlib import asynccontextmanager
import time

from metagpt.agent import Agent
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.environment import Environment

from ..models.messages import (
    StreamFrame, ProcessingResult, ProcessingStatus, FailureType,
    QualityReport, DetectionResult, OCRResult, ValidationResult,
    FallbackDecision, VLMResponse, SystemStatus, AgentMessage,
    CameraInfo, InstrumentType
)
from ..core.config import get_config, MeterGPTConfig
from ..utils.logger import get_logger, log_agent_action
from ..agents.stream_manager import StreamManager
from ..agents.quality_assessor import QualityAssessor
from ..agents.detection_agent import DetectionAgent
from ..agents.ocr_agent import OCRAgent
from ..agents.validation_agent import ValidationAgent
from ..agents.fallback_agent import FallbackAgent


class ProcessingStage(str, Enum):
    """處理階段枚舉"""
    STREAM_CAPTURE = "stream_capture"
    QUALITY_ASSESSMENT = "quality_assessment"
    INSTRUMENT_DETECTION = "instrument_detection"
    OCR_PROCESSING = "ocr_processing"
    RESULT_VALIDATION = "result_validation"
    FALLBACK_PROCESSING = "fallback_processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OrchestratorAction(Action):
    """協調器主要動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("OrchestratorAction")
    
    async def run(self, frame_data: Dict[str, Any]) -> ProcessingResult:
        """
        執行完整的處理流程
        
        Args:
            frame_data: 影像幀資料
            
        Returns:
            ProcessingResult: 處理結果
        """
        # 這個方法會被 Orchestrator 覆寫
        pass


class MeterGPTOrchestrator(Role):
    """MeterGPT 系統協調器"""
    
    def __init__(self, config: Optional[MeterGPTConfig] = None):
        """
        初始化協調器
        
        Args:
            config: 系統配置
        """
        super().__init__(name="MeterGPTOrchestrator", profile="系統協調器")
        
        self.config = config or get_config()
        self.logger = get_logger("MeterGPTOrchestrator")
        
        # 系統狀態
        self.is_running = False
        self.processing_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.system.queue_size if self.config else 100
        )
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.system_metrics = {
            "total_processed": 0,
            "success_count": 0,
            "failure_count": 0,
            "average_processing_time": 0.0,
            "last_processing_time": 0.0
        }
        
        # 代理人實例
        self.agents: Dict[str, Agent] = {}
        self.environment = Environment()
        
        # 初始化代理人
        self._initialize_agents()
        
        # 設置動作
        self._set_actions([OrchestratorAction()])
    
    def _initialize_agents(self):
        """初始化所有代理人"""
        try:
            # 初始化各個代理人
            self.agents["stream_manager"] = StreamManager(config=self.config)
            self.agents["quality_assessor"] = QualityAssessor(config=self.config)
            self.agents["detection_agent"] = DetectionAgent(config=self.config)
            self.agents["ocr_agent"] = OCRAgent(config=self.config)
            self.agents["validation_agent"] = ValidationAgent(config=self.config)
            self.agents["fallback_agent"] = FallbackAgent(config=self.config)
            
            # 將代理人加入環境
            for agent in self.agents.values():
                self.environment.add_role(agent)
            
            self.logger.info("所有代理人初始化完成")
            
        except Exception as e:
            self.logger.error(f"代理人初始化失敗: {e}")
            raise
    
    async def start_system(self):
        """啟動系統"""
        if self.is_running:
            self.logger.warning("系統已經在運行中")
            return
        
        try:
            self.is_running = True
            self.logger.info("MeterGPT 系統啟動中...")
            
            # 啟動處理任務
            processing_task = asyncio.create_task(self._processing_loop())
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # 等待任務完成
            await asyncio.gather(processing_task, monitoring_task)
            
        except Exception as e:
            self.logger.error(f"系統啟動失敗: {e}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """停止系統"""
        if not self.is_running:
            return
        
        self.logger.info("MeterGPT 系統停止中...")
        self.is_running = False
        
        # 取消所有活躍任務
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.active_tasks.clear()
        self.logger.info("MeterGPT 系統已停止")
    
    async def process_frame(self, camera_id: str, frame_data: bytes, 
                          metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        處理單一影像幀
        
        Args:
            camera_id: 攝影機 ID
            frame_data: 影像資料
            metadata: 額外元資料
            
        Returns:
            ProcessingResult: 處理結果
        """
        frame_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 建立處理結果物件
        result = ProcessingResult(
            frame_id=frame_id,
            camera_id=camera_id,
            status=ProcessingStatus.PROCESSING,
            processing_pipeline=[]
        )
        
        try:
            self.logger.info(f"開始處理影像幀: {frame_id}")
            
            # 建立串流幀物件
            camera_info = self._get_camera_info(camera_id)
            stream_frame = StreamFrame(
                frame_id=frame_id,
                camera_info=camera_info,
                frame_data=frame_data,
                frame_shape=(0, 0, 3),  # 會在後續步驟中更新
                metadata=metadata or {}
            )
            
            # 執行完整處理流程
            result = await self._execute_processing_pipeline(stream_frame, result)
            
            # 更新系統指標
            processing_time = time.time() - start_time
            self._update_metrics(result.status == ProcessingStatus.SUCCESS, processing_time)
            
            result.processing_time = processing_time
            self.logger.info(f"影像幀處理完成: {frame_id}, 狀態: {result.status}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"處理影像幀時發生錯誤: {e}")
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            return result
    
    async def _execute_processing_pipeline(self, stream_frame: StreamFrame, 
                                         result: ProcessingResult) -> ProcessingResult:
        """
        執行完整的處理管線
        
        Args:
            stream_frame: 串流影像幀
            result: 處理結果物件
            
        Returns:
            ProcessingResult: 更新後的處理結果
        """
        current_stage = ProcessingStage.QUALITY_ASSESSMENT
        
        try:
            # 階段 1: 品質評估
            result.processing_pipeline.append(current_stage.value)
            quality_report = await self._execute_quality_assessment(stream_frame)
            result.quality_report = quality_report
            
            if not quality_report.is_acceptable:
                # 品質不佳，觸發備援流程
                return await self._handle_fallback(
                    stream_frame, result, FailureType.LOW_QUALITY, 
                    {"quality_report": quality_report}
                )
            
            # 階段 2: 儀器偵測
            current_stage = ProcessingStage.INSTRUMENT_DETECTION
            result.processing_pipeline.append(current_stage.value)
            detection_result = await self._execute_instrument_detection(stream_frame)
            result.detection_result = detection_result
            
            if detection_result.confidence < self.config.detection_model.confidence_threshold:
                # 偵測失敗，觸發備援流程
                return await self._handle_fallback(
                    stream_frame, result, FailureType.DETECTION_FAILED,
                    {"detection_result": detection_result}
                )
            
            # 階段 3: OCR 處理
            current_stage = ProcessingStage.OCR_PROCESSING
            result.processing_pipeline.append(current_stage.value)
            ocr_results = await self._execute_ocr_processing(stream_frame, detection_result)
            result.ocr_results = ocr_results
            
            if not ocr_results or all(ocr.confidence < self.config.ocr.confidence_threshold for ocr in ocr_results):
                # OCR 失敗，觸發備援流程
                return await self._handle_fallback(
                    stream_frame, result, FailureType.OCR_FAILED,
                    {"ocr_results": ocr_results}
                )
            
            # 階段 4: 結果驗證
            current_stage = ProcessingStage.RESULT_VALIDATION
            result.processing_pipeline.append(current_stage.value)
            validation_result = await self._execute_validation(ocr_results)
            result.validation_result = validation_result
            
            if not validation_result.is_valid:
                # 驗證失敗，觸發備援流程
                return await self._handle_fallback(
                    stream_frame, result, FailureType.VALIDATION_FAILED,
                    {"validation_result": validation_result}
                )
            
            # 階段 5: 完成處理
            current_stage = ProcessingStage.COMPLETED
            result.processing_pipeline.append(current_stage.value)
            result.status = ProcessingStatus.SUCCESS
            
            # 提取最終讀值
            result.final_reading = self._extract_final_reading(ocr_results, validation_result)
            result.confidence = self._calculate_final_confidence(
                quality_report, detection_result, ocr_results, validation_result
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"處理超時，當前階段: {current_stage}")
            return await self._handle_fallback(
                stream_frame, result, FailureType.TIMEOUT,
                {"current_stage": current_stage.value}
            )
        except Exception as e:
            self.logger.error(f"處理管線執行失敗，當前階段: {current_stage}, 錯誤: {e}")
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            return result
    
    async def _execute_quality_assessment(self, stream_frame: StreamFrame) -> QualityReport:
        """執行品質評估"""
        quality_assessor = self.agents["quality_assessor"]
        message = Message(content={"stream_frame": stream_frame})
        
        response = await quality_assessor.handle(message)
        return response.content["quality_report"]
    
    async def _execute_instrument_detection(self, stream_frame: StreamFrame) -> DetectionResult:
        """執行儀器偵測"""
        detection_agent = self.agents["detection_agent"]
        message = Message(content={"stream_frame": stream_frame})
        
        response = await detection_agent.handle(message)
        return response.content["detection_result"]
    
    async def _execute_ocr_processing(self, stream_frame: StreamFrame, 
                                    detection_result: DetectionResult) -> List[OCRResult]:
        """執行 OCR 處理"""
        ocr_agent = self.agents["ocr_agent"]
        message = Message(content={
            "stream_frame": stream_frame,
            "detection_result": detection_result
        })
        
        response = await ocr_agent.handle(message)
        return response.content["ocr_results"]
    
    async def _execute_validation(self, ocr_results: List[OCRResult]) -> ValidationResult:
        """執行結果驗證"""
        validation_agent = self.agents["validation_agent"]
        message = Message(content={"ocr_results": ocr_results})
        
        response = await validation_agent.handle(message)
        return response.content["validation_result"]
    
    async def _handle_fallback(self, stream_frame: StreamFrame, result: ProcessingResult,
                             failure_type: FailureType, context: Dict[str, Any]) -> ProcessingResult:
        """
        處理備援流程
        
        Args:
            stream_frame: 串流影像幀
            result: 當前處理結果
            failure_type: 失敗類型
            context: 上下文資訊
            
        Returns:
            ProcessingResult: 備援處理結果
        """
        try:
            result.processing_pipeline.append(ProcessingStage.FALLBACK_PROCESSING.value)
            
            # 執行備援決策
            fallback_agent = self.agents["fallback_agent"]
            message = Message(content={
                "failure_type": failure_type,
                "context": context,
                "stream_frame": stream_frame
            })
            
            response = await fallback_agent.handle(message)
            fallback_decision = response.content["fallback_decision"]
            result.fallback_decision = fallback_decision
            
            # 根據備援決策執行相應動作
            if fallback_decision.recommended_action.value == "use_vlm":
                vlm_response = await self._execute_vlm_processing(stream_frame, context)
                result.vlm_response = vlm_response
                
                if vlm_response and vlm_response.confidence >= self.config.vlm.confidence_threshold:
                    result.final_reading = vlm_response.response_text
                    result.confidence = vlm_response.confidence
                    result.status = ProcessingStatus.SUCCESS
                else:
                    result.status = ProcessingStatus.FAILED
                    result.error_message = "VLM 處理失敗或信心度不足"
            
            elif fallback_decision.recommended_action.value == "switch_camera":
                # 切換攝影機邏輯
                if fallback_decision.alternative_camera_id:
                    result.status = ProcessingStatus.RETRY
                    result.error_message = f"建議切換至攝影機: {fallback_decision.alternative_camera_id}"
                else:
                    result.status = ProcessingStatus.FAILED
                    result.error_message = "沒有可用的備選攝影機"
            
            elif fallback_decision.recommended_action.value == "manual_review":
                result.status = ProcessingStatus.FAILED
                result.error_message = "需要人工審核"
            
            else:
                result.status = ProcessingStatus.FAILED
                result.error_message = f"未知的備援動作: {fallback_decision.recommended_action}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"備援處理失敗: {e}")
            result.status = ProcessingStatus.FAILED
            result.error_message = f"備援處理失敗: {str(e)}"
            return result
    
    async def _execute_vlm_processing(self, stream_frame: StreamFrame, 
                                    context: Dict[str, Any]) -> Optional[VLMResponse]:
        """執行 VLM 處理"""
        try:
            # 這裡應該實作 VLM 處理邏輯
            # 由於 VLM 處理比較複雜，這裡提供基本框架
            self.logger.info("執行 VLM 處理...")
            
            # 模擬 VLM 回應
            vlm_response = VLMResponse(
                request_id=str(uuid.uuid4()),
                response_text="模擬 VLM 讀值結果",
                confidence=0.85,
                processing_time=2.5,
                model_name=self.config.vlm.model_name,
                token_usage={"prompt_tokens": 100, "completion_tokens": 50}
            )
            
            return vlm_response
            
        except Exception as e:
            self.logger.error(f"VLM 處理失敗: {e}")
            return None
    
    def _extract_final_reading(self, ocr_results: List[OCRResult], 
                             validation_result: ValidationResult) -> str:
        """提取最終讀值"""
        if not ocr_results:
            return ""
        
        # 選擇信心度最高的 OCR 結果
        best_ocr = max(ocr_results, key=lambda x: x.confidence)
        return best_ocr.recognized_text
    
    def _calculate_final_confidence(self, quality_report: QualityReport,
                                  detection_result: DetectionResult,
                                  ocr_results: List[OCRResult],
                                  validation_result: ValidationResult) -> float:
        """計算最終信心度"""
        if not ocr_results:
            return 0.0
        
        # 綜合各階段的信心度
        quality_confidence = quality_report.metrics.overall_score
        detection_confidence = detection_result.confidence
        ocr_confidence = max(ocr.confidence for ocr in ocr_results)
        validation_confidence = validation_result.validation_score
        
        # 加權平均
        weights = [0.2, 0.3, 0.3, 0.2]  # 品質、偵測、OCR、驗證
        confidences = [quality_confidence, detection_confidence, ocr_confidence, validation_confidence]
        
        final_confidence = sum(w * c for w, c in zip(weights, confidences))
        return min(max(final_confidence, 0.0), 1.0)
    
    def _get_camera_info(self, camera_id: str) -> CameraInfo:
        """取得攝影機資訊"""
        if self.config and self.config.cameras:
            for camera_config in self.config.cameras:
                if camera_config.camera_id == camera_id:
                    return CameraInfo(
                        camera_id=camera_config.camera_id,
                        camera_name=camera_config.camera_name,
                        position=camera_config.position,
                        is_active=camera_config.is_active,
                        is_primary=camera_config.is_primary
                    )
        
        # 預設攝影機資訊
        return CameraInfo(
            camera_id=camera_id,
            camera_name=f"Camera_{camera_id}",
            position=(0.0, 0.0, 0.0),
            is_active=True,
            is_primary=False
        )
    
    def _update_metrics(self, success: bool, processing_time: float):
        """更新系統指標"""
        self.system_metrics["total_processed"] += 1
        if success:
            self.system_metrics["success_count"] += 1
        else:
            self.system_metrics["failure_count"] += 1
        
        # 更新平均處理時間
        total = self.system_metrics["total_processed"]
        current_avg = self.system_metrics["average_processing_time"]
        self.system_metrics["average_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        self.system_metrics["last_processing_time"] = processing_time
    
    async def _processing_loop(self):
        """處理循環"""
        while self.is_running:
            try:
                # 從佇列取得處理任務
                task_data = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=1.0
                )
                
                # 建立處理任務
                task_id = str(uuid.uuid4())
                task = asyncio.create_task(
                    self.process_frame(**task_data)
                )
                self.active_tasks[task_id] = task
                
                # 等待任務完成並清理
                try:
                    await task
                finally:
                    self.active_tasks.pop(task_id, None)
                    
            except asyncio.TimeoutError:
                # 佇列為空，繼續等待
                continue
            except Exception as e:
                self.logger.error(f"處理循環錯誤: {e}")
    
    async def _monitoring_loop(self):
        """監控循環"""
        while self.is_running:
            try:
                # 每 30 秒輸出系統狀態
                await asyncio.sleep(30)
                
                status = self.get_system_status()
                self.logger.info(f"系統狀態 - 處理總數: {status.processing_queue_size}, "
                               f"成功率: {status.success_rate:.2%}, "
                               f"平均處理時間: {status.average_processing_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"監控循環錯誤: {e}")
    
    def get_system_status(self) -> SystemStatus:
        """取得系統狀態"""
        total_processed = self.system_metrics["total_processed"]
        success_rate = (
            self.system_metrics["success_count"] / total_processed 
            if total_processed > 0 else 0.0
        )
        
        return SystemStatus(
            active_cameras=[cam.camera_id for cam in self.config.cameras] if self.config else [],
            processing_queue_size=self.processing_queue.qsize(),
            average_processing_time=self.system_metrics["average_processing_time"],
            success_rate=success_rate,
            error_count=self.system_metrics["failure_count"],
            system_health=min(success_rate + 0.1, 1.0)  # 簡單的健康度計算
        )
    
    async def add_processing_task(self, camera_id: str, frame_data: bytes, 
                                metadata: Optional[Dict[str, Any]] = None):
        """
        添加處理任務到佇列
        
        Args:
            camera_id: 攝影機 ID
            frame_data: 影像資料
            metadata: 額外元資料
        """
        task_data = {
            "camera_id": camera_id,
            "frame_data": frame_data,
            "metadata": metadata
        }
        
        try:
            await self.processing_queue.put(task_data)
            self.logger.debug(f"處理任務已加入佇列: {camera_id}")
        except asyncio.QueueFull:
            self.logger.warning("處理佇列已滿，丟棄任務")
    
    @asynccontextmanager
    async def processing_context(self):
        """處理上下文管理器"""
        try:
            await self.start_system()
            yield self
        finally:
            await self.stop_system()


# 全域協調器實例
orchestrator: Optional[MeterGPTOrchestrator] = None


def get_orchestrator(config: Optional[MeterGPTConfig] = None) -> MeterGPTOrchestrator:
    """取得全域協調器實例"""
    global orchestrator
    if orchestrator is None:
        orchestrator = MeterGPTOrchestrator(config)
    return orchestrator


async def initialize_system(config: Optional[MeterGPTConfig] = None) -> MeterGPTOrchestrator:
    """初始化系統"""
    orch = get_orchestrator(config)
    await orch.start_system()
    return orch


async def shutdown_system():
    """關閉系統"""
    global orchestrator
    if orchestrator:
        await orchestrator.stop_system()
        orchestrator = None