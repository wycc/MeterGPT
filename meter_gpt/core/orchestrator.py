"""
MeterGPT 系統核心協調器 (Orchestrator)
基於 MetaGPT 框架的真正代理人協作架構
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging

from metagpt.environment import Environment
from metagpt.roles import Role
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.team import Team

from ..models.messages import (
    StreamFrame, ProcessingResult, ProcessingStatus, FailureType,
    QualityReport, DetectionResult, OCRResult, ValidationResult,
    FallbackDecision, VLMResponse, SystemStatus, AgentMessage,
    CameraInfo, InstrumentType
)
from ..core.config import get_config, MeterGPTConfig
from ..utils.logger import get_logger


class MessageType(str, Enum):
    """訊息類型枚舉"""
    STREAM_FRAME = "stream_frame"
    QUALITY_REPORT = "quality_report"
    DETECTION_RESULT = "detection_result"
    OCR_RESULT = "ocr_result"
    VALIDATION_RESULT = "validation_result"
    FALLBACK_DECISION = "fallback_decision"
    PROCESSING_COMPLETE = "processing_complete"
    SYSTEM_STATUS = "system_status"


class SystemMonitorAction(Action):
    """系統監控動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("SystemMonitorAction")
    
    async def run(self, messages: List[Message]) -> Message:
        """
        監控系統狀態並生成報告
        
        Args:
            messages: 收到的訊息列表
            
        Returns:
            Message: 系統狀態訊息
        """
        try:
            # 分析收到的訊息，生成系統狀態
            status_data = {
                "timestamp": datetime.now(),
                "active_processes": len(messages),
                "message_types": [msg.content.get("type") for msg in messages],
                "system_health": 1.0  # 簡化的健康度計算
            }
            
            return Message(
                content={
                    "type": MessageType.SYSTEM_STATUS,
                    "status": status_data
                },
                role="SystemMonitor"
            )
            
        except Exception as e:
            self.logger.error(f"系統監控失敗: {e}")
            return Message(
                content={
                    "type": MessageType.SYSTEM_STATUS,
                    "error": str(e)
                },
                role="SystemMonitor"
            )


class ProcessingCoordinationAction(Action):
    """處理協調動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("ProcessingCoordinationAction")
        self.processing_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def run(self, messages: List[Message]) -> List[Message]:
        """
        協調處理流程
        
        Args:
            messages: 收到的訊息列表
            
        Returns:
            List[Message]: 協調指令訊息列表
        """
        coordination_messages = []
        
        for message in messages:
            try:
                msg_type = message.content.get("type")
                frame_id = message.content.get("frame_id")
                
                if not frame_id:
                    continue
                
                # 初始化處理會話
                if frame_id not in self.processing_sessions:
                    self.processing_sessions[frame_id] = {
                        "start_time": datetime.now(),
                        "stages": {},
                        "status": ProcessingStatus.PROCESSING
                    }
                
                session = self.processing_sessions[frame_id]
                
                # 根據訊息類型更新處理狀態
                if msg_type == MessageType.STREAM_FRAME:
                    coordination_messages.append(
                        self._create_quality_assessment_request(message)
                    )
                
                elif msg_type == MessageType.QUALITY_REPORT:
                    session["stages"]["quality"] = message.content
                    if message.content.get("is_acceptable", False):
                        coordination_messages.append(
                            self._create_detection_request(message)
                        )
                    else:
                        coordination_messages.append(
                            self._create_fallback_request(message, FailureType.LOW_QUALITY)
                        )
                
                elif msg_type == MessageType.DETECTION_RESULT:
                    session["stages"]["detection"] = message.content
                    if message.content.get("confidence", 0) > 0.5:  # 閾值應該從配置讀取
                        coordination_messages.append(
                            self._create_ocr_request(message)
                        )
                    else:
                        coordination_messages.append(
                            self._create_fallback_request(message, FailureType.DETECTION_FAILED)
                        )
                
                elif msg_type == MessageType.OCR_RESULT:
                    session["stages"]["ocr"] = message.content
                    coordination_messages.append(
                        self._create_validation_request(message)
                    )
                
                elif msg_type == MessageType.VALIDATION_RESULT:
                    session["stages"]["validation"] = message.content
                    if message.content.get("is_valid", False):
                        coordination_messages.append(
                            self._create_completion_message(frame_id, session)
                        )
                    else:
                        coordination_messages.append(
                            self._create_fallback_request(message, FailureType.VALIDATION_FAILED)
                        )
                
                elif msg_type == MessageType.FALLBACK_DECISION:
                    session["stages"]["fallback"] = message.content
                    # 根據備援決策執行相應動作
                    coordination_messages.extend(
                        self._handle_fallback_decision(message)
                    )
                
            except Exception as e:
                self.logger.error(f"處理協調失敗: {e}")
        
        return coordination_messages
    
    def _create_quality_assessment_request(self, stream_message: Message) -> Message:
        """建立品質評估請求"""
        return Message(
            content={
                "type": "quality_assessment_request",
                "stream_frame": stream_message.content.get("stream_frame"),
                "frame_id": stream_message.content.get("frame_id")
            },
            role="Coordinator",
            cause_by=type(self)
        )
    
    def _create_detection_request(self, quality_message: Message) -> Message:
        """建立偵測請求"""
        return Message(
            content={
                "type": "detection_request",
                "stream_frame": quality_message.content.get("stream_frame"),
                "frame_id": quality_message.content.get("frame_id"),
                "quality_report": quality_message.content.get("quality_report")
            },
            role="Coordinator",
            cause_by=type(self)
        )
    
    def _create_ocr_request(self, detection_message: Message) -> Message:
        """建立 OCR 請求"""
        return Message(
            content={
                "type": "ocr_request",
                "stream_frame": detection_message.content.get("stream_frame"),
                "frame_id": detection_message.content.get("frame_id"),
                "detection_result": detection_message.content.get("detection_result")
            },
            role="Coordinator",
            cause_by=type(self)
        )
    
    def _create_validation_request(self, ocr_message: Message) -> Message:
        """建立驗證請求"""
        return Message(
            content={
                "type": "validation_request",
                "frame_id": ocr_message.content.get("frame_id"),
                "ocr_results": ocr_message.content.get("ocr_results")
            },
            role="Coordinator",
            cause_by=type(self)
        )
    
    def _create_fallback_request(self, message: Message, failure_type: FailureType) -> Message:
        """建立備援請求"""
        return Message(
            content={
                "type": "fallback_request",
                "frame_id": message.content.get("frame_id"),
                "failure_type": failure_type,
                "context": message.content
            },
            role="Coordinator",
            cause_by=type(self)
        )
    
    def _create_completion_message(self, frame_id: str, session: Dict[str, Any]) -> Message:
        """建立完成訊息"""
        # 整合所有階段的結果
        result = ProcessingResult(
            frame_id=frame_id,
            camera_id=session["stages"].get("quality", {}).get("camera_id", "unknown"),
            status=ProcessingStatus.SUCCESS,
            quality_report=session["stages"].get("quality"),
            detection_result=session["stages"].get("detection"),
            ocr_results=session["stages"].get("ocr", []),
            validation_result=session["stages"].get("validation"),
            processing_time=(datetime.now() - session["start_time"]).total_seconds()
        )
        
        # 計算最終讀值和信心度
        if session["stages"].get("ocr"):
            ocr_results = session["stages"]["ocr"]
            if ocr_results:
                best_ocr = max(ocr_results, key=lambda x: x.get("confidence", 0))
                result.final_reading = best_ocr.get("recognized_text", "")
                result.confidence = best_ocr.get("confidence", 0)
        
        return Message(
            content={
                "type": MessageType.PROCESSING_COMPLETE,
                "frame_id": frame_id,
                "result": result.dict()
            },
            role="Coordinator",
            cause_by=type(self)
        )
    
    def _handle_fallback_decision(self, fallback_message: Message) -> List[Message]:
        """處理備援決策"""
        decision = fallback_message.content.get("fallback_decision", {})
        action = decision.get("recommended_action")
        
        messages = []
        
        if action == "use_vlm":
            messages.append(Message(
                content={
                    "type": "vlm_request",
                    "frame_id": fallback_message.content.get("frame_id"),
                    "context": fallback_message.content
                },
                role="Coordinator",
                cause_by=type(self)
            ))
        elif action == "switch_camera":
            messages.append(Message(
                content={
                    "type": "camera_switch_request",
                    "frame_id": fallback_message.content.get("frame_id"),
                    "alternative_camera_id": decision.get("alternative_camera_id")
                },
                role="Coordinator",
                cause_by=type(self)
            ))
        
        return messages


class MeterGPTOrchestrator(Role):
    """MeterGPT 系統協調器 - 基於 MetaGPT 的真正代理人協作"""
    
    def __init__(self, config: Optional[MeterGPTConfig] = None):
        """
        初始化協調器
        
        Args:
            config: 系統配置
        """
        super().__init__(
            name="MeterGPTOrchestrator",
            profile="系統協調器",
            goal="協調所有代理人完成儀器讀值任務",
            constraints="確保處理流程的正確性和效率"
        )
        
        self.config = config or get_config()
        self.logger = get_logger("MeterGPTOrchestrator")
        
        # 設置動作
        self._set_actions([
            ProcessingCoordinationAction(),
            SystemMonitorAction()
        ])
        
        # 設置觀察的訊息類型
        self._watch([
            MessageType.STREAM_FRAME,
            MessageType.QUALITY_REPORT,
            MessageType.DETECTION_RESULT,
            MessageType.OCR_RESULT,
            MessageType.VALIDATION_RESULT,
            MessageType.FALLBACK_DECISION
        ])
        
        # 系統狀態
        self.system_metrics = {
            "total_processed": 0,
            "success_count": 0,
            "failure_count": 0,
            "average_processing_time": 0.0
        }
    
    async def _act(self) -> Message:
        """
        執行協調動作
        
        Returns:
            Message: 協調結果訊息
        """
        try:
            # 取得所有相關訊息
            messages = self.rc.memory.get_by_actions([
                MessageType.STREAM_FRAME,
                MessageType.QUALITY_REPORT,
                MessageType.DETECTION_RESULT,
                MessageType.OCR_RESULT,
                MessageType.VALIDATION_RESULT,
                MessageType.FALLBACK_DECISION
            ])
            
            if not messages:
                return Message(content={"type": "no_action"}, role=self.profile)
            
            # 執行處理協調
            coordination_action = ProcessingCoordinationAction()
            coordination_messages = await coordination_action.run(messages)
            
            # 發送協調訊息
            for msg in coordination_messages:
                await self.rc.env.publish_message(msg)
            
            # 執行系統監控
            monitor_action = SystemMonitorAction()
            status_message = await monitor_action.run(messages)
            
            # 更新系統指標
            self._update_metrics(messages)
            
            return status_message
            
        except Exception as e:
            self.logger.error(f"協調動作執行失敗: {e}")
            return Message(
                content={
                    "type": "error",
                    "error": str(e)
                },
                role=self.profile
            )
    
    def _update_metrics(self, messages: List[Message]):
        """更新系統指標"""
        for message in messages:
            if message.content.get("type") == MessageType.PROCESSING_COMPLETE:
                self.system_metrics["total_processed"] += 1
                result = message.content.get("result", {})
                if result.get("status") == ProcessingStatus.SUCCESS:
                    self.system_metrics["success_count"] += 1
                else:
                    self.system_metrics["failure_count"] += 1
                
                # 更新平均處理時間
                processing_time = result.get("processing_time", 0)
                total = self.system_metrics["total_processed"]
                current_avg = self.system_metrics["average_processing_time"]
                self.system_metrics["average_processing_time"] = (
                    (current_avg * (total - 1) + processing_time) / total
                )
    
    def get_system_status(self) -> SystemStatus:
        """取得系統狀態"""
        total_processed = self.system_metrics["total_processed"]
        success_rate = (
            self.system_metrics["success_count"] / total_processed 
            if total_processed > 0 else 0.0
        )
        
        return SystemStatus(
            active_cameras=[cam.camera_id for cam in self.config.cameras] if self.config else [],
            processing_queue_size=0,  # 在新架構中，這個概念不太適用
            average_processing_time=self.system_metrics["average_processing_time"],
            success_rate=success_rate,
            error_count=self.system_metrics["failure_count"],
            system_health=min(success_rate + 0.1, 1.0)
        )


class MeterGPTTeam:
    """MeterGPT 代理人團隊管理器"""
    
    def __init__(self, config: Optional[MeterGPTConfig] = None):
        """
        初始化團隊
        
        Args:
            config: 系統配置
        """
        self.config = config or get_config()
        self.logger = get_logger("MeterGPTTeam")
        
        # 建立環境
        self.environment = Environment()
        
        # 建立代理人團隊
        self.team = None
        self._setup_team()
    
    def _setup_team(self):
        """設置代理人團隊"""
        try:
            from ..agents.stream_manager import StreamManager
            from ..agents.quality_assessor import QualityAssessor
            from ..agents.detection_agent import DetectionAgent
            from ..agents.ocr_agent import OCRAgent
            from ..agents.validation_agent import ValidationAgent
            from ..agents.fallback_agent import FallbackAgent
            
            # 建立所有代理人
            roles = [
                StreamManager(config=self.config),
                QualityAssessor(config=self.config),
                DetectionAgent(config=self.config),
                OCRAgent(config=self.config),
                ValidationAgent(config=self.config),
                FallbackAgent(config=self.config),
                MeterGPTOrchestrator(config=self.config)
            ]
            
            # 建立團隊
            self.team = Team(
                env=self.environment,
                roles=roles,
                investment=10.0,  # 預算
                idea="智慧儀器讀值系統"
            )
            
            self.logger.info("代理人團隊設置完成")
            
        except Exception as e:
            self.logger.error(f"團隊設置失敗: {e}")
            raise
    
    async def start_processing(self, camera_id: str, frame_data: bytes, 
                             metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        啟動處理流程
        
        Args:
            camera_id: 攝影機 ID
            frame_data: 影像資料
            metadata: 額外元資料
            
        Returns:
            ProcessingResult: 處理結果
        """
        try:
            frame_id = str(uuid.uuid4())
            
            # 建立串流幀物件
            camera_info = self._get_camera_info(camera_id)
            stream_frame = StreamFrame(
                frame_id=frame_id,
                camera_info=camera_info,
                frame_data=frame_data,
                frame_shape=(0, 0, 3),  # 會在後續步驟中更新
                metadata=metadata or {}
            )
            
            # 發送初始訊息到環境
            initial_message = Message(
                content={
                    "type": MessageType.STREAM_FRAME,
                    "frame_id": frame_id,
                    "stream_frame": stream_frame.dict()
                },
                role="System"
            )
            
            # 啟動團隊處理
            await self.team.run_project(initial_message)
            
            # 等待處理完成並取得結果
            # 這裡需要實作結果收集邏輯
            result = await self._collect_processing_result(frame_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"處理流程啟動失敗: {e}")
            return ProcessingResult(
                frame_id=frame_id,
                camera_id=camera_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _collect_processing_result(self, frame_id: str) -> ProcessingResult:
        """收集處理結果"""
        # 這裡需要實作從環境中收集結果的邏輯
        # 暫時返回一個基本結果
        return ProcessingResult(
            frame_id=frame_id,
            camera_id="unknown",
            status=ProcessingStatus.SUCCESS,
            final_reading="模擬讀值",
            confidence=0.8
        )
    
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


# 全域團隊實例
team_instance: Optional[MeterGPTTeam] = None


def get_team(config: Optional[MeterGPTConfig] = None) -> MeterGPTTeam:
    """取得全域團隊實例"""
    global team_instance
    if team_instance is None:
        team_instance = MeterGPTTeam(config)
    return team_instance


async def initialize_system(config: Optional[MeterGPTConfig] = None) -> MeterGPTTeam:
    """初始化系統"""
    team = get_team(config)
    return team


async def shutdown_system():
    """關閉系統"""
    global team_instance
    if team_instance:
        # 這裡可以添加清理邏輯
        team_instance = None