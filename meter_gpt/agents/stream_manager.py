"""
StreamManager 代理人
負責管理多個攝影機的影像串流，主動推送影像幀到系統中
基於 MetaGPT 的真正代理人協作模式
"""

import asyncio
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
import uuid
import threading
from queue import Queue, Empty
import time

from metagpt.roles import Role
from metagpt.actions import Action
from metagpt.schema import Message

from ..models.messages import (
    StreamFrame, CameraInfo, AgentMessage, ProcessingStatus
)
from ..core.config import get_config, MeterGPTConfig
from ..utils.logger import get_logger, log_agent_action


class StreamCaptureAction(Action):
    """串流捕獲動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("StreamCaptureAction")
    
    async def run(self, camera_info: CameraInfo) -> Optional[StreamFrame]:
        """
        捕獲攝影機串流
        
        Args:
            camera_info: 攝影機資訊
            
        Returns:
            Optional[StreamFrame]: 串流影像幀
        """
        try:
            frame_id = str(uuid.uuid4())
            
            # 實際應用中應該從 RTSP 串流讀取
            # 這裡使用模擬資料進行示範
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加一些模擬的儀器圖案
            cv2.rectangle(dummy_frame, (100, 100), (300, 200), (255, 255, 255), 2)
            cv2.putText(dummy_frame, "123.45", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            _, encoded_frame = cv2.imencode('.jpg', dummy_frame)
            frame_data = encoded_frame.tobytes()
            
            stream_frame = StreamFrame(
                frame_id=frame_id,
                camera_info=camera_info,
                frame_data=frame_data,
                frame_shape=(480, 640, 3),
                metadata={
                    'capture_method': 'rtsp',
                    'encoding': 'jpg',
                    'simulated': True
                }
            )
            
            self.logger.debug(f"捕獲影像幀: {frame_id}", camera_id=camera_info.camera_id)
            return stream_frame
            
        except Exception as e:
            self.logger.error(f"串流捕獲失敗: {e}", camera_id=camera_info.camera_id)
            return None


class StreamPublishAction(Action):
    """串流發布動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("StreamPublishAction")
    
    async def run(self, stream_frame: StreamFrame) -> Message:
        """
        發布串流幀到環境
        
        Args:
            stream_frame: 串流影像幀
            
        Returns:
            Message: 發布的訊息
        """
        try:
            message = Message(
                content={
                    "type": "stream_frame",
                    "frame_id": stream_frame.frame_id,
                    "camera_id": stream_frame.camera_info.camera_id,
                    "stream_frame": stream_frame.dict(),
                    "timestamp": datetime.now().isoformat()
                },
                role="StreamManager",
                cause_by=type(self)
            )
            
            self.logger.info(
                f"發布影像幀: {stream_frame.frame_id}",
                camera_id=stream_frame.camera_info.camera_id
            )
            
            return message
            
        except Exception as e:
            self.logger.error(f"串流發布失敗: {e}")
            raise


class StreamManager(Role):
    """串流管理代理人 - 基於 MetaGPT 的協作模式"""
    
    def __init__(self, config: Optional[MeterGPTConfig] = None):
        """
        初始化串流管理代理人
        
        Args:
            config: 系統配置
        """
        super().__init__(
            name="StreamManager",
            profile="串流管理代理人",
            goal="管理攝影機串流並主動推送影像幀",
            constraints="確保串流的穩定性和品質"
        )
        
        self.config = config or get_config()
        self.logger = get_logger("StreamManager")
        
        # 設置動作
        self._set_actions([
            StreamCaptureAction(),
            StreamPublishAction()
        ])
        
        # 監聽系統啟動和攝影機控制訊息
        self._watch([
            "system_start",
            "camera_switch_request",
            "stream_request"
        ])
        
        # 串流狀態
        self.active_cameras: Dict[str, CameraInfo] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        self.is_streaming = False
        
        # 初始化攝影機
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """初始化攝影機配置"""
        try:
            if self.config and self.config.cameras:
                for camera_config in self.config.cameras:
                    if camera_config.is_active:
                        camera_info = CameraInfo(
                            camera_id=camera_config.camera_id,
                            camera_name=camera_config.camera_name,
                            position=camera_config.position,
                            is_active=camera_config.is_active,
                            is_primary=camera_config.is_primary
                        )
                        self.active_cameras[camera_config.camera_id] = camera_info
                
                self.logger.info(f"初始化 {len(self.active_cameras)} 台攝影機")
            else:
                # 建立預設攝影機
                default_camera = CameraInfo(
                    camera_id="default_cam",
                    camera_name="預設攝影機",
                    position=(0.0, 0.0, 0.0),
                    is_active=True,
                    is_primary=True
                )
                self.active_cameras["default_cam"] = default_camera
                self.logger.info("使用預設攝影機配置")
                
        except Exception as e:
            self.logger.error(f"攝影機初始化失敗: {e}")
    
    async def _act(self) -> Message:
        """
        執行串流管理動作
        
        Returns:
            Message: 動作結果訊息
        """
        try:
            # 檢查是否有新的訊息需要處理
            messages = self.rc.memory.get_by_actions([
                "system_start",
                "camera_switch_request", 
                "stream_request"
            ])
            
            # 處理控制訊息
            for message in messages:
                await self._handle_control_message(message)
            
            # 如果串流已啟動，繼續捕獲和發布影像
            if self.is_streaming:
                return await self._capture_and_publish()
            else:
                # 自動啟動串流（如果有活躍攝影機）
                if self.active_cameras:
                    await self._start_streaming()
                    return await self._capture_and_publish()
            
            return Message(
                content={"type": "no_action", "status": "waiting"},
                role=self.profile
            )
            
        except Exception as e:
            self.logger.error(f"串流管理動作失敗: {e}")
            return Message(
                content={"type": "error", "error": str(e)},
                role=self.profile
            )
    
    async def _handle_control_message(self, message: Message):
        """處理控制訊息"""
        try:
            msg_type = message.content.get("type")
            
            if msg_type == "system_start":
                await self._start_streaming()
            
            elif msg_type == "camera_switch_request":
                camera_id = message.content.get("alternative_camera_id")
                if camera_id and camera_id in self.active_cameras:
                    await self._switch_camera(camera_id)
            
            elif msg_type == "stream_request":
                frame_id = message.content.get("frame_id")
                if frame_id:
                    await self._handle_stream_request(frame_id)
                    
        except Exception as e:
            self.logger.error(f"控制訊息處理失敗: {e}")
    
    async def _start_streaming(self):
        """啟動串流"""
        try:
            if not self.is_streaming:
                self.is_streaming = True
                self.logger.info("串流已啟動")
                
                # 為每個活躍攝影機啟動串流任務
                for camera_id, camera_info in self.active_cameras.items():
                    if camera_info.is_active:
                        task = asyncio.create_task(
                            self._continuous_streaming(camera_info)
                        )
                        self.stream_tasks[camera_id] = task
                        
        except Exception as e:
            self.logger.error(f"串流啟動失敗: {e}")
    
    async def _continuous_streaming(self, camera_info: CameraInfo):
        """持續串流處理"""
        capture_action = StreamCaptureAction()
        publish_action = StreamPublishAction()
        
        while self.is_streaming and camera_info.is_active:
            try:
                # 捕獲影像幀
                stream_frame = await capture_action.run(camera_info)
                
                if stream_frame:
                    # 發布到環境
                    message = await publish_action.run(stream_frame)
                    await self.rc.env.publish_message(message)
                
                # 控制幀率（例如每秒 1 幀）
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"持續串流錯誤: {e}")
                await asyncio.sleep(5.0)  # 錯誤後等待重試
    
    async def _capture_and_publish(self) -> Message:
        """捕獲並發布單一影像幀"""
        try:
            # 選擇主要攝影機或第一個活躍攝影機
            primary_camera = None
            for camera_info in self.active_cameras.values():
                if camera_info.is_primary and camera_info.is_active:
                    primary_camera = camera_info
                    break
            
            if not primary_camera:
                # 選擇第一個活躍攝影機
                for camera_info in self.active_cameras.values():
                    if camera_info.is_active:
                        primary_camera = camera_info
                        break
            
            if not primary_camera:
                return Message(
                    content={"type": "error", "error": "沒有可用的攝影機"},
                    role=self.profile
                )
            
            # 捕獲影像
            capture_action = StreamCaptureAction()
            stream_frame = await capture_action.run(primary_camera)
            
            if stream_frame:
                # 發布影像
                publish_action = StreamPublishAction()
                message = await publish_action.run(stream_frame)
                
                # 發布到環境
                await self.rc.env.publish_message(message)
                
                return Message(
                    content={
                        "type": "stream_published",
                        "frame_id": stream_frame.frame_id,
                        "camera_id": primary_camera.camera_id
                    },
                    role=self.profile
                )
            else:
                return Message(
                    content={"type": "capture_failed"},
                    role=self.profile
                )
                
        except Exception as e:
            self.logger.error(f"捕獲發布失敗: {e}")
            return Message(
                content={"type": "error", "error": str(e)},
                role=self.profile
            )
    
    async def _switch_camera(self, camera_id: str):
        """切換攝影機"""
        try:
            if camera_id in self.active_cameras:
                # 停用其他攝影機
                for cam_id, camera_info in self.active_cameras.items():
                    camera_info.is_primary = (cam_id == camera_id)
                
                self.logger.info(f"切換到攝影機: {camera_id}")
            else:
                self.logger.warning(f"攝影機不存在: {camera_id}")
                
        except Exception as e:
            self.logger.error(f"攝影機切換失敗: {e}")
    
    async def _handle_stream_request(self, frame_id: str):
        """處理串流請求"""
        try:
            # 這裡可以實作特定的串流請求處理邏輯
            self.logger.info(f"處理串流請求: {frame_id}")
            
        except Exception as e:
            self.logger.error(f"串流請求處理失敗: {e}")
    
    async def stop_streaming(self):
        """停止串流"""
        try:
            self.is_streaming = False
            
            # 取消所有串流任務
            for task in self.stream_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.stream_tasks.clear()
            self.logger.info("串流已停止")
            
        except Exception as e:
            self.logger.error(f"串流停止失敗: {e}")
    
    def get_active_cameras(self) -> List[CameraInfo]:
        """取得活躍攝影機列表"""
        return [camera for camera in self.active_cameras.values() if camera.is_active]
    
    def get_camera_status(self) -> Dict[str, Dict[str, Any]]:
        """取得攝影機狀態"""
        status = {}
        for camera_id, camera_info in self.active_cameras.items():
            status[camera_id] = {
                "name": camera_info.camera_name,
                "is_active": camera_info.is_active,
                "is_primary": camera_info.is_primary,
                "position": camera_info.position,
                "has_stream_task": camera_id in self.stream_tasks
            }
        return status