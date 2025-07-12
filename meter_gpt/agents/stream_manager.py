"""
StreamManager 代理人
負責管理多個攝影機的影像串流，作為系統的資料中心匯流排
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

from metagpt.agent import Agent
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message

from ..models.messages import (
    StreamFrame, CameraInfo, AgentMessage, ProcessingStatus
)
from ..core.config import get_config
from ..utils.logger import get_logger, log_agent_action
from ..integrations.opencv_utils import StreamProcessor


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
            # 這裡應該實作實際的串流捕獲邏輯
            # 為了示範，我們使用模擬資料
            frame_id = str(uuid.uuid4())
            
            # 模擬影像資料 (實際應用中應該從 RTSP 串流讀取)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, encoded_frame = cv2.imencode('.jpg', dummy_frame)
            frame_data = encoded_frame.tobytes()
            
            stream_frame = StreamFrame(
                frame_id=frame_id,
                camera_info=camera_info,
                frame_data=frame_data,
                frame_shape=(480, 640, 3),
                metadata={
                    'capture_method': 'rtsp',
                    'encoding': 'jpg'
                }
            )
            
            self.logger.debug(f"捕獲影像幀: {frame_id}", camera_id=camera_info.camera_id)
            return stream_frame
            
        except Exception as e:
            self.logger.error(f"串流捕獲失敗: {e}", camera_id=camera_info.camera_id)
            return None


class StreamManagementAction(Action):
    """串流管理動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("StreamManagementAction")
    
    async def run(self, cameras: List[CameraInfo], 
                 stream_buffers: Dict[str, Queue]) -> Dict[str, StreamFrame]:
        """
        管理多個攝影機串流
        
        Args:
            cameras: 攝影機列表
            stream_buffers: 串流緩衝區
            
        Returns:
            Dict[str, StreamFrame]: 最新的串流幀
        """
        latest_frames = {}
        
        for camera in cameras:
            if not camera.is_active:
                continue
            
            try:
                # 從緩衝區取得最新幀
                buffer = stream_buffers.get(camera.camera_id)
                if buffer and not buffer.empty():
                    try:
                        latest_frame = buffer.get_nowait()
                        latest_frames[camera.camera_id] = latest_frame
                    except Empty:
                        pass
                
            except Exception as e:
                self.logger.error(
                    f"串流管理失敗: {e}", 
                    camera_id=camera.camera_id
                )
        
        return latest_frames


class StreamManager(Role):
    """串流管理代理人"""
    
    def __init__(self, name: str = "StreamManager", **kwargs):
        """
        初始化串流管理代理人
        
        Args:
            name: 代理人名稱
        """
        super().__init__(name=name, **kwargs)
        
        # 設置動作
        self._init_actions([StreamCaptureAction(), StreamManagementAction()])
        
        # 初始化屬性
        self.cameras: List[CameraInfo] = []
        self.stream_buffers: Dict[str, Queue] = {}
        self.stream_processors: Dict[str, StreamProcessor] = {}
        self.capture_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        
        self.logger = get_logger("StreamManager")
        
        # 載入配置
        self._load_configuration()
    
    def _load_configuration(self):
        """載入配置"""
        try:
            config = get_config()
            if config and config.cameras:
                self.cameras = [
                    CameraInfo(
                        camera_id=cam.camera_id,
                        camera_name=cam.camera_name,
                        position=cam.position,
                        is_active=cam.is_active,
                        is_primary=cam.is_primary
                    )
                    for cam in config.cameras
                ]
                
                self.logger.info(f"載入了 {len(self.cameras)} 個攝影機配置")
            else:
                self.logger.warning("沒有找到攝影機配置")
                
        except Exception as e:
            self.logger.error(f"載入配置失敗: {e}")
    
    @log_agent_action("StreamManager")
    async def _act(self) -> Message:
        """執行代理人動作"""
        try:
            # 啟動串流捕獲
            if not self.is_running:
                await self.start_streaming()
            
            # 管理串流
            action = StreamManagementAction()
            latest_frames = await action.run(self.cameras, self.stream_buffers)
            
            # 建立回應訊息
            message_content = {
                'action': 'stream_management',
                'active_cameras': len([c for c in self.cameras if c.is_active]),
                'latest_frames': len(latest_frames),
                'timestamp': datetime.now().isoformat()
            }
            
            return Message(
                content=str(message_content),
                role=self.profile,
                cause_by=StreamManagementAction
            )
            
        except Exception as e:
            self.logger.error(f"代理人動作執行失敗: {e}")
            return Message(
                content=f"Error: {str(e)}",
                role=self.profile,
                cause_by=StreamManagementAction
            )
    
    async def start_streaming(self):
        """啟動串流捕獲"""
        try:
            self.is_running = True
            
            for camera in self.cameras:
                if camera.is_active:
                    # 建立緩衝區和處理器
                    self.stream_buffers[camera.camera_id] = Queue(maxsize=10)
                    self.stream_processors[camera.camera_id] = StreamProcessor()
                    
                    # 啟動捕獲執行緒
                    capture_thread = threading.Thread(
                        target=self._capture_loop,
                        args=(camera,),
                        daemon=True
                    )
                    capture_thread.start()
                    self.capture_threads[camera.camera_id] = capture_thread
            
            self.logger.info("串流捕獲已啟動")
            
        except Exception as e:
            self.logger.error(f"啟動串流失敗: {e}")
            raise
    
    def _capture_loop(self, camera: CameraInfo):
        """攝影機捕獲迴圈"""
        try:
            # 實際應用中應該使用 OpenCV 連接 RTSP 串流
            # cap = cv2.VideoCapture(camera.rtsp_url)
            
            while self.is_running:
                try:
                    # 模擬捕獲影像
                    frame_id = str(uuid.uuid4())
                    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    _, encoded_frame = cv2.imencode('.jpg', dummy_frame)
                    frame_data = encoded_frame.tobytes()
                    
                    stream_frame = StreamFrame(
                        frame_id=frame_id,
                        camera_info=camera,
                        frame_data=frame_data,
                        frame_shape=(480, 640, 3),
                        metadata={'source': 'simulation'}
                    )
                    
                    # 添加到緩衝區
                    buffer = self.stream_buffers[camera.camera_id]
                    if buffer.full():
                        try:
                            buffer.get_nowait()  # 移除舊幀
                        except Empty:
                            pass
                    
                    buffer.put(stream_frame)
                    
                    # 添加到處理器
                    processor = self.stream_processors[camera.camera_id]
                    frame_array = cv2.imdecode(
                        np.frombuffer(frame_data, np.uint8), 
                        cv2.IMREAD_COLOR
                    )
                    processor.add_frame(frame_array, time.time())
                    
                    # 控制幀率
                    time.sleep(1.0 / 30.0)  # 30 FPS
                    
                except Exception as e:
                    self.logger.error(f"捕獲迴圈錯誤: {e}", camera_id=camera.camera_id)
                    time.sleep(1.0)
            
        except Exception as e:
            self.logger.error(f"捕獲迴圈失敗: {e}", camera_id=camera.camera_id)
    
    async def stop_streaming(self):
        """停止串流捕獲"""
        try:
            self.is_running = False
            
            # 等待所有執行緒結束
            for thread in self.capture_threads.values():
                if thread.is_alive():
                    thread.join(timeout=5.0)
            
            self.capture_threads.clear()
            self.stream_buffers.clear()
            self.stream_processors.clear()
            
            self.logger.info("串流捕獲已停止")
            
        except Exception as e:
            self.logger.error(f"停止串流失敗: {e}")
    
    def get_latest_frame(self, camera_id: str) -> Optional[StreamFrame]:
        """
        取得指定攝影機的最新影像幀
        
        Args:
            camera_id: 攝影機 ID
            
        Returns:
            Optional[StreamFrame]: 最新的影像幀
        """
        try:
            buffer = self.stream_buffers.get(camera_id)
            if buffer and not buffer.empty():
                # 取得最新幀 (清空緩衝區並取得最後一幀)
                latest_frame = None
                while not buffer.empty():
                    try:
                        latest_frame = buffer.get_nowait()
                    except Empty:
                        break
                return latest_frame
            
            return None
            
        except Exception as e:
            self.logger.error(f"取得最新幀失敗: {e}", camera_id=camera_id)
            return None
    
    def get_stable_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        取得指定攝影機的穩定影像幀
        
        Args:
            camera_id: 攝影機 ID
            
        Returns:
            Optional[np.ndarray]: 穩定的影像幀
        """
        try:
            processor = self.stream_processors.get(camera_id)
            if processor:
                return processor.get_stable_frame()
            
            return None
            
        except Exception as e:
            self.logger.error(f"取得穩定幀失敗: {e}", camera_id=camera_id)
            return None
    
    def get_primary_camera(self) -> Optional[CameraInfo]:
        """取得主要攝影機"""
        for camera in self.cameras:
            if camera.is_primary and camera.is_active:
                return camera
        return None
    
    def get_backup_cameras(self) -> List[CameraInfo]:
        """取得備援攝影機列表"""
        return [
            camera for camera in self.cameras 
            if not camera.is_primary and camera.is_active
        ]
    
    def switch_primary_camera(self, new_primary_id: str) -> bool:
        """
        切換主要攝影機
        
        Args:
            new_primary_id: 新的主要攝影機 ID
            
        Returns:
            bool: 是否成功切換
        """
        try:
            # 找到新的主要攝影機
            new_primary = None
            for camera in self.cameras:
                if camera.camera_id == new_primary_id:
                    new_primary = camera
                    break
            
            if not new_primary or not new_primary.is_active:
                self.logger.error(f"無法切換到攝影機: {new_primary_id}")
                return False
            
            # 更新主要攝影機標記
            for camera in self.cameras:
                camera.is_primary = (camera.camera_id == new_primary_id)
            
            self.logger.info(f"已切換主要攝影機到: {new_primary_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"切換主要攝影機失敗: {e}")
            return False
    
    async def get_stream_status(self) -> Dict[str, any]:
        """取得串流狀態"""
        try:
            status = {
                'is_running': self.is_running,
                'total_cameras': len(self.cameras),
                'active_cameras': len([c for c in self.cameras if c.is_active]),
                'primary_camera': None,
                'camera_status': {}
            }
            
            # 主要攝影機資訊
            primary_camera = self.get_primary_camera()
            if primary_camera:
                status['primary_camera'] = primary_camera.camera_id
            
            # 各攝影機狀態
            for camera in self.cameras:
                buffer = self.stream_buffers.get(camera.camera_id)
                buffer_size = buffer.qsize() if buffer else 0
                
                status['camera_status'][camera.camera_id] = {
                    'is_active': camera.is_active,
                    'is_primary': camera.is_primary,
                    'buffer_size': buffer_size,
                    'thread_alive': camera.camera_id in self.capture_threads and 
                                   self.capture_threads[camera.camera_id].is_alive()
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"取得串流狀態失敗: {e}")
            return {'error': str(e)}


# 建立全域 StreamManager 實例的工廠函數
def create_stream_manager() -> StreamManager:
    """建立 StreamManager 實例"""
    return StreamManager()