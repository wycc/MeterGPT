"""
MeterGPT 系統日誌工具
提供統一的日誌記錄功能，支援結構化日誌和效能監控
"""

import logging
import logging.handlers
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
from functools import wraps
import traceback
from contextlib import contextmanager

from ..models.messages import ProcessingStatus, FailureType


class StructuredFormatter(logging.Formatter):
    """結構化日誌格式器"""
    
    def __init__(self, include_extra: bool = True):
        """
        初始化格式器
        
        Args:
            include_extra: 是否包含額外欄位
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日誌記錄"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加異常資訊
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # 添加額外欄位
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                           'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                           'relativeCreated', 'thread', 'threadName', 'processName',
                           'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']
            }
            if extra_fields:
                log_data['extra'] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class MeterGPTLogger:
    """MeterGPT 專用日誌記錄器"""
    
    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        初始化日誌記錄器
        
        Args:
            name: 日誌記錄器名稱
            log_level: 日誌等級
            log_file: 日誌檔案路徑
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 避免重複添加處理器
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[str]):
        """設置日誌處理器"""
        # 控制台處理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 檔案處理器
        if log_file:
            # 確保日誌目錄存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用輪轉檔案處理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_formatter = StructuredFormatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """記錄 DEBUG 等級日誌"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """記錄 INFO 等級日誌"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """記錄 WARNING 等級日誌"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """記錄 ERROR 等級日誌"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """記錄 CRITICAL 等級日誌"""
        self.logger.critical(message, extra=kwargs)
    
    def log_processing_start(self, frame_id: str, agent_name: str, **kwargs):
        """記錄處理開始"""
        self.info(
            f"開始處理影像幀",
            frame_id=frame_id,
            agent_name=agent_name,
            event_type="processing_start",
            **kwargs
        )
    
    def log_processing_end(self, frame_id: str, agent_name: str, 
                          status: ProcessingStatus, processing_time: float, **kwargs):
        """記錄處理結束"""
        self.info(
            f"處理完成",
            frame_id=frame_id,
            agent_name=agent_name,
            status=status.value,
            processing_time=processing_time,
            event_type="processing_end",
            **kwargs
        )
    
    def log_error(self, frame_id: str, agent_name: str, 
                  error_type: FailureType, error_message: str, **kwargs):
        """記錄錯誤"""
        self.error(
            f"處理錯誤: {error_message}",
            frame_id=frame_id,
            agent_name=agent_name,
            error_type=error_type.value,
            event_type="processing_error",
            **kwargs
        )
    
    def log_quality_assessment(self, frame_id: str, camera_id: str, 
                              quality_score: float, is_acceptable: bool, **kwargs):
        """記錄品質評估"""
        self.info(
            f"品質評估完成",
            frame_id=frame_id,
            camera_id=camera_id,
            quality_score=quality_score,
            is_acceptable=is_acceptable,
            event_type="quality_assessment",
            **kwargs
        )
    
    def log_detection_result(self, frame_id: str, instrument_type: str, 
                           confidence: float, **kwargs):
        """記錄偵測結果"""
        self.info(
            f"儀器偵測完成",
            frame_id=frame_id,
            instrument_type=instrument_type,
            confidence=confidence,
            event_type="detection_result",
            **kwargs
        )
    
    def log_ocr_result(self, frame_id: str, roi_id: str, 
                      recognized_text: str, confidence: float, **kwargs):
        """記錄 OCR 結果"""
        self.info(
            f"OCR 識別完成",
            frame_id=frame_id,
            roi_id=roi_id,
            recognized_text=recognized_text,
            confidence=confidence,
            event_type="ocr_result",
            **kwargs
        )
    
    def log_validation_result(self, frame_id: str, is_valid: bool, 
                            validation_score: float, **kwargs):
        """記錄驗證結果"""
        self.info(
            f"驗證完成",
            frame_id=frame_id,
            is_valid=is_valid,
            validation_score=validation_score,
            event_type="validation_result",
            **kwargs
        )
    
    def log_fallback_decision(self, frame_id: str, trigger_reason: FailureType, 
                            recommended_action: str, **kwargs):
        """記錄備援決策"""
        self.warning(
            f"觸發備援機制",
            frame_id=frame_id,
            trigger_reason=trigger_reason.value,
            recommended_action=recommended_action,
            event_type="fallback_decision",
            **kwargs
        )
    
    def log_system_status(self, active_cameras: int, queue_size: int, 
                         success_rate: float, **kwargs):
        """記錄系統狀態"""
        self.info(
            f"系統狀態更新",
            active_cameras=active_cameras,
            queue_size=queue_size,
            success_rate=success_rate,
            event_type="system_status",
            **kwargs
        )


class PerformanceLogger:
    """效能監控日誌記錄器"""
    
    def __init__(self, logger: MeterGPTLogger):
        """
        初始化效能日誌記錄器
        
        Args:
            logger: 主日誌記錄器
        """
        self.logger = logger
        self.metrics: Dict[str, Any] = {}
    
    @contextmanager
    def measure_time(self, operation_name: str, **context):
        """測量操作執行時間的上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.debug(
                f"操作執行時間: {operation_name}",
                operation_name=operation_name,
                execution_time=execution_time,
                event_type="performance_metric",
                **context
            )
    
    def log_throughput(self, operation_name: str, count: int, 
                      time_window: float, **context):
        """記錄吞吐量指標"""
        throughput = count / time_window if time_window > 0 else 0
        self.logger.info(
            f"吞吐量指標: {operation_name}",
            operation_name=operation_name,
            count=count,
            time_window=time_window,
            throughput=throughput,
            event_type="throughput_metric",
            **context
        )
    
    def log_resource_usage(self, cpu_percent: float, memory_mb: float, 
                          gpu_percent: Optional[float] = None, **context):
        """記錄資源使用情況"""
        self.logger.info(
            f"資源使用情況",
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_percent=gpu_percent,
            event_type="resource_usage",
            **context
        )


def log_execution_time(logger: Optional[MeterGPTLogger] = None):
    """裝飾器：記錄函數執行時間"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                if logger:
                    logger.debug(
                        f"函數執行完成: {func.__name__}",
                        function_name=func.__name__,
                        execution_time=execution_time,
                        event_type="function_execution"
                    )
                
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                if logger:
                    logger.error(
                        f"函數執行失敗: {func.__name__}",
                        function_name=func.__name__,
                        execution_time=execution_time,
                        error_message=str(e),
                        event_type="function_error"
                    )
                
                raise
        return wrapper
    return decorator


def log_agent_action(agent_name: str, logger: Optional[MeterGPTLogger] = None):
    """裝飾器：記錄代理人動作"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            frame_id = kwargs.get('frame_id', 'unknown')
            
            if logger:
                logger.log_processing_start(frame_id, agent_name)
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                end_time = time.time()
                processing_time = end_time - start_time
                
                if logger:
                    logger.log_processing_end(
                        frame_id, agent_name, 
                        ProcessingStatus.SUCCESS, processing_time
                    )
                
                return result
            except Exception as e:
                end_time = time.time()
                processing_time = end_time - start_time
                
                if logger:
                    logger.log_processing_end(
                        frame_id, agent_name, 
                        ProcessingStatus.FAILED, processing_time
                    )
                    logger.log_error(
                        frame_id, agent_name, 
                        FailureType.UNKNOWN_ERROR, str(e)
                    )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            frame_id = kwargs.get('frame_id', 'unknown')
            
            if logger:
                logger.log_processing_start(frame_id, agent_name)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                processing_time = end_time - start_time
                
                if logger:
                    logger.log_processing_end(
                        frame_id, agent_name, 
                        ProcessingStatus.SUCCESS, processing_time
                    )
                
                return result
            except Exception as e:
                end_time = time.time()
                processing_time = end_time - start_time
                
                if logger:
                    logger.log_processing_end(
                        frame_id, agent_name, 
                        ProcessingStatus.FAILED, processing_time
                    )
                    logger.log_error(
                        frame_id, agent_name, 
                        FailureType.UNKNOWN_ERROR, str(e)
                    )
                
                raise
        
        # 根據函數是否為協程選擇包裝器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# 全域日誌記錄器實例
_loggers: Dict[str, MeterGPTLogger] = {}


def get_logger(name: str, log_level: str = "INFO", 
               log_file: Optional[str] = None) -> MeterGPTLogger:
    """
    取得或建立日誌記錄器
    
    Args:
        name: 日誌記錄器名稱
        log_level: 日誌等級
        log_file: 日誌檔案路徑
        
    Returns:
        MeterGPTLogger: 日誌記錄器實例
    """
    if name not in _loggers:
        _loggers[name] = MeterGPTLogger(name, log_level, log_file)
    return _loggers[name]


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    設置全域日誌配置
    
    Args:
        log_level: 日誌等級
        log_file: 日誌檔案路徑
    """
    # 設置根日誌記錄器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除現有處理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加新的處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)