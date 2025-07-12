"""
MeterGPT 系統配置管理
提供系統配置的載入、驗證和管理功能
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
import logging

from ..models.messages import ConfigModel, CameraInfo, ValidationRule


class CameraConfig(BaseModel):
    """攝影機配置"""
    camera_id: str = Field(..., description="攝影機識別碼")
    camera_name: str = Field(..., description="攝影機名稱")
    rtsp_url: str = Field(..., description="RTSP 串流網址")
    position: tuple = Field((0.0, 0.0, 0.0), description="攝影機位置")
    is_primary: bool = Field(False, description="是否為主要攝影機")
    is_active: bool = Field(True, description="是否啟用")
    resolution: tuple = Field((1920, 1080), description="解析度")
    fps: int = Field(30, description="幀率")
    ptz_enabled: bool = Field(False, description="是否支援 PTZ")
    ptz_presets: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="PTZ 預設位置")


class ModelConfig(BaseModel):
    """模型配置"""
    model_name: str = Field(..., description="模型名稱")
    model_path: str = Field(..., description="模型路徑")
    model_type: str = Field(..., description="模型類型")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="信心度閾值")
    device: str = Field("cpu", description="運算設備")
    batch_size: int = Field(1, description="批次大小")
    input_size: tuple = Field((640, 640), description="輸入尺寸")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="前處理參數")
    postprocessing: Dict[str, Any] = Field(default_factory=dict, description="後處理參數")


class OCRConfig(BaseModel):
    """OCR 配置"""
    engine: str = Field("paddleocr", description="OCR 引擎")
    language: str = Field("en", description="語言")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="信心度閾值")
    use_gpu: bool = Field(False, description="是否使用 GPU")
    det_model_dir: Optional[str] = Field(None, description="檢測模型路徑")
    rec_model_dir: Optional[str] = Field(None, description="識別模型路徑")
    cls_model_dir: Optional[str] = Field(None, description="分類模型路徑")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="前處理參數")


class QualityConfig(BaseModel):
    """品質評估配置"""
    sharpness_threshold: float = Field(0.3, ge=0.0, le=1.0, description="清晰度閾值")
    brightness_range: tuple = Field((0.2, 0.8), description="亮度範圍")
    contrast_threshold: float = Field(0.3, ge=0.0, le=1.0, description="對比度閾值")
    occlusion_threshold: float = Field(0.3, ge=0.0, le=1.0, description="遮擋閾值")
    overall_threshold: float = Field(0.6, ge=0.0, le=1.0, description="整體品質閾值")
    enable_hdr_enhancement: bool = Field(True, description="是否啟用 HDR 增強")
    noise_reduction: bool = Field(True, description="是否降噪")


class VLMConfig(BaseModel):
    """VLM 配置"""
    model_name: str = Field("gpt-4-vision-preview", description="VLM 模型名稱")
    api_key: Optional[str] = Field(None, description="API 金鑰")
    api_base: Optional[str] = Field(None, description="API 基礎網址")
    max_tokens: int = Field(500, description="最大 token 數")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="溫度參數")
    timeout: int = Field(30, description="請求超時時間")
    retry_attempts: int = Field(3, description="重試次數")
    confidence_threshold: float = Field(0.9, ge=0.0, le=1.0, description="信心度閾值")


class SystemConfig(BaseModel):
    """系統配置"""
    system_name: str = Field("MeterGPT", description="系統名稱")
    version: str = Field("1.0.0", description="版本號")
    log_level: str = Field("INFO", description="日誌等級")
    log_file: str = Field("meter_gpt.log", description="日誌檔案")
    max_workers: int = Field(4, description="最大工作執行緒數")
    queue_size: int = Field(100, description="佇列大小")
    processing_timeout: int = Field(60, description="處理超時時間")
    enable_metrics: bool = Field(True, description="是否啟用指標收集")
    metrics_port: int = Field(8080, description="指標服務埠")
    health_check_interval: int = Field(30, description="健康檢查間隔")


class FallbackConfig(BaseModel):
    """備援配置"""
    max_retries: int = Field(3, description="最大重試次數")
    retry_delay: float = Field(1.0, description="重試延遲時間")
    camera_switch_threshold: float = Field(0.4, ge=0.0, le=1.0, description="攝影機切換閾值")
    vlm_activation_threshold: float = Field(0.3, ge=0.0, le=1.0, description="VLM 啟動閾值")
    manual_review_threshold: float = Field(0.1, ge=0.0, le=1.0, description="人工審核閾值")
    ptz_adjustment_enabled: bool = Field(True, description="是否啟用 PTZ 調整")
    ptz_scan_range: Dict[str, float] = Field(
        default_factory=lambda: {"pan": 30.0, "tilt": 15.0, "zoom": 2.0},
        description="PTZ 掃描範圍"
    )


class MeterGPTConfig(ConfigModel):
    """MeterGPT 主配置"""
    # 基本資訊
    config_name: str = Field("meter_gpt_config", description="配置名稱")
    
    # 各模組配置
    system: SystemConfig = Field(default_factory=SystemConfig, description="系統配置")
    cameras: List[CameraConfig] = Field(default_factory=list, description="攝影機配置列表")
    detection_model: ModelConfig = Field(..., description="偵測模型配置")
    corner_detection_model: ModelConfig = Field(..., description="角點偵測模型配置")
    ocr: OCRConfig = Field(default_factory=OCRConfig, description="OCR 配置")
    quality: QualityConfig = Field(default_factory=QualityConfig, description="品質評估配置")
    vlm: VLMConfig = Field(default_factory=VLMConfig, description="VLM 配置")
    fallback: FallbackConfig = Field(default_factory=FallbackConfig, description="備援配置")
    
    # 儀器模板配置
    instrument_templates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="儀器模板配置"
    )
    
    # 驗證規則
    validation_rules: List[ValidationRule] = Field(
        default_factory=list, 
        description="驗證規則列表"
    )
    
    @validator('cameras')
    def validate_cameras(cls, v):
        """驗證攝影機配置"""
        if not v:
            raise ValueError("至少需要配置一台攝影機")
        
        camera_ids = [cam.camera_id for cam in v]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("攝影機 ID 不能重複")
        
        primary_cameras = [cam for cam in v if cam.is_primary]
        if len(primary_cameras) != 1:
            raise ValueError("必須且只能有一台主要攝影機")
        
        return v


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置檔案路徑
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Optional[MeterGPTConfig] = None
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config_path(self) -> str:
        """取得預設配置檔案路徑"""
        return os.path.join(os.getcwd(), "config", "meter_gpt_config.yaml")
    
    def load_config(self, config_path: Optional[str] = None) -> MeterGPTConfig:
        """
        載入配置檔案
        
        Args:
            config_path: 配置檔案路徑
            
        Returns:
            MeterGPTConfig: 配置物件
        """
        if config_path:
            self.config_path = config_path
        
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"配置檔案不存在: {self.config_path}，使用預設配置")
                self.config = self._create_default_config()
                self.save_config()
                return self.config
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self.config = MeterGPTConfig(**config_data)
            self.logger.info(f"成功載入配置檔案: {self.config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"載入配置檔案失敗: {e}")
            self.config = self._create_default_config()
            return self.config
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        儲存配置檔案
        
        Args:
            config_path: 配置檔案路徑
            
        Returns:
            bool: 是否成功儲存
        """
        if not self.config:
            self.logger.error("沒有配置可以儲存")
            return False
        
        save_path = config_path or self.config_path
        
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            config_dict = self.config.dict()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功儲存配置檔案: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"儲存配置檔案失敗: {e}")
            return False
    
    def _create_default_config(self) -> MeterGPTConfig:
        """建立預設配置"""
        return MeterGPTConfig(
            config_id="default_config",
            detection_model=ModelConfig(
                model_name="yolov8m",
                model_path="models/yolov8m.pt",
                model_type="detection",
                confidence_threshold=0.5,
                device="cpu"
            ),
            corner_detection_model=ModelConfig(
                model_name="corner_detector",
                model_path="models/corner_detector.pt",
                model_type="corner_detection",
                confidence_threshold=0.7,
                device="cpu"
            ),
            cameras=[
                CameraConfig(
                    camera_id="cam_001",
                    camera_name="主要攝影機",
                    rtsp_url="rtsp://192.168.1.100:554/stream1",
                    is_primary=True
                )
            ]
        )
    
    def get_config(self) -> Optional[MeterGPTConfig]:
        """取得當前配置"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        更新配置
        
        Args:
            updates: 更新的配置項目
            
        Returns:
            bool: 是否成功更新
        """
        if not self.config:
            self.logger.error("沒有載入的配置可以更新")
            return False
        
        try:
            config_dict = self.config.dict()
            self._deep_update(config_dict, updates)
            self.config = MeterGPTConfig(**config_dict)
            self.logger.info("配置更新成功")
            return True
            
        except Exception as e:
            self.logger.error(f"配置更新失敗: {e}")
            return False
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """
        驗證配置
        
        Returns:
            tuple: (是否有效, 錯誤訊息列表)
        """
        if not self.config:
            return False, ["沒有載入的配置"]
        
        errors = []
        
        try:
            # 驗證模型檔案是否存在
            if not os.path.exists(self.config.detection_model.model_path):
                errors.append(f"偵測模型檔案不存在: {self.config.detection_model.model_path}")
            
            if not os.path.exists(self.config.corner_detection_model.model_path):
                errors.append(f"角點偵測模型檔案不存在: {self.config.corner_detection_model.model_path}")
            
            # 驗證攝影機配置
            for camera in self.config.cameras:
                if not camera.rtsp_url:
                    errors.append(f"攝影機 {camera.camera_id} 缺少 RTSP URL")
            
            # 驗證 VLM 配置
            if self.config.vlm.api_key is None and "gpt" in self.config.vlm.model_name.lower():
                errors.append("使用 GPT 模型需要提供 API 金鑰")
            
        except Exception as e:
            errors.append(f"配置驗證過程中發生錯誤: {e}")
        
        return len(errors) == 0, errors


# 全域配置管理器實例
config_manager = ConfigManager()


def get_config() -> Optional[MeterGPTConfig]:
    """取得全域配置"""
    return config_manager.get_config()


def load_config(config_path: Optional[str] = None) -> MeterGPTConfig:
    """載入配置"""
    return config_manager.load_config(config_path)


def save_config(config_path: Optional[str] = None) -> bool:
    """儲存配置"""
    return config_manager.save_config(config_path)