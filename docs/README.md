# MeterGPT 系統設計文件

## 概述

MeterGPT 是一個基於 MetaGPT 框架的智慧儀器讀值系統，專為自動化儀器監控和資料擷取而設計。系統採用多代理人架構，結合電腦視覺、光學字符識別 (OCR) 和人工智慧技術，提供高精度、高可靠性的儀器讀值解決方案。

## 系統架構

### 整體架構圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MeterGPT 系統                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   外部介面      │    │   核心協調器     │    │   監控與管理     │         │
│  │                │    │                │    │                │         │
│  │ • REST API     │◄──►│ Orchestrator   │◄──►│ • 效能監控      │         │
│  │ • WebSocket    │    │                │    │ • 日誌管理      │         │
│  │ • 資料庫介面    │    │ • 流程協調      │    │ • 配置管理      │         │
│  └─────────────────┘    │ • 代理人管理    │    └─────────────────┘         │
│                         │ • 錯誤處理      │                               │
│                         └─────────────────┘                               │
│                                   │                                       │
│  ┌─────────────────────────────────┼─────────────────────────────────┐     │
│  │                    代理人生態系統                                   │     │
│  │                                                                   │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │     │
│  │  │ Stream      │  │ Quality     │  │ Detection   │  │ OCR         │ │     │
│  │  │ Manager     │─►│ Assessor    │─►│ Agent       │─►│ Agent       │ │     │
│  │  │             │  │             │  │             │  │             │ │     │
│  │  │ • 串流管理   │  │ • 品質評估   │  │ • 儀器偵測   │  │ • 文字識別   │ │     │
│  │  │ • 多攝影機   │  │ • 健康分數   │  │ • 角點偵測   │  │ • 多引擎     │ │     │
│  │  │ • 緩衝控制   │  │ • 趨勢分析   │  │ • ROI提取   │  │ • 後處理     │ │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │     │
│  │                                                           │         │     │
│  │  ┌─────────────┐  ┌─────────────┐                        │         │     │
│  │  │ Validation  │  │ Fallback    │◄───────────────────────┘         │     │
│  │  │ Agent       │─►│ Agent       │                                  │     │
│  │  │             │  │             │                                  │     │
│  │  │ • 結果驗證   │  │ • 備援決策   │                                  │     │
│  │  │ • 規則引擎   │  │ • VLM備援   │                                  │     │
│  │  │ • 交叉驗證   │  │ • PTZ控制   │                                  │     │
│  │  └─────────────┘  └─────────────┘                                  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                        整合層                                        │     │
│  │                                                                     │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │     │
│  │  │ YOLO        │  │ PaddleOCR   │  │ OpenCV      │  │ VLM APIs    │ │     │
│  │  │ Wrapper     │  │ Wrapper     │  │ Utils       │  │             │ │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心組件

| 組件 | 功能描述 | 主要技術 |
|------|----------|----------|
| **Orchestrator** | 系統核心協調器，管理整個處理流程 | MetaGPT, AsyncIO |
| **StreamManager** | 串流管理代理人，處理多攝影機串流 | OpenCV, Threading |
| **QualityAssessor** | 品質評估代理人，計算影像健康分數 | 影像處理演算法 |
| **DetectionAgent** | 偵測代理人，執行儀器和角點偵測 | YOLO, 深度學習 |
| **OCRAgent** | OCR代理人，執行文字識別 | PaddleOCR, Tesseract |
| **ValidationAgent** | 驗證代理人，驗證識別結果合理性 | 規則引擎, 統計分析 |
| **FallbackAgent** | 備援代理人，處理異常和備援策略 | VLM APIs, PTZ控制 |

## 代理人詳細設計

### 1. [StreamManager - 串流管理代理人](./agents/stream_manager.md)

**核心職責**：
- 管理多個攝影機的 RTSP 串流
- 提供影像緩衝和穩定化功能
- 支援主備攝影機動態切換

**關鍵特性**：
- 支援最多 8 個攝影機同時串流
- 自動重連和錯誤恢復機制
- 影像穩定化和品質增強
- 30 FPS 高幀率處理能力

### 2. [QualityAssessor - 品質評估代理人](./agents/quality_assessor.md)

**核心職責**：
- 評估影像品質並計算健康分數
- 分析品質趨勢和異常
- 提供品質改善建議

**評估指標**：
- 清晰度分數 (Sharpness Score)
- 亮度分數 (Brightness Score)
- 對比度分數 (Contrast Score)
- 遮擋比例 (Occlusion Ratio)
- 失真分數 (Distortion Score)
- 整體品質分數 (Overall Score)

### 3. [DetectionAgent - 偵測代理人](./agents/detection_agent.md)

**核心職責**：
- 使用 YOLO 模型偵測儀器類型
- 精確定位儀器四個角點
- 執行透視校正和 ROI 提取

**支援儀器類型**：
- 數位顯示器 (Digital Display)
- 七段顯示器 (Seven Segment)
- LCD 螢幕 (LCD Screen)
- 類比儀表 (Analog Gauge)
- LED 顯示器 (LED Display)

### 4. [OCRAgent - OCR代理人](./agents/ocr_agent.md)

**核心職責**：
- 多引擎 OCR 識別 (PaddleOCR, Tesseract)
- 專用格式識別 (七段顯示器等)
- 批次處理和結果後處理

**識別格式**：
- 一般文字識別
- 數字識別
- 七段顯示器識別
- LCD/LED 顯示器識別
- 類比儀表讀值

### 5. [ValidationAgent - 驗證代理人](./agents/validation_agent.md)

**核心職責**：
- 多重驗證規則執行
- 交叉驗證和一致性檢查
- 異常檢測和品質評分

**驗證規則**：
- 格式驗證 (Format Validation)
- 範圍驗證 (Range Validation)
- 長度驗證 (Length Validation)
- 一致性驗證 (Consistency Validation)
- 信心度驗證 (Confidence Validation)

### 6. [FallbackAgent - 備援代理人](./agents/fallback_agent.md)

**核心職責**：
- 智慧備援決策制定
- VLM 視覺語言模型備援
- PTZ 攝影機控制
- 人工審核流程管理

**備援策略**：
- 攝影機切換 (Camera Switch)
- VLM 備援識別 (VLM Fallback)
- PTZ 位置調整 (PTZ Adjustment)
- OCR 重試 (OCR Retry)
- 人工審核 (Manual Review)

### 7. [Orchestrator - 系統協調器](./core/orchestrator.md)

**核心職責**：
- 管理完整的處理流程
- 協調所有代理人協作
- 統一錯誤處理和監控
- 系統效能最佳化

**處理流程**：
1. 串流捕獲 → 2. 品質評估 → 3. 儀器偵測 → 4. OCR 處理 → 5. 結果驗證 → 6. 備援處理 (如需要)

## 技術特性

### 高可靠性設計

- **多層備援機制**：攝影機備援、演算法備援、VLM 備援
- **智慧錯誤恢復**：自動重試、降級處理、人工介入
- **實時監控**：系統健康監控、效能指標追蹤
- **容錯處理**：優雅降級、資源保護、異常隔離

### 高效能架構

- **並行處理**：多執行緒串流處理、非同步代理人協作
- **智慧快取**：結果快取、模型快取、影像快取
- **資源最佳化**：GPU 加速、記憶體管理、負載平衡
- **批次處理**：批次 OCR、批次驗證、批次品質評估

### 可擴展性

- **模組化設計**：獨立代理人、可插拔組件
- **配置驅動**：靈活配置、動態調整
- **API 介面**：RESTful API、WebSocket 支援
- **雲端整合**：支援雲端部署、邊緣計算

## 部署架構

### 單機部署
```
┌─────────────────────────────────────┐
│           單機部署架構               │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────────┐ │
│  │        MeterGPT 系統            │ │
│  │                                 │ │
│  │  • 所有代理人                   │ │
│  │  • 本地模型                     │ │
│  │  • SQLite 資料庫                │ │
│  │  • 本地檔案儲存                 │ │
│  └─────────────────────────────────┘ │
│                                     │
│  ┌─────────────────────────────────┐ │
│  │        硬體需求                 │ │
│  │                                 │ │
│  │  • CPU: 8 核心以上              │ │
│  │  • RAM: 16GB 以上               │ │
│  │  • GPU: RTX 3060 以上 (可選)    │ │
│  │  • 儲存: 500GB SSD              │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### 分散式部署
```
┌─────────────────────────────────────────────────────────────┐
│                    分散式部署架構                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   邊緣節點   │  │   邊緣節點   │  │   邊緣節點   │         │
│  │             │  │             │  │             │         │
│  │ • 串流管理   │  │ • 串流管理   │  │ • 串流管理   │         │
│  │ • 品質評估   │  │ • 品質評估   │  │ • 品質評估   │         │
│  │ • 基礎偵測   │  │ • 基礎偵測   │  │ • 基礎偵測   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │               │
│         └────────────────┼────────────────┘               │
│                          │                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  中央處理節點                        │   │
│  │                                                     │   │
│  │  • Orchestrator                                    │   │
│  │  • 高精度 OCR 處理                                  │   │
│  │  • 複雜驗證邏輯                                     │   │
│  │  • VLM 備援服務                                     │   │
│  │  • 資料庫和儲存                                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 雲端部署
```
┌─────────────────────────────────────────────────────────────┐
│                      雲端部署架構                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                   負載平衡器                        │     │
│  └─────────────────────┬───────────────────────────────┘     │
│                        │                                   │
│  ┌─────────────────────┼───────────────────────────────┐     │
│  │              Kubernetes 集群                       │     │
│  │                     │                               │     │
│  │  ┌─────────────┐   ┌─┴─────────────┐   ┌─────────────┐ │     │
│  │  │ 串流處理 Pod │   │ OCR 處理 Pod  │   │ 驗證處理 Pod │ │     │
│  │  │             │   │               │   │             │ │     │
│  │  │ • 多副本     │   │ • GPU 加速    │   │ • 規則引擎   │ │     │
│  │  │ • 自動擴展   │   │ • 模型快取    │   │ • 批次處理   │ │     │
│  │  └─────────────┘   └───────────────┘   └─────────────┘ │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                   雲端服務                          │     │
│  │                                                     │     │
│  │  • 雲端資料庫 (PostgreSQL/MongoDB)                  │     │
│  │  • 物件儲存 (S3/GCS)                               │     │
│  │  • 監控服務 (Prometheus/Grafana)                   │     │
│  │  • 日誌服務 (ELK Stack)                            │     │
│  │  • VLM API 服務 (OpenAI/Claude)                    │     │
│  └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 配置管理

### 系統配置結構
```yaml
# config.yaml
system:
  config_id: "production_v1"
  environment: "production"
  log_level: "INFO"
  
  processing:
    queue_size: 100
    max_concurrent_tasks: 10
    default_timeout: 30
    
cameras:
  - camera_id: "cam_001"
    camera_name: "主要攝影機"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    is_primary: true
    
models:
  detection_model:
    model_path: "./models/instrument_yolo.pt"
    confidence_threshold: 0.7
    device: "cuda"
    
  corner_detection_model:
    model_path: "./models/corner_yolo.pt"
    confidence_threshold: 0.8
    
ocr:
  paddle_ocr:
    language: "ch"
    use_gpu: true
    confidence_threshold: 0.7
    
validation:
  overall_threshold: 0.7
  rules:
    - rule_id: "numeric_format"
      enabled: true
      parameters:
        pattern: "^-?\\d+\\.?\\d*$"
        
fallback:
  vlm:
    enabled: true
    model_name: "gpt-4-vision-preview"
    confidence_threshold: 0.7
    
  camera_switch:
    enabled: true
    quality_threshold: 0.4
```

## API 介面

### RESTful API

#### 處理單一影像
```http
POST /api/v1/process
Content-Type: multipart/form-data

{
  "camera_id": "cam_001",
  "image": <binary_data>,
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "priority": 1
  }
}
```

#### 取得系統狀態
```http
GET /api/v1/status

Response:
{
  "system_status": "running",
  "active_agents": {
    "stream_manager": true,
    "quality_assessor": true,
    "detection_agent": true,
    "ocr_agent": true,
    "validation_agent": true,
    "fallback_agent": true
  },
  "processing_queue_size": 5,
  "system_metrics": {
    "total_processed": 1250,
    "success_rate": 0.94,
    "average_processing_time": 2.3
  }
}
```

#### 取得歷史資料
```http
GET /api/v1/history?camera_id=cam_001&hours=24

Response:
{
  "results": [
    {
      "frame_id": "frame_001",
      "camera_id": "cam_001",
      "final_reading": "123.45",
      "confidence": 0.92,
      "timestamp": "2024-01-01T12:00:00Z",
      "processing_time": 2.1
    }
  ],
  "total_count": 1440,
  "page": 1,
  "page_size": 100
}
```

### WebSocket API

#### 即時串流處理
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

// 發送處理請求
ws.send(JSON.stringify({
  "action": "start_processing",
  "camera_id": "cam_001",
  "options": {
    "continuous": true,
    "interval": 5
  }
}));

// 接收處理結果
ws.onmessage = function(event) {
  const result = JSON.parse(event.data);
  console.log('處理結果:', result);
};
```

## 監控與維護

### 效能監控指標

#### 系統級指標
- **處理吞吐量**：每秒處理幀數 (FPS)
- **成功率**：處理成功的比例
- **平均延遲**：從輸入到輸出的平均時間
- **資源使用率**：CPU、記憶體、GPU 使用率
- **錯誤率**：各類錯誤的發生頻率

#### 代理人級指標
- **StreamManager**：串流穩定性、緩衝區使用率
- **QualityAssessor**：平均品質分數、品質趨勢
- **DetectionAgent**：偵測準確率、處理時間
- **OCRAgent**：識別準確率、引擎效能
- **ValidationAgent**：驗證通過率、規則效能
- **FallbackAgent**：備援觸發率、成功率

### 日誌管理

#### 日誌層級
- **DEBUG**：詳細的除錯資訊
- **INFO**：一般資訊和狀態更新
- **WARNING**：警告訊息和潛在問題
- **ERROR**：錯誤訊息和異常情況
- **CRITICAL**：嚴重錯誤和系統故障

#### 日誌格式
```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "agent": "OCRAgent",
  "action": "process_roi",
  "frame_id": "frame_001",
  "camera_id": "cam_001",
  "message": "OCR 處理完成",
  "metadata": {
    "processing_time": 0.85,
    "confidence": 0.92,
    "recognized_text": "123.45"
  }
}
```

### 維護操作

#### 系統健康檢查
```bash
# 檢查系統狀態
curl -X GET http://localhost:8000/api/v1/health

# 檢查代理人狀態
curl -X GET http://localhost:8000/api/v1/agents/status

# 檢查模型狀態
curl -X GET http://localhost:8000/api/v1/models/status
```

#### 配置更新
```bash
# 重新載入配置
curl -X POST http://localhost:8000/api/v1/config/reload

# 更新特定代理人配置
curl -X PUT http://localhost:8000/api/v1/agents/ocr_agent/config \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.8}'
```

#### 模型管理
```bash
# 更新模型
curl -X POST http://localhost:8000/api/v1/models/update \
  -H "Content-Type: application/json" \
  -d '{"model_type": "detection", "model_path": "./models/new_model.pt"}'

# 模型效能測試
curl -X POST http://localhost:8000/api/v1/models/benchmark \
  -H "Content-Type: application/json" \
  -d '{"model_type": "detection", "test_images": 100}'
```

## 故障排除

### 常見問題

#### 1. 系統啟動失敗
**症狀**：系統無法正常啟動
**可能原因**：
- 配置檔案錯誤
- 模型檔案缺失
- 依賴套件問題
- 硬體資源不足

**解決方案**：
```bash
# 檢查配置檔案
python -m meter_gpt.utils.config_validator config.yaml

# 檢查模型檔案
ls -la models/

# 檢查依賴套件
pip check

# 檢查硬體資源
nvidia-smi  # GPU 檢查
free -h     # 記憶體檢查
```

#### 2. 處理效能低下
**症狀**：處理速度明顯下降
**可能原因**：
- 記憶體洩漏
- GPU 記憶體不足
- 網路延遲
- 模型效能問題

**解決方案**：
```bash
# 監控資源使用
htop
nvidia-smi -l 1

# 檢查網路延遲
ping camera_ip

# 重啟系統服務
systemctl restart meter-gpt
```

#### 3. 識別準確率下降
**症狀**：OCR 識別準確率明顯下降
**可能原因**：
- 影像品質問題
- 模型版本過舊
- 配置參數不當
- 環境光線變化

**解決方案**：
```bash
# 檢查影像品質
curl -X GET http://localhost:8000/api/v1/quality/report

# 更新模型
curl -X POST http://localhost:8000/api/v1/models/update

# 調整配置參數
curl -X PUT http://localhost:8000/api/v1/config/ocr \
  -d '{"confidence_threshold": 0.6}'
```

### 除錯工具

#### 1. 視覺化除錯
```python
# 啟用視覺化除錯模式
from meter_gpt.utils.debug import enable_visual_debug

enable_visual_debug(
    save_intermediate_images=True,
    save_path="./debug/",
    show_bounding_boxes=True,
    show_roi_extraction=True
)
```

#### 2. 效能分析
```python
# 啟用效能分析
from meter_gpt.utils.profiler import enable_profiling

enable_profiling(
    profile_agents=True,
    profile_models=True,
    save_report=True
)
```

#### 3. 日誌分析
```bash
# 分析錯誤日誌
grep "ERROR" logs/meter_gpt.log | tail -100

# 分析效能日誌
grep "processing_time" logs/meter_gpt.log | \
  awk '{print $NF}' | \
  sort -n | \
  tail -10
```

## 開發指南

### 環境設置

#### 1. 開發環境
```bash
# 克隆專案
git clone https://github.com/your-org/meter-gpt.git
cd meter-gpt

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements-dev.txt

# 安裝開發工具
pip install pre-commit black flake8 pytest
pre-commit install
```

#### 2. 測試環境
```bash
# 執行單元測試
pytest tests/unit/

# 執行整合測試
pytest tests/integration/

# 執行效能測試
pytest tests/performance/

# 生成測試報告
pytest --cov=meter_gpt --cov-report=html
```

### 新增代理人

#### 1. 建立代理人類別
```python
# meter_gpt/agents/new_agent.py
from metagpt.roles import Role
from metagpt.actions import Action
from ..utils.logger import get_logger

class NewAgentAction(Action):
    def __init__(self):
        super().__init__()
        self.logger = get_logger("NewAgentAction")
    
    async def run(self, input_data):
        # 實作代理人邏輯
        pass

class NewAgent(Role):
    def __init__(self, name: str = "NewAgent", **kwargs):
        super().__init__(name=name, **kwargs)
        self._init_actions([NewAgentAction()])
        self.logger = get_logger("NewAgent")
    
    async def _act(self):
        # 實作代理人行為
        pass
```

#### 2. 註冊代理人
```python
# meter_gpt/core/orchestrator.py
def _initialize_agents(self):
    # 添加新代理人
    self.agents["new_agent"] = NewAgent(config=self.config)
```

#### 3. 建立測試
```python
# tests/unit/agents/test_new_agent.py
import pytest
from meter_gpt.agents.new_agent import NewAgent

class TestNewAgent:
    @pytest.fixture
    def agent(self):
        return NewAgent()
    
    async def test_agent_action(self, agent):
        # 測試代理人功能
        pass
```

### 貢獻指南

#### 1. 程式碼風格
- 使用 Black 