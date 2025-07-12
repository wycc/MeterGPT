# MeterGPT - 智慧儀器讀值系統

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MetaGPT](https://img.shields.io/badge/MetaGPT-0.8%2B-green.svg)](https://github.com/geekan/MetaGPT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%2B-yellow.svg)](https://ultralytics.com/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7%2B-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 項目簡介

MeterGPT 是一個基於 MetaGPT 多代理人協作架構的智慧儀器讀值系統，專為自動化儀器監控和資料擷取而設計。系統採用「Code = SOP(Team)」的設計理念，將精心編排的標準作業程序實體化，確保各模組協同作業以實現可靠、零遺漏、零誤抄的儀器讀值目標。

### 🎯 核心理念

- **多代理人協作**：基於 MetaGPT 框架，將複雜的儀器讀值任務分解為多個專業代理人
- **標準作業程序**：實現精確的 SOP 流程，確保處理的一致性和可靠性
- **智慧備援機制**：多層次的容錯和備援策略，保證系統高可用性
- **可追溯性**：完整的處理記錄和影像封存，滿足稽核需求

### 🏥 應用場景

- **醫療設備監控**：手術室、ICU 等關鍵環境的儀器讀值
- **工業自動化**：生產線上的儀表監控和資料收集
- **實驗室管理**：科研設備的自動化資料記錄
- **能源監控**：電力、水務等基礎設施的儀表讀值

## ✨ 核心特性

### 🤖 多代理人協作
- **專業分工**：每個代理人專注於特定任務，提高處理效率
- **智慧協調**：Orchestrator 統一管理代理人間的協作流程
- **動態調度**：根據系統負載和任務優先級動態分配資源

### 📹 多攝影機支援
- **同時管理**：支援最多 8 個攝影機同時串流處理
- **主備切換**：智慧的主要和備援攝影機動態切換機制
- **PTZ 控制**：支援 Pan-Tilt-Zoom 攝影機的遠程控制

### 🔍 智慧偵測識別
- **多類型儀器**：支援數位顯示器、七段顯示器、LCD 螢幕、類比儀表等
- **精確定位**：使用 YOLO 模型進行儀器偵測和角點定位
- **透視校正**：自動進行透視變換，獲得標準化的讀值區域

### 🛡️ 多層備援機制
- **攝影機備援**：自動切換到品質更好的備援攝影機
- **演算法備援**：OCR 失敗時啟用 VLM 視覺語言模型
- **人工審核**：最終備援的人工介入機制

### ⚡ 即時品質評估
- **健康分數**：即時計算影像品質的綜合健康分數
- **多維度評估**：清晰度、亮度、對比度、遮擋比例等多項指標
- **趨勢分析**：品質變化趨勢監控和預警

### 🔄 自動容錯恢復
- **智慧重試**：根據失敗類型選擇最佳的重試策略
- **優雅降級**：在部分功能失效時維持核心服務
- **自動恢復**：系統異常後的自動恢復機制

## 🏗️ 系統架構

### 整體架構圖

```mermaid
graph TB
    subgraph "外部介面層"
        API[REST API]
        WS[WebSocket]
        DB[(資料庫)]
    end
    
    subgraph "核心協調層"
        ORCH[Orchestrator<br/>系統協調器]
    end
    
    subgraph "代理人生態系統"
        SM[StreamManager<br/>串流管理]
        QA[QualityAssessor<br/>品質評估]
        DA[DetectionAgent<br/>儀器偵測]
        OA[OCRAgent<br/>文字識別]
        VA[ValidationAgent<br/>結果驗證]
        FA[FallbackAgent<br/>備援處理]
    end
    
    subgraph "整合層"
        YOLO[YOLO Wrapper]
        PADDLE[PaddleOCR Wrapper]
        CV[OpenCV Utils]
        VLM[VLM APIs]
    end
    
    subgraph "硬體層"
        CAM1[攝影機 1]
        CAM2[攝影機 2]
        CAMN[攝影機 N]
    end
    
    API --> ORCH
    WS --> ORCH
    ORCH --> SM
    ORCH --> QA
    ORCH --> DA
    ORCH --> OA
    ORCH --> VA
    ORCH --> FA
    
    SM --> QA
    QA --> DA
    DA --> OA
    OA --> VA
    VA --> FA
    
    DA --> YOLO
    OA --> PADDLE
    SM --> CV
    FA --> VLM
    
    SM --> CAM1
    SM --> CAM2
    SM --> CAMN
    
    ORCH --> DB
```

### 核心組件說明

| 組件 | 功能描述 | 主要技術 |
|------|----------|----------|
| **Orchestrator** | 系統核心協調器，管理整個處理流程 | MetaGPT, AsyncIO |
| **StreamManager** | 串流管理代理人，處理多攝影機串流 | OpenCV, Threading |
| **QualityAssessor** | 品質評估代理人，計算影像健康分數 | 影像處理演算法 |
| **DetectionAgent** | 偵測代理人，執行儀器和角點偵測 | YOLO, 深度學習 |
| **OCRAgent** | OCR代理人，執行文字識別 | PaddleOCR, Tesseract |
| **ValidationAgent** | 驗證代理人，驗證識別結果合理性 | 規則引擎, 統計分析 |
| **FallbackAgent** | 備援代理人，處理異常和備援策略 | VLM APIs, PTZ控制 |

### 處理流程

```mermaid
sequenceDiagram
    participant C as 攝影機
    participant SM as StreamManager
    participant QA as QualityAssessor
    participant DA as DetectionAgent
    participant OA as OCRAgent
    participant VA as ValidationAgent
    participant FA as FallbackAgent
    
    C->>SM: 影像串流
    SM->>QA: 影像幀
    QA->>QA: 品質評估
    QA->>DA: 高品質影像
    DA->>DA: 儀器偵測
    DA->>DA: 角點偵測
    DA->>OA: ROI 區域
    OA->>OA: OCR 識別
    OA->>VA: 識別結果
    VA->>VA: 結果驗證
    
    alt 驗證通過
        VA->>SM: 最終結果
    else 驗證失敗
        VA->>FA: 啟動備援
        FA->>FA: 備援決策
        alt 攝影機切換
            FA->>SM: 切換攝影機
        else VLM 備援
            FA->>FA: VLM 識別
        end
    end
```

## 🚀 快速開始

### 環境要求

- **Python**: 3.8 或更高版本
- **作業系統**: Linux (推薦 Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **硬體要求**:
  - CPU: 8 核心以上
  - RAM: 16GB 以上
  - GPU: NVIDIA RTX 3060 以上 (可選，用於加速)
  - 儲存: 500GB SSD

### 安裝步驟

1. **克隆專案**
```bash
git clone https://github.com/your-org/meter-gpt.git
cd meter-gpt
```

2. **建立虛擬環境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

3. **安裝依賴套件**
```bash
pip install -r requirements.txt
```

4. **下載模型檔案**
```bash
# 下載 YOLO 模型
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -O models/instrument_yolo.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O models/corner_yolo.pt
```

5. **配置系統**
```bash
cp config/meter_gpt_config.example.yaml config/meter_gpt_config.yaml
# 編輯配置檔案以符合您的環境
```

### 基本使用範例

#### 1. 驗證配置
```bash
python main.py validate --config config/meter_gpt_config.yaml
```

#### 2. 處理單一影像
```bash
python main.py process --image test_images/meter.jpg --camera cam_001
```

#### 3. 啟動系統服務
```bash
python main.py run --config config/meter_gpt_config.yaml
```

#### 4. 使用 Python API
```python
import asyncio
from meter_gpt.core.orchestrator import MeterGPTOrchestrator
from meter_gpt.core.config import load_config

async def main():
    # 載入配置
    config = load_config("config/meter_gpt_config.yaml")
    
    # 初始化系統
    orchestrator = await MeterGPTOrchestrator.create(config)
    
    # 處理影像
    with open("test_image.jpg", "rb") as f:
        frame_data = f.read()
    
    result = await orchestrator.process_frame("cam_001", frame_data)
    print(f"讀值結果: {result.final_reading}")
    print(f"信心度: {result.confidence}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ 技術棧

### 核心框架
- **[MetaGPT](https://github.com/geekan/MetaGPT)**: 多代理人協作框架
- **[AsyncIO](https://docs.python.org/3/library/asyncio.html)**: 異步處理和並發控制
- **[Pydantic](https://pydantic-docs.helpmanual.io/)**: 資料驗證和設定管理

### 深度學習和電腦視覺
- **[PyTorch](https://pytorch.org/)**: 深度學習框架
- **[Ultralytics YOLO](https://ultralytics.com/)**: 物件偵測模型
- **[OpenCV](https://opencv.org/)**: 電腦視覺處理
- **[scikit-image](https://scikit-image.org/)**: 影像處理演算法

### 資料處理和驗證
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: 光學字符識別
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)**: 備援 OCR 引擎
- **[pandas](https://pandas.pydata.org/)**: 資料處理和分析
- **[scikit-learn](https://scikit-learn.org/)**: 機器學習工具

### 異步處理和網路
- **[FastAPI](https://fastapi.tiangolo.com/)**: 現代 Web API 框架
- **[WebSockets](https://websockets.readthedocs.io/)**: 即時通訊
- **[aiohttp](https://docs.aiohttp.org/)**: 異步 HTTP 客戶端/伺服器

## 📁 項目結構

```
meter_gpt/
├── 📁 meter_gpt/                 # 核心套件
│   ├── 📁 agents/                # 代理人模組
│   │   ├── 📄 stream_manager.py      # 串流管理代理人
│   │   ├── 📄 quality_assessor.py    # 品質評估代理人
│   │   ├── 📄 detection_agent.py     # 偵測代理人
│   │   ├── 📄 ocr_agent.py           # OCR 代理人
│   │   ├── 📄 validation_agent.py    # 驗證代理人
│   │   └── 📄 fallback_agent.py      # 備援代理人
│   ├── 📁 core/                  # 核心模組
│   │   ├── 📄 orchestrator.py        # 系統協調器
│   │   └── 📄 config.py              # 配置管理
│   ├── 📁 integrations/          # 第三方整合
│   │   ├── 📄 yolo_wrapper.py        # YOLO 模型封裝
│   │   ├── 📄 paddle_ocr_wrapper.py  # PaddleOCR 封裝
│   │   └── 📄 opencv_utils.py        # OpenCV 工具
│   ├── 📁 models/                # 資料模型
│   │   └── 📄 messages.py            # 訊息和資料結構
│   └── 📁 utils/                 # 工具模組
│       └── 📄 logger.py              # 日誌管理
├── 📁 docs/                      # 文檔
│   ├── 📁 agents/                # 代理人文檔
│   ├── 📁 architecture/          # 架構文檔
│   └── 📁 core/                  # 核心模組文檔
├── 📁 config/                    # 配置檔案
├── 📁 models/                    # 模型檔案
├── 📁 tests/                     # 測試檔案
├── 📄 main.py                    # 主程式入口
├── 📄 requirements.txt           # 依賴套件
└── 📄 README.md                  # 專案說明
```

### 核心模組介紹

#### 代理人模組 (`meter_gpt/agents/`)
- **StreamManager**: 管理多攝影機串流，提供影像緩衝和品質監控
- **QualityAssessor**: 評估影像品質，計算健康分數和趨勢分析
- **DetectionAgent**: 執行儀器偵測、角點偵測和透視校正
- **OCRAgent**: 多引擎 OCR 識別，支援各種顯示器類型
- **ValidationAgent**: 結果驗證，執行多重驗證規則
- **FallbackAgent**: 備援處理，智慧決策和容錯恢復

#### 核心模組 (`meter_gpt/core/`)
- **Orchestrator**: 系統協調器，管理代理人協作流程
- **Config**: 配置管理，支援 YAML 配置和動態更新

#### 整合模組 (`meter_gpt/integrations/`)
- **YOLO Wrapper**: YOLO 模型的統一介面和最佳化
- **PaddleOCR Wrapper**: PaddleOCR 的封裝和配置管理
- **OpenCV Utils**: OpenCV 的常用工具和影像處理函數

## ⚙️ 配置說明

### 配置檔案格式

系統使用 YAML 格式的配置檔案，主要包含以下部分：

```yaml
# 系統基本配置
system:
  config_id: "production_v1"
  environment: "production"
  version: "1.0.0"
  log_level: "INFO"
  log_file: "logs/meter_gpt.log"
  
  processing:
    queue_size: 100
    max_concurrent_tasks: 10
    default_timeout: 30

# 攝影機配置
cameras:
  - camera_id: "cam_001"
    camera_name: "主要攝影機"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    is_primary: true
    resolution: [1920, 1080]
    fps: 30
    
  - camera_id: "cam_002"
    camera_name: "備援攝影機"
    rtsp_url: "rtsp://192.168.1.101:554/stream"
    is_primary: false
    resolution: [1920, 1080]
    fps: 30

# 模型配置
models:
  detection_model:
    model_name: "instrument_detector"
    model_path: "./models/instrument_yolo.pt"
    model_type: "detection"
    confidence_threshold: 0.7
    device: "cuda"
    
  corner_detection_model:
    model_name: "corner_detector"
    model_path: "./models/corner_yolo.pt"
    model_type: "corner_detection"
    confidence_threshold: 0.8
    device: "cuda"

# OCR 配置
ocr:
  paddle_ocr:
    language: "ch"
    use_gpu: true
    confidence_threshold: 0.7
    
  easy_ocr:
    languages: ["ch_sim", "en"]
    gpu: true
    confidence_threshold: 0.6

# 驗證規則配置
validation:
  overall_threshold: 0.7
  rules:
    - rule_id: "numeric_format"
      enabled: true
      parameters:
        pattern: "^-?\\d+\\.?\\d*$"
        
    - rule_id: "range_check"
      enabled: true
      parameters:
        min_value: 0
        max_value: 9999

# 備援配置
fallback:
  vlm:
    enabled: true
    model_name: "gpt-4-vision-preview"
    api_key: "${OPENAI_API_KEY}"
    confidence_threshold: 0.7
    
  camera_switch:
    enabled: true
    quality_threshold: 0.4
    
  ptz_control:
    enabled: false
    adjustment_steps: 5
```

### 主要參數說明

#### 系統參數
- `queue_size`: 處理佇列大小，控制系統吞吐量
- `max_concurrent_tasks`: 最大並發任務數
- `default_timeout`: 預設處理超時時間

#### 攝影機參數
- `rtsp_url`: RTSP 串流地址
- `is_primary`: 是否為主要攝影機
- `resolution`: 影像解析度
- `fps`: 幀率設定

#### 模型參數
- `confidence_threshold`: 信心度閾值
- `device`: 運算設備 (cpu/cuda)
- `model_path`: 模型檔案路徑

## 🔌 API 文檔

### REST API 介面

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

**回應格式:**
```json
{
  "frame_id": "frame_12345",
  "camera_id": "cam_001",
  "status": "success",
  "final_reading": "123.45",
  "confidence": 0.92,
  "processing_time": 2.1,
  "timestamp": "2024-01-01T12:00:00Z",
  "metadata": {
    "instrument_type": "digital_display",
    "quality_score": 0.85,
    "fallback_used": false
  }
}
```

#### 取得系統狀態
```http
GET /api/v1/status
```

**回應格式:**
```json
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
  "active_cameras": ["cam_001", "cam_002"],
  "processing_queue_size": 5,
  "system_metrics": {
    "total_processed": 1250,
    "success_rate": 0.94,
    "average_processing_time": 2.3,
    "system_health": 0.96
  }
}
```

#### 取得歷史資料
```http
GET /api/v1/history?camera_id=cam_001&hours=24&page=1&page_size=100
```

**回應格式:**
```json
{
  "results": [
    {
      "frame_id": "frame_001",
      "camera_id": "cam_001",
      "final_reading": "123.45",
      "confidence": 0.92,
      "timestamp": "2024-01-01T12:00:00Z",
      "processing_time": 2.1,
      "instrument_type": "digital_display"
    }
  ],
  "total_count": 1440,
  "page": 1,
  "page_size": 100,
  "total_pages": 15
}
```

#### 系統控制
```http
# 重新載入配置
POST /api/v1/config/reload

# 更新代理人配置
PUT /api/v1/agents/{agent_name}/config
Content-Type: application/json

{
  "confidence_threshold": 0.8,
  "enabled": true
}

# 攝影機控制
POST /api/v1/cameras/{camera_id}/control
Content-Type: application/json

{
  "action": "switch_primary",
  "parameters": {}
}
```

### WebSocket 介面

#### 即時串流處理
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

// 發送處理請求
ws.send(JSON.stringify({
  "action": "start_processing",
  "camera_id": "cam_001",
  "options": {
    "continuous": true,
    "interval": 5,
    "quality_threshold": 0.6
  }
}));

// 接收處理結果
ws.onmessage = function(event) {
  const result = JSON.parse(event.data);
  console.log('處理結果:', result);
  
  // 處理不同類型的訊息
  switch(result.type) {
    case 'processing_result':
      updateDisplay(result.data);
      break;
    case 'system_status':
      updateSystemStatus(result.data);
      break;
    case 'error':
      handleError(result.data);
      break;
  }
};

// 停止處理
ws.send(JSON.stringify({
  "action": "stop_processing",
  "camera_id": "cam_001"
}));
```

#### 系統監控
```javascript
const monitorWs = new WebSocket('ws://localhost:8000/ws/monitor');

monitorWs.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  // 即時系統指標
  if (data.type === 'metrics') {
    updateMetrics(data.metrics);
  }
  
  // 代理人狀態更新
  if (data.type === 'agent_status') {
    updateAgentStatus(data.agent, data.status);
  }
  
  // 警告和錯誤
  if (data.type === 'alert') {
    showAlert(data.level, data.message);
  }
};
```

## 🚀 部署指南

### 單機部署

#### 使用 Python 直接部署
```bash
# 1. 準備環境
git clone https://github.com/your-org/meter-gpt.git
cd meter-gpt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 配置系統
cp config/meter_gpt_config.example.yaml config/meter_gpt_config.yaml
# 編輯配置檔案

# 3. 下載模型
mkdir -p models
# 下載必要的模型檔案

# 4. 啟動系統
python main.py run --config config/meter_gpt_config.yaml
```

#### 硬體需求
- **CPU**: Intel i7-8700K 或 AMD Ryzen 7 2700X 以上
- **RAM**: 16GB DDR4 以上
- **GPU**: NVIDIA RTX 3060 以上 (可選)
- **儲存**: 500GB NVMe SSD
- **網路**: Gigabit Ethernet

### 分散式部署

#### 邊緣節點配置
```yaml
# edge_config.yaml
system:
  deployment_mode: "edge"
  node_id: "edge_001"
  
agents:
  enabled:
    - stream_manager
    - quality_assessor
    - detection_agent
  
central_node:
  host: "192.168.1.10"
  port: 8000
  api_key: "${CENTRAL_API_KEY}"
```

#### 中央節點配置
```yaml
# central_config.yaml
system:
  deployment_mode: "central"
  
agents:
  enabled:
    - orchestrator
    - ocr_agent
    - validation_agent
    - fallback_agent
  
edge_nodes:
  - node_id: "edge_001"
    host: "192.168.1.11"
    capabilities: ["stream", "detection"]
```

### Docker 部署

#### Dockerfile
```dockerfile
FROM python:3.9-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式
COPY . .

# 建立必要目錄
RUN mkdir -p logs models config

# 設置環境變數
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# 暴露端口
EXPOSE 8000

# 啟動命令
CMD ["python", "main.py", "run", "--config", "config/meter_gpt_config.yaml"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  meter-gpt:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
      - ./models:/app/models
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: meter_gpt
      POSTGRES_USER: meter_gpt
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

#### 部署命令
```bash
# 1. 準備環境變數
cp .env.example .env
# 編輯 .env 檔案

# 2. 建置和啟動
docker-compose up -d

# 3. 檢查狀態
docker-compose ps
docker-compose logs meter-gpt

# 4. 擴展服務
docker
```bash
# 1. 準備環境變數
cp .env.example .env
# 編輯 .env 檔案

# 2. 建置和啟動
docker-compose up -d

# 3. 檢查狀態
docker-compose ps
docker-compose logs meter-gpt

# 4. 擴展服務
docker-compose up -d --scale meter-gpt=3
```

### Kubernetes 部署

#### 部署清單
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: meter-gpt
  labels:
    app: meter-gpt
spec:
  replicas: 3
  selector:
    matchLabels:
      app: meter-gpt
  template:
    metadata:
      labels:
        app: meter-gpt
    spec:
      containers:
      - name: meter-gpt
        image: meter-gpt:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: meter-gpt-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: models-volume
          mountPath: /app/models
      volumes:
      - name: config-volume
        configMap:
          name: meter-gpt-config
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: meter-gpt-service
spec:
  selector:
    app: meter-gpt
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 👨‍💻 開發指南

### 開發環境設置

#### 1. 開發依賴安裝
```bash
# 安裝開發依賴
pip install -r requirements-dev.txt

# 安裝 pre-commit hooks
pre-commit install

# 設置 IDE 配置
cp .vscode/settings.example.json .vscode/settings.json
```

#### 2. 開發工具配置
```bash
# 代碼格式化
black meter_gpt/ tests/
isort meter_gpt/ tests/

# 代碼檢查
flake8 meter_gpt/
mypy meter_gpt/

# 測試執行
pytest tests/ -v --cov=meter_gpt
```

### 代碼規範

#### Python 代碼風格
- 遵循 **PEP 8** 標準
- 使用 **Black** 進行代碼格式化
- 使用 **isort** 進行 import 排序
- 使用 **type hints** 進行類型註解

#### 命名規範
```python
# 類別名稱：PascalCase
class StreamManager:
    pass

# 函數和變數：snake_case
def process_frame(camera_id: str) -> ProcessingResult:
    frame_data = get_frame_data()
    return result

# 常數：UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30

# 私有成員：前綴底線
class Agent:
    def __init__(self):
        self._internal_state = {}
        self.__private_method()
```

#### 文檔字串規範
```python
def process_frame(self, camera_id: str, frame_data: bytes) -> ProcessingResult:
    """
    處理單一影像幀
    
    Args:
        camera_id: 攝影機識別碼
        frame_data: 影像二進位資料
        
    Returns:
        ProcessingResult: 處理結果，包含讀值和信心度
        
    Raises:
        ValueError: 當 camera_id 無效時
        ProcessingError: 當處理失敗時
        
    Example:
        >>> result = agent.process_frame("cam_001", frame_data)
        >>> print(f"讀值: {result.final_reading}")
    """
```

### 測試指南

#### 單元測試
```python
# tests/unit/agents/test_detection_agent.py
import pytest
from unittest.mock import Mock, patch
from meter_gpt.agents.detection_agent import DetectionAgent
from meter_gpt.models.messages import DetectionResult

class TestDetectionAgent:
    @pytest.fixture
    def agent(self):
        config = Mock()
        return DetectionAgent(config)
    
    @pytest.mark.asyncio
    async def test_detect_instrument_success(self, agent):
        # 準備測試資料
        frame_data = b"fake_image_data"
        
        # 模擬 YOLO 偵測結果
        with patch.object(agent.yolo_wrapper, 'detect') as mock_detect:
            mock_detect.return_value = [
                {"class": "digital_display", "confidence": 0.9, "bbox": [100, 100, 200, 200]}
            ]
            
            # 執行測試
            result = await agent.detect_instrument(frame_data)
            
            # 驗證結果
            assert result.success
            assert result.instrument_type == "digital_display"
            assert result.confidence >= 0.9
```

#### 整合測試
```python
# tests/integration/test_full_pipeline.py
import pytest
from meter_gpt.core.orchestrator import MeterGPTOrchestrator
from meter_gpt.core.config import load_config

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        # 載入測試配置
        config = load_config("tests/fixtures/test_config.yaml")
        
        # 初始化系統
        orchestrator = await MeterGPTOrchestrator.create(config)
        
        # 載入測試影像
        with open("tests/fixtures/test_meter.jpg", "rb") as f:
            frame_data = f.read()
        
        # 執行完整處理流程
        result = await orchestrator.process_frame("test_camera", frame_data)
        
        # 驗證結果
        assert result.status == ProcessingStatus.SUCCESS
        assert result.final_reading is not None
        assert result.confidence > 0.5
```

#### 效能測試
```python
# tests/performance/test_throughput.py
import asyncio
import time
import pytest
from meter_gpt.core.orchestrator import MeterGPTOrchestrator

class TestPerformance:
    @pytest.mark.asyncio
    async def test_processing_throughput(self):
        orchestrator = await MeterGPTOrchestrator.create(test_config)
        
        # 準備測試資料
        test_frames = [load_test_frame(i) for i in range(100)]
        
        # 測量處理時間
        start_time = time.time()
        
        tasks = [
            orchestrator.process_frame(f"cam_{i%4}", frame)
            for i, frame in enumerate(test_frames)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # 計算效能指標
        total_time = end_time - start_time
        fps = len(test_frames) / total_time
        success_rate = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS) / len(results)
        
        # 驗證效能要求
        assert fps >= 10  # 至少 10 FPS
        assert success_rate >= 0.9  # 90% 成功率
```

### 貢獻指南

#### 1. 開發流程
```bash
# 1. Fork 專案並克隆
git clone https://github.com/your-username/meter-gpt.git
cd meter-gpt

# 2. 建立功能分支
git checkout -b feature/new-agent

# 3. 開發和測試
# ... 進行開發 ...
pytest tests/
black meter_gpt/
flake8 meter_gpt/

# 4. 提交變更
git add .
git commit -m "feat: 新增 XXX 代理人功能"

# 5. 推送並建立 Pull Request
git push origin feature/new-agent
```

#### 2. Commit 訊息規範
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**類型 (type):**
- `feat`: 新功能
- `fix`: 錯誤修復
- `docs`: 文檔更新
- `style`: 代碼格式調整
- `refactor`: 代碼重構
- `test`: 測試相關
- `chore`: 建置或輔助工具變更

**範例:**
```
feat(agents): 新增 QualityAssessor 代理人

- 實作影像品質評估演算法
- 支援多維度品質指標計算
- 新增品質趨勢分析功能

Closes #123
```

#### 3. Pull Request 檢查清單
- [ ] 代碼通過所有測試
- [ ] 新功能包含對應的測試
- [ ] 文檔已更新
- [ ] 代碼風格符合規範
- [ ] Commit 訊息清晰明確
- [ ] 無重大效能影響

## 🔧 故障排除

### 常見問題

#### 1. 系統啟動失敗

**問題描述**: 系統無法正常啟動，出現模組導入錯誤

**可能原因**:
- Python 環境配置不正確
- 依賴套件未正確安裝
- CUDA 環境配置問題

**解決方案**:
```bash
# 檢查 Python 版本
python --version  # 應該是 3.8+

# 檢查虛擬環境
which python
which pip

# 重新安裝依賴
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# 檢查 CUDA 環境
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. 攝影機連接失敗

**問題描述**: 無法連接到 RTSP 攝影機串流

**可能原因**:
- 網路連接問題
- RTSP URL 配置錯誤
- 攝影機認證失敗

**解決方案**:
```bash
# 測試網路連接
ping 192.168.1.100

# 測試 RTSP 串流
ffmpeg -i rtsp://192.168.1.100:554/stream -t 10 -f null -

# 檢查攝影機配置
curl -u admin:password http://192.168.1.100/api/status
```

**配置檢查**:
```yaml
cameras:
  - camera_id: "cam_001"
    rtsp_url: "rtsp://username:password@192.168.1.100:554/stream"
    timeout: 30
    retry_count: 3
```

#### 3. 模型載入失敗

**問題描述**: YOLO 或 OCR 模型無法正確載入

**可能原因**:
- 模型檔案路徑錯誤
- 模型檔案損壞
- GPU 記憶體不足

**解決方案**:
```bash
# 檢查模型檔案
ls -la models/
file models/instrument_yolo.pt

# 測試模型載入
python -c "
from ultralytics import YOLO
model = YOLO('models/instrument_yolo.pt')
print('模型載入成功')
"

# 檢查 GPU 記憶體
nvidia-smi
```

#### 4. 處理效能低下

**問題描述**: 系統處理速度過慢，無法達到預期 FPS

**可能原因**:
- 硬體資源不足
- 配置參數不當
- 模型過於複雜

**解決方案**:
```bash
# 監控系統資源
htop
nvidia-smi -l 1

# 調整配置參數
# 在 config.yaml 中：
processing:
  max_concurrent_tasks: 4  # 降低並發數
  queue_size: 50          # 減少佇列大小

models:
  detection_model:
    device: "cpu"          # 使用 CPU 處理
    confidence_threshold: 0.8  # 提高閾值
```

#### 5. OCR 識別準確率低

**問題描述**: OCR 識別結果不準確或信心度過低

**可能原因**:
- 影像品質不佳
- OCR 引擎配置不當
- 儀器類型不匹配

**解決方案**:
```bash
# 啟用除錯模式
export METER_GPT_DEBUG=1
python main.py process --image test.jpg --camera test

# 檢查影像品質
python -c "
from meter_gpt.agents.quality_assessor import QualityAssessor
assessor = QualityAssessor()
score = assessor.assess_quality(image)
print(f'品質分數: {score}')
"
```

**配置調整**:
```yaml
ocr:
  paddle_ocr:
    confidence_threshold: 0.5  # 降低閾值
    use_gpu: true
    det_db_thresh: 0.3
    det_db_box_thresh: 0.5
    
validation:
  rules:
    - rule_id: "confidence_check"
      parameters:
        min_confidence: 0.6  # 調整最小信心度
```

### 除錯工具

#### 1. 日誌分析
```bash
# 即時查看日誌
tail -f logs/meter_gpt.log

# 搜尋錯誤
grep "ERROR" logs/meter_gpt.log

# 分析特定代理人日誌
grep "DetectionAgent" logs/meter_gpt.log | tail -20
```

#### 2. 效能分析
```bash
# 啟用效能分析
export METER_GPT_PROFILE=1
python main.py run --config config.yaml

# 分析效能報告
python -m cProfile -o profile.stats main.py process --image test.jpg
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

#### 3. 視覺化除錯
```python
# 啟用視覺化除錯模式
from meter_gpt.utils.debug import enable_visual_debug

enable_visual_debug(
    save_intermediate=True,
    output_dir="debug_output/",
    show_bboxes=True,
    show_roi=True
)
```

### 監控和警報

#### 1. 系統監控
```bash
# 檢查系統狀態
curl -s http://localhost:8000/api/v1/status | jq .

# 監控處理佇列
watch -n 1 'curl -s http://localhost:8000/api/v1/metrics | jq .processing_queue_size'

# 檢查代理人健康狀態
curl -s http://localhost:8000/api/v1/agents/status | jq .
```

#### 2. 警報設置
```yaml
# config/alerts.yaml
alerts:
  processing_queue_size:
    threshold: 50
    action: "email"
    recipients: ["admin@company.com"]
    
  success_rate:
    threshold: 0.8
    window: "5m"
    action: "webhook"
    url: "https://hooks.slack.com/..."
    
  agent_failure:
    threshold: 3
    window: "1m"
    action: "restart_agent"
```

## 📄 授權和致謝

### 授權條款

本專案採用 [MIT License](LICENSE) 授權條款。

```
MIT License

Copyright (c) 2025 MeterGPT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 技術致謝

我們感謝以下開源專案和技術社群的貢獻：

#### 核心框架
- **[MetaGPT](https://github.com/geekan/MetaGPT)** - 多代理人協作框架
- **[PyTorch](https://pytorch.org/)** - 深度學習框架
- **[OpenCV](https://opencv.org/)** - 電腦視覺處理庫

#### 深度學習模型
- **[Ultralytics YOLO](https://ultralytics.com/)** - 物件偵測模型
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** - 光學字符識別
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** - 多語言 OCR 引擎

#### 開發工具
- **[FastAPI](https://fastapi.tiangolo.com/)** - 現代 Web API 框架
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - 資料驗證和設定管理
- **[pytest](https://pytest.org/)** - Python 測試框架

#### 特別感謝
- **醫療設備製造商** - 提供儀器規格和測試資料
- **開源社群** - 持續的技術支援和回饋
- **研究機構** - 演算法研究和最佳化建議

### 貢獻者

感謝所有為 MeterGPT 專案做出貢獻的開發者和研究人員。

### 聯絡資訊

- **專案首頁**: https://github.com/your-org/meter-gpt
- **問題回報**: https://github.com/your-org/meter-gpt/issues
- **技術討論**: https://github.com/your-org/meter-gpt/discussions
- **電子郵件**: meter-gpt@your-org.com

### 引用

如果您在研究或專案中使用了 MeterGPT，請引用：

```bibtex
@software{meter_gpt_2025,
  title={MeterGPT: Intelligent Instrument Reading System with Multi-Agent Collaboration},
  author={MeterGPT Team},
  year={2025},
  url={https://github.com/your-org/meter-gpt},
  version={1.0.0}
}
```

---

<div align="center">

**🚀 讓智慧儀器讀值變得簡單可靠 🚀**

[開始使用](#快速開始) • [查看文檔](docs/) • [回報問題](https://github.com/your-org/meter-gpt/issues) • [參與貢獻](#貢獻指南)

</div>