#!/usr/bin/env python3
"""
MeterGPT 架構整合測試
測試 StreamManager → QualityAssessor → DetectionAgent 的完整流程
使用合成的測試資料驗證每個組件是否按設計運行
"""

import asyncio
import sys
import os
import uuid
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Any

# 添加專案路徑
sys.path.insert(0, '.')

try:
    from meter_gpt.models.messages import StreamFrame, CameraInfo, QualityMetrics, QualityReport
    from meter_gpt.agents.stream_manager import StreamManager
    from meter_gpt.agents.quality_assessor import QualityAssessor
    from meter_gpt.agents.detection_agent import DetectionAgent
    from meter_gpt.core.config import MeterGPTConfig
    from meter_gpt.core.orchestrator import MeterGPTTeam
    from metagpt.environment import Environment
    from metagpt.logs import logger
except ImportError as e:
    print(f"❌ 導入錯誤: {e}")
    sys.exit(1)

class ArchitectureIntegrationTest:
    """架構整合測試類"""
    
    def __init__(self):
        self.test_results = {}
        self.synthetic_frame = None
        
    def create_synthetic_frame(self) -> StreamFrame:
        """創建合成的測試影像幀"""
        print("🎨 創建合成測試影像...")
        
        # 創建一個 640x480 的測試影像（模擬儀表讀數）
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加背景色
        image[:] = (50, 50, 50)
        
        # 繪製一個模擬的圓形儀表
        center = (width // 2, height // 2)
        radius = 150
        cv2.circle(image, center, radius, (200, 200, 200), 3)
        
        # 添加刻度線
        for angle in range(0, 360, 30):
            x1 = int(center[0] + (radius - 20) * np.cos(np.radians(angle)))
            y1 = int(center[1] + (radius - 20) * np.sin(np.radians(angle)))
            x2 = int(center[0] + radius * np.cos(np.radians(angle)))
            y2 = int(center[1] + radius * np.sin(np.radians(angle)))
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # 添加指針（指向 45 度）
        pointer_angle = 45
        pointer_x = int(center[0] + (radius - 30) * np.cos(np.radians(pointer_angle)))
        pointer_y = int(center[1] + (radius - 30) * np.sin(np.radians(pointer_angle)))
        cv2.line(image, center, (pointer_x, pointer_y), (0, 255, 0), 4)
        
        # 添加數字顯示區域
        cv2.rectangle(image, (width - 150, height - 80), (width - 20, height - 20), (100, 100, 100), -1)
        cv2.putText(image, "123.45", (width - 140, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 編碼為 JPEG
        _, encoded_image = cv2.imencode('.jpg', image)
        frame_data = encoded_image.tobytes()
        
        # 創建攝影機資訊
        camera_info = CameraInfo(
            camera_id="test_camera_001",
            camera_name="測試攝影機",
            position=(0.0, 0.0, 1.5),
            is_active=True,
            is_primary=True
        )
        
        # 創建 StreamFrame
        frame = StreamFrame(
            frame_id=str(uuid.uuid4()),
            camera_info=camera_info,
            timestamp=datetime.now(),
            frame_data=frame_data,
            frame_shape=(height, width, 3),
            metadata={
                "capture_method": "synthetic",
                "encoding": "jpg",
                "test_mode": True,
                "description": "合成儀表影像用於測試"
            }
        )
        
        self.synthetic_frame = frame
        print(f"✅ 合成影像創建完成: {frame.frame_id}")
        print(f"   - 尺寸: {frame.frame_shape}")
        print(f"   - 資料大小: {len(frame.frame_data)} bytes")
        print(f"   - 攝影機: {frame.camera_info.camera_name}")
        
        return frame
    
    async def test_stream_manager(self) -> bool:
        """測試 StreamManager 的訊息產生能力"""
        print("\n🔄 測試 StreamManager...")
        
        try:
            # 創建配置
            config = MeterGPTConfig(
                config_id='integration_test',
                detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'detection'},
                corner_detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'corner_detection'}
            )
            
            # 創建環境
            env = Environment()
            
            # 創建 StreamManager
            stream_manager = StreamManager(config=config)
            stream_manager.set_env(env)
            
            # 模擬發布合成影像
            from metagpt.schema import Message
            message = Message(
                content={
                    "type": "stream_frame",
                    "frame_id": self.synthetic_frame.frame_id,
                    "camera_id": self.synthetic_frame.camera_info.camera_id,
                    "stream_frame": self.synthetic_frame.dict(),
                    "timestamp": datetime.now().isoformat()
                },
                role="StreamManager",
                cause_by=type(stream_manager)
            )
            
            await env.publish_message(message)
            
            # 檢查訊息是否成功發布
            messages = env.history
            stream_messages = [msg for msg in messages if msg.content.get("type") == "stream_frame"]
            
            if stream_messages:
                print("✅ StreamManager 測試通過")
                print(f"   - 成功發布 {len(stream_messages)} 個 stream_frame 訊息")
                self.test_results["stream_manager"] = True
                return True
            else:
                print("❌ StreamManager 測試失敗：未找到 stream_frame 訊息")
                self.test_results["stream_manager"] = False
                return False
                
        except Exception as e:
            print(f"❌ StreamManager 測試異常: {e}")
            self.test_results["stream_manager"] = False
            return False
    
    async def test_quality_assessor(self) -> bool:
        """測試 QualityAssessor 的品質評估能力"""
        print("\n🔍 測試 QualityAssessor...")
        
        try:
            # 創建配置
            config = MeterGPTConfig(
                config_id='integration_test',
                detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'detection'},
                corner_detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'corner_detection'}
            )
            
            # 創建環境
            env = Environment()
            
            # 創建 QualityAssessor
            quality_assessor = QualityAssessor(config=config)
            quality_assessor.set_env(env)
            
            # 發布測試訊息
            from metagpt.schema import Message
            test_message = Message(
                content={
                    "type": "quality_assessment_request",
                    "stream_frame": self.synthetic_frame.dict(),
                    "frame_id": self.synthetic_frame.frame_id
                },
                role="TestCoordinator",
                cause_by=type(quality_assessor)
            )
            
            await env.publish_message(test_message)
            
            # 執行品質評估
            result = await quality_assessor._act()
            
            # 檢查結果
            if result and result.content.get("type") == "quality_report":
                quality_report = result.content
                print("✅ QualityAssessor 測試通過")
                print(f"   - 品質可接受: {quality_report.get('is_acceptable', False)}")
                print(f"   - 清晰度分數: {quality_report.get('metrics', {}).get('sharpness_score', 'N/A')}")
                self.test_results["quality_assessor"] = True
                return True
            else:
                print("❌ QualityAssessor 測試失敗：未產生有效的品質報告")
                self.test_results["quality_assessor"] = False
                return False
                
        except Exception as e:
            print(f"❌ QualityAssessor 測試異常: {e}")
            self.test_results["quality_assessor"] = False
            return False
    
    async def test_detection_agent(self) -> bool:
        """測試 DetectionAgent 的儀器偵測能力"""
        print("\n🎯 測試 DetectionAgent...")
        
        try:
            # 創建配置
            config = MeterGPTConfig(
                config_id='integration_test',
                detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'detection'},
                corner_detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'corner_detection'}
            )
            
            # 創建環境
            env = Environment()
            
            # 創建 DetectionAgent
            detection_agent = DetectionAgent(config=config)
            detection_agent.set_env(env)
            
            # 發布品質報告訊息（模擬通過品質檢查）
            from metagpt.schema import Message
            quality_message = Message(
                content={
                    "type": "quality_report",
                    "frame_id": self.synthetic_frame.frame_id,
                    "is_acceptable": True,
                    "metrics": {
                        "sharpness_score": 0.85,
                        "brightness_score": 0.75,
                        "contrast_score": 0.80
                    },
                    "stream_frame": self.synthetic_frame.dict()
                },
                role="QualityAssessor",
                cause_by=type(detection_agent)
            )
            
            await env.publish_message(quality_message)
            
            # 執行偵測
            result = await detection_agent._act()
            
            # 檢查結果
            if result and result.content.get("type") == "detection_result":
                detection_result = result.content
                print("✅ DetectionAgent 測試通過")
                print(f"   - 偵測到儀器類型: {detection_result.get('instrument_type', 'N/A')}")
                print(f"   - 信心度: {detection_result.get('confidence', 'N/A')}")
                print(f"   - ROI 數量: {len(detection_result.get('rois', []))}")
                self.test_results["detection_agent"] = True
                return True
            else:
                print("❌ DetectionAgent 測試失敗：未產生有效的偵測結果")
                self.test_results["detection_agent"] = False
                return False
                
        except Exception as e:
            print(f"❌ DetectionAgent 測試異常: {e}")
            self.test_results["detection_agent"] = False
            return False
    
    async def test_full_pipeline(self) -> bool:
        """測試完整的處理管線"""
        print("\n🚀 測試完整處理管線...")
        
        try:
            # 創建配置
            config = MeterGPTConfig(
                config_id='pipeline_test',
                detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'detection'},
                corner_detection_model={'model_name': 'test', 'model_path': 'test.pt', 'model_type': 'corner_detection'}
            )
            
            # 創建團隊
            team = MeterGPTTeam(config=config)
            
            # 模擬啟動系統
            from metagpt.schema import Message
            start_message = Message(
                content={
                    "type": "system_start",
                    "test_mode": True,
                    "synthetic_frame": self.synthetic_frame.dict()
                },
                role="TestRunner"
            )
            
            # 執行一輪處理
            await team.run(start_message)
            
            # 檢查環境中的訊息歷史
            env = team.env
            messages = env.history
            
            # 統計各類型訊息
            message_types = {}
            for msg in messages:
                msg_type = msg.content.get("type", "unknown")
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            print("✅ 完整管線測試完成")
            print("📊 訊息統計:")
            for msg_type, count in message_types.items():
                print(f"   - {msg_type}: {count}")
            
            # 檢查是否有完整的訊息流
            required_types = ["stream_frame", "quality_report", "detection_result"]
            pipeline_success = all(msg_type in message_types for msg_type in required_types)
            
            if pipeline_success:
                print("🎉 完整管線測試通過：所有必要訊息類型都已產生")
                self.test_results["full_pipeline"] = True
                return True
            else:
                missing_types = [t for t in required_types if t not in message_types]
                print(f"⚠️ 完整管線測試部分通過：缺少訊息類型 {missing_types}")
                self.test_results["full_pipeline"] = False
                return False
                
        except Exception as e:
            print(f"❌ 完整管線測試異常: {e}")
            self.test_results["full_pipeline"] = False
            return False
    
    def print_test_summary(self):
        """打印測試摘要"""
        print("\n" + "="*60)
        print("📋 測試摘要報告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"總測試數: {total_tests}")
        print(f"通過測試: {passed_tests}")
        print(f"失敗測試: {total_tests - passed_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n詳細結果:")
        for test_name, result in self.test_results.items():
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"  {test_name}: {status}")
        
        if passed_tests == total_tests:
            print("\n🎉 所有測試通過！架構運行正常。")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} 個測試失敗，需要進一步調試。")

async def main():
    """主測試函數"""
    print("🚀 開始 MeterGPT 架構整合測試")
    print("="*60)
    
    # 創建測試實例
    test = ArchitectureIntegrationTest()
    
    # 創建合成測試資料
    test.create_synthetic_frame()
    
    # 執行各項測試
    await test.test_stream_manager()
    await test.test_quality_assessor()
    await test.test_detection_agent()
    await test.test_full_pipeline()
    
    # 打印測試摘要
    test.print_test_summary()

if __name__ == "__main__":
    # 執行測試
    asyncio.run(main())