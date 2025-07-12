#!/usr/bin/env python3
"""
MeterGPT 系統主程式入口點
提供命令列介面和系統啟動功能
"""

import asyncio
import argparse
import sys
import os
import signal
from pathlib import Path
from typing import Optional
import logging

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from meter_gpt.core.config import load_config, get_config
from meter_gpt.core.orchestrator import get_orchestrator, initialize_system, shutdown_system
from meter_gpt.utils.logger import get_logger, setup_logging
from meter_gpt.models.messages import SystemStatus


class MeterGPTApplication:
    """MeterGPT 應用程式主類別"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化應用程式
        
        Args:
            config_path: 配置檔案路徑
        """
        self.config_path = config_path
        self.config = None
        self.orchestrator = None
        self.logger = None
        self.is_running = False
        
        # 設置信號處理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信號處理器"""
        if self.logger:
            self.logger.info(f"收到信號 {signum}，正在關閉系統...")
        self.is_running = False
    
    async def initialize(self):
        """初始化系統"""
        try:
            # 載入配置
            self.config = load_config(self.config_path)
            
            # 設置日誌
            setup_logging(
                level=self.config.system.log_level,
                log_file=self.config.system.log_file
            )
            self.logger = get_logger("MeterGPTApplication")
            
            self.logger.info("=" * 60)
            self.logger.info(f"MeterGPT 系統 v{self.config.system.version} 啟動中...")
            self.logger.info("=" * 60)
            
            # 驗證配置
            is_valid, errors = self.config.validate_config() if hasattr(self.config, 'validate_config') else (True, [])
            if not is_valid:
                self.logger.error("配置驗證失敗:")
                for error in errors:
                    self.logger.error(f"  - {error}")
                return False
            
            # 初始化協調器
            self.orchestrator = await initialize_system(self.config)
            self.logger.info("系統初始化完成")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"系統初始化失敗: {e}")
            else:
                print(f"系統初始化失敗: {e}")
            return False
    
    async def run(self):
        """運行系統"""
        if not await self.initialize():
            return False
        
        try:
            self.is_running = True
            self.logger.info("MeterGPT 系統開始運行")
            
            # 主運行循環
            while self.is_running:
                try:
                    # 檢查系統狀態
                    status = self.orchestrator.get_system_status()
                    
                    # 每分鐘輸出一次狀態
                    await asyncio.sleep(60)
                    
                    if self.is_running:  # 再次檢查，避免關閉時輸出
                        self._log_system_status(status)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"主循環錯誤: {e}")
                    await asyncio.sleep(5)
            
            return True
            
        except Exception as e:
            self.logger.error(f"系統運行錯誤: {e}")
            return False
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """關閉系統"""
        if self.logger:
            self.logger.info("正在關閉 MeterGPT 系統...")
        
        try:
            if self.orchestrator:
                await shutdown_system()
            
            if self.logger:
                self.logger.info("MeterGPT 系統已安全關閉")
        except Exception as e:
            if self.logger:
                self.logger.error(f"系統關閉時發生錯誤: {e}")
    
    def _log_system_status(self, status: SystemStatus):
        """記錄系統狀態"""
        self.logger.info(
            f"系統狀態 - "
            f"活躍攝影機: {len(status.active_cameras)}, "
            f"佇列大小: {status.processing_queue_size}, "
            f"成功率: {status.success_rate:.1%}, "
            f"平均處理時間: {status.average_processing_time:.2f}s, "
            f"系統健康度: {status.system_health:.1%}"
        )
    
    async def process_single_image(self, image_path: str, camera_id: str = "test_camera"):
        """
        處理單一影像檔案
        
        Args:
            image_path: 影像檔案路徑
            camera_id: 攝影機 ID
        """
        if not await self.initialize():
            return
        
        try:
            # 讀取影像檔案
            with open(image_path, 'rb') as f:
                frame_data = f.read()
            
            self.logger.info(f"開始處理影像: {image_path}")
            
            # 處理影像
            result = await self.orchestrator.process_frame(camera_id, frame_data)
            
            # 輸出結果
            self.logger.info(f"處理結果:")
            self.logger.info(f"  狀態: {result.status}")
            self.logger.info(f"  最終讀值: {result.final_reading}")
            self.logger.info(f"  信心度: {result.confidence:.3f}")
            self.logger.info(f"  處理時間: {result.processing_time:.2f}s")
            
            if result.error_message:
                self.logger.warning(f"  錯誤訊息: {result.error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"處理影像失敗: {e}")
            return None
        finally:
            await self.shutdown()


def create_parser():
    """建立命令列參數解析器"""
    parser = argparse.ArgumentParser(
        description="MeterGPT 儀器讀值系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 啟動系統服務
  python main.py run --config config/meter_gpt_config.yaml
  
  # 處理單一影像
  python main.py process --image test_images/meter.jpg --camera cam_001
  
  # 驗證配置檔案
  python main.py validate --config config/meter_gpt_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # run 命令
    run_parser = subparsers.add_parser('run', help='啟動系統服務')
    run_parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置檔案路徑'
    )
    
    # process 命令
    process_parser = subparsers.add_parser('process', help='處理單一影像')
    process_parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='影像檔案路徑'
    )
    process_parser.add_argument(
        '--camera',
        type=str,
        default='test_camera',
        help='攝影機 ID (預設: test_camera)'
    )
    process_parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置檔案路徑'
    )
    
    # validate 命令
    validate_parser = subparsers.add_parser('validate', help='驗證配置檔案')
    validate_parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='配置檔案路徑'
    )
    
    # version 命令
    subparsers.add_parser('version', help='顯示版本資訊')
    
    return parser


async def main():
    """主函數"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'version':
            print("MeterGPT v1.0.0")
            print("智慧儀器讀值系統")
            return 0
        
        elif args.command == 'validate':
            # 驗證配置檔案
            try:
                config = load_config(args.config)
                is_valid, errors = config.validate_config() if hasattr(config, 'validate_config') else (True, [])
                
                if is_valid:
                    print("✅ 配置檔案驗證通過")
                    return 0
                else:
                    print("❌ 配置檔案驗證失敗:")
                    for error in errors:
                        print(f"  - {error}")
                    return 1
                    
            except Exception as e:
                print(f"❌ 配置檔案載入失敗: {e}")
                return 1
        
        elif args.command == 'run':
            # 啟動系統服務
            app = MeterGPTApplication(args.config)
            success = await app.run()
            return 0 if success else 1
        
        elif args.command == 'process':
            # 處理單一影像
            if not os.path.exists(args.image):
                print(f"❌ 影像檔案不存在: {args.image}")
                return 1
            
            app = MeterGPTApplication(args.config)
            result = await app.process_single_image(args.image, args.camera)
            return 0 if result and result.status.value == 'success' else 1
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
        return 0
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        return 1


if __name__ == "__main__":
    # 設置事件循環策略 (Windows 相容性)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 執行主程式
    exit_code = asyncio.run(main())
    sys.exit(exit_code)