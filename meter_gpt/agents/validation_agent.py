"""
ValidationAgent 代理人
負責驗證 OCR 結果的合理性和準確性，確保讀值品質
"""

import re
import math
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import asyncio

from metagpt.agent import Agent
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message

from ..models.messages import (
    OCRResult, ValidationResult, ValidationRule, ProcessingStatus, FailureType
)
from ..core.config import get_config
from ..utils.logger import get_logger, log_agent_action


class ValidationRuleEngine:
    """驗證規則引擎"""
    
    def __init__(self):
        self.logger = get_logger("ValidationRuleEngine")
        self.built_in_rules = self._create_built_in_rules()
    
    def _create_built_in_rules(self) -> List[ValidationRule]:
        """建立內建驗證規則"""
        rules = [
            ValidationRule(
                rule_id="numeric_format",
                rule_name="數值格式驗證",
                rule_type="format",
                parameters={
                    "pattern": r"^-?\d+\.?\d*$",
                    "allow_empty": False
                }
            ),
            ValidationRule(
                rule_id="range_check",
                rule_name="數值範圍檢查",
                rule_type="range",
                parameters={
                    "min_value": -999999,
                    "max_value": 999999
                }
            ),
            ValidationRule(
                rule_id="length_check",
                rule_name="文字長度檢查",
                rule_type="length",
                parameters={
                    "min_length": 1,
                    "max_length": 50
                }
            ),
            ValidationRule(
                rule_id="consistency_check",
                rule_name="一致性檢查",
                rule_type="consistency",
                parameters={
                    "tolerance": 0.1,
                    "history_window": 5
                }
            ),
            ValidationRule(
                rule_id="confidence_threshold",
                rule_name="信心度閾值檢查",
                rule_type="confidence",
                parameters={
                    "min_confidence": 0.7
                }
            )
        ]
        return rules
    
    def apply_rule(self, rule: ValidationRule, ocr_result: OCRResult, 
                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        應用驗證規則
        
        Args:
            rule: 驗證規則
            ocr_result: OCR 結果
            context: 上下文資訊
            
        Returns:
            Dict[str, Any]: 驗證結果
        """
        try:
            if rule.rule_type == "format":
                return self._apply_format_rule(rule, ocr_result)
            elif rule.rule_type == "range":
                return self._apply_range_rule(rule, ocr_result)
            elif rule.rule_type == "length":
                return self._apply_length_rule(rule, ocr_result)
            elif rule.rule_type == "consistency":
                return self._apply_consistency_rule(rule, ocr_result, context)
            elif rule.rule_type == "confidence":
                return self._apply_confidence_rule(rule, ocr_result)
            else:
                return {
                    "passed": False,
                    "message": f"未知的規則類型: {rule.rule_type}"
                }
                
        except Exception as e:
            self.logger.error(f"應用規則失敗 {rule.rule_id}: {e}")
            return {
                "passed": False,
                "message": f"規則執行異常: {str(e)}"
            }
    
    def _apply_format_rule(self, rule: ValidationRule, ocr_result: OCRResult) -> Dict[str, Any]:
        """應用格式驗證規則"""
        text = ocr_result.recognized_text.strip()
        pattern = rule.parameters.get("pattern", ".*")
        allow_empty = rule.parameters.get("allow_empty", True)
        
        if not text and not allow_empty:
            return {
                "passed": False,
                "message": "識別結果為空"
            }
        
        if text and not re.match(pattern, text):
            return {
                "passed": False,
                "message": f"格式不符合要求: {text}"
            }
        
        return {
            "passed": True,
            "message": "格式驗證通過"
        }
    
    def _apply_range_rule(self, rule: ValidationRule, ocr_result: OCRResult) -> Dict[str, Any]:
        """應用範圍驗證規則"""
        text = ocr_result.recognized_text.strip()
        
        try:
            value = float(text)
            min_value = rule.parameters.get("min_value", float('-inf'))
            max_value = rule.parameters.get("max_value", float('inf'))
            
            if value < min_value or value > max_value:
                return {
                    "passed": False,
                    "message": f"數值超出範圍 [{min_value}, {max_value}]: {value}"
                }
            
            return {
                "passed": True,
                "message": "範圍驗證通過"
            }
            
        except ValueError:
            return {
                "passed": False,
                "message": f"無法轉換為數值: {text}"
            }
    
    def _apply_length_rule(self, rule: ValidationRule, ocr_result: OCRResult) -> Dict[str, Any]:
        """應用長度驗證規則"""
        text = ocr_result.recognized_text.strip()
        min_length = rule.parameters.get("min_length", 0)
        max_length = rule.parameters.get("max_length", float('inf'))
        
        if len(text) < min_length or len(text) > max_length:
            return {
                "passed": False,
                "message": f"文字長度超出範圍 [{min_length}, {max_length}]: {len(text)}"
            }
        
        return {
            "passed": True,
            "message": "長度驗證通過"
        }
    
    def _apply_consistency_rule(self, rule: ValidationRule, ocr_result: OCRResult, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """應用一致性驗證規則"""
        if not context or 'history' not in context:
            return {
                "passed": True,
                "message": "無歷史資料，跳過一致性檢查"
            }
        
        try:
            current_text = ocr_result.recognized_text.strip()
            current_value = float(current_text)
            
            history = context['history']
            window_size = rule.parameters.get("history_window", 5)
            tolerance = rule.parameters.get("tolerance", 0.1)
            
            # 取得最近的歷史資料
            recent_history = history[-window_size:] if len(history) > window_size else history
            
            if len(recent_history) < 2:
                return {
                    "passed": True,
                    "message": "歷史資料不足，跳過一致性檢查"
                }
            
            # 計算歷史平均值
            historical_values = []
            for hist_result in recent_history:
                try:
                    hist_value = float(hist_result.recognized_text.strip())
                    historical_values.append(hist_value)
                except ValueError:
                    continue
            
            if not historical_values:
                return {
                    "passed": True,
                    "message": "無有效歷史數值，跳過一致性檢查"
                }
            
            avg_value = sum(historical_values) / len(historical_values)
            
            # 檢查偏差
            if avg_value != 0:
                deviation = abs(current_value - avg_value) / abs(avg_value)
            else:
                deviation = abs(current_value)
            
            if deviation > tolerance:
                return {
                    "passed": False,
                    "message": f"數值與歷史平均值偏差過大: {deviation:.2%} > {tolerance:.2%}"
                }
            
            return {
                "passed": True,
                "message": "一致性驗證通過"
            }
            
        except ValueError:
            return {
                "passed": True,
                "message": "非數值資料，跳過一致性檢查"
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"一致性檢查異常: {str(e)}"
            }
    
    def _apply_confidence_rule(self, rule: ValidationRule, ocr_result: OCRResult) -> Dict[str, Any]:
        """應用信心度驗證規則"""
        min_confidence = rule.parameters.get("min_confidence", 0.7)
        
        if ocr_result.confidence < min_confidence:
            return {
                "passed": False,
                "message": f"信心度過低: {ocr_result.confidence:.3f} < {min_confidence:.3f}"
            }
        
        return {
            "passed": True,
            "message": "信心度驗證通過"
        }


class ValidationAction(Action):
    """驗證動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("ValidationAction")
        self.config = get_config()
        self.rule_engine = ValidationRuleEngine()
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> List[ValidationRule]:
        """載入驗證規則"""
        try:
            rules = self.rule_engine.built_in_rules.copy()
            
            # 從配置載入自定義規則
            if self.config and self.config.validation_rules:
                rules.extend(self.config.validation_rules)
            
            # 過濾啟用的規則
            active_rules = [rule for rule in rules if rule.is_active]
            
            self.logger.info(f"載入了 {len(active_rules)} 個驗證規則")
            return active_rules
            
        except Exception as e:
            self.logger.error(f"載入驗證規則失敗: {e}")
            return self.rule_engine.built_in_rules
    
    async def run(self, ocr_results: List[OCRResult], 
                 context: Dict[str, Any] = None) -> List[ValidationResult]:
        """
        執行驗證
        
        Args:
            ocr_results: OCR 結果列表
            context: 上下文資訊
            
        Returns:
            List[ValidationResult]: 驗證結果列表
        """
        try:
            validation_results = []
            
            for ocr_result in ocr_results:
                validation_result = await self._validate_single_result(ocr_result, context)
                validation_results.append(validation_result)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"驗證執行失敗: {e}")
            return []
    
    async def _validate_single_result(self, ocr_result: OCRResult, 
                                    context: Dict[str, Any] = None) -> ValidationResult:
        """驗證單個 OCR 結果"""
        try:
            applied_rules = []
            error_messages = []
            warnings = []
            passed_count = 0
            
            for rule in self.validation_rules:
                try:
                    rule_result = self.rule_engine.apply_rule(rule, ocr_result, context)
                    applied_rules.append(rule.rule_id)
                    
                    if rule_result["passed"]:
                        passed_count += 1
                    else:
                        if rule.rule_type in ["format", "range", "confidence"]:
                            error_messages.append(rule_result["message"])
                        else:
                            warnings.append(rule_result["message"])
                            
                except Exception as e:
                    self.logger.error(f"規則 {rule.rule_id} 執行失敗: {e}")
                    warnings.append(f"規則 {rule.rule_id} 執行異常")
            
            # 計算驗證分數
            validation_score = passed_count / len(self.validation_rules) if self.validation_rules else 0.0
            
            # 判斷是否通過驗證
            is_valid = len(error_messages) == 0 and validation_score >= 0.7
            
            validation_result = ValidationResult(
                frame_id=ocr_result.frame_id,
                is_valid=is_valid,
                applied_rules=applied_rules,
                validation_score=validation_score,
                error_messages=error_messages,
                warnings=warnings
            )
            
            self.logger.log_validation_result(
                ocr_result.frame_id,
                is_valid,
                validation_score
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"單個結果驗證失敗: {e}")
            return ValidationResult(
                frame_id=ocr_result.frame_id,
                is_valid=False,
                applied_rules=[],
                validation_score=0.0,
                error_messages=[f"驗證過程異常: {str(e)}"],
                warnings=[]
            )


class CrossValidationAction(Action):
    """交叉驗證動作"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("CrossValidationAction")
    
    async def run(self, ocr_results: List[OCRResult]) -> Dict[str, Any]:
        """
        執行交叉驗證
        
        Args:
            ocr_results: OCR 結果列表
            
        Returns:
            Dict[str, Any]: 交叉驗證結果
        """
        try:
            if len(ocr_results) < 2:
                return {
                    "cross_validation_possible": False,
                    "message": "需要至少兩個 OCR 結果進行交叉驗證"
                }
            
            # 按 ROI 分組
            roi_groups = {}
            for result in ocr_results:
                roi_id = result.roi_id
                if roi_id not in roi_groups:
                    roi_groups[roi_id] = []
                roi_groups[roi_id].append(result)
            
            cross_validation_results = {}
            
            for roi_id, results in roi_groups.items():
                if len(results) >= 2:
                    cross_validation_results[roi_id] = self._cross_validate_roi_results(results)
            
            return {
                "cross_validation_possible": True,
                "results": cross_validation_results,
                "summary": self._summarize_cross_validation(cross_validation_results)
            }
            
        except Exception as e:
            self.logger.error(f"交叉驗證失敗: {e}")
            return {"error": str(e)}
    
    def _cross_validate_roi_results(self, results: List[OCRResult]) -> Dict[str, Any]:
        """對同一 ROI 的多個結果進行交叉驗證"""
        try:
            texts = [r.recognized_text.strip() for r in results]
            confidences = [r.confidence for r in results]
            
            # 文字一致性檢查
            unique_texts = list(set(texts))
            text_consistency = len(unique_texts) == 1
            
            # 信心度分析
            avg_confidence = sum(confidences) / len(confidences)
            confidence_std = math.sqrt(sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences))
            
            # 數值一致性檢查（如果是數值）
            numeric_consistency = None
            try:
                values = [float(text) for text in texts if text]
                if values:
                    avg_value = sum(values) / len(values)
                    value_std = math.sqrt(sum((v - avg_value) ** 2 for v in values) / len(values))
                    numeric_consistency = {
                        "average": avg_value,
                        "std_deviation": value_std,
                        "coefficient_of_variation": value_std / avg_value if avg_value != 0 else float('inf')
                    }
            except ValueError:
                pass
            
            return {
                "text_consistency": text_consistency,
                "unique_texts": unique_texts,
                "confidence_stats": {
                    "average": avg_confidence,
                    "std_deviation": confidence_std
                },
                "numeric_consistency": numeric_consistency,
                "recommendation": self._generate_cross_validation_recommendation(
                    text_consistency, avg_confidence, numeric_consistency
                )
            }
            
        except Exception as e:
            self.logger.error(f"ROI 交叉驗證失敗: {e}")
            return {"error": str(e)}
    
    def _generate_cross_validation_recommendation(self, text_consistency: bool, 
                                                avg_confidence: float, 
                                                numeric_consistency: Optional[Dict]) -> str:
        """生成交叉驗證建議"""
        if text_consistency and avg_confidence > 0.8:
            return "結果一致且信心度高，建議採用"
        elif text_consistency and avg_confidence > 0.6:
            return "結果一致但信心度中等，建議謹慎採用"
        elif not text_consistency and avg_confidence > 0.8:
            return "結果不一致但信心度高，建議進一步檢查"
        else:
            return "結果不一致且信心度低，建議重新處理"
    
    def _summarize_cross_validation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """總結交叉驗證結果"""
        try:
            total_rois = len(results)
            consistent_rois = sum(1 for r in results.values() if r.get("text_consistency", False))
            
            return {
                "total_rois": total_rois,
                "consistent_rois": consistent_rois,
                "consistency_rate": consistent_rois / total_rois if total_rois > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"交叉驗證總結失敗: {e}")
            return {"error": str(e)}


class ValidationAgent(Role):
    """驗證代理人"""
    
    def __init__(self, name: str = "ValidationAgent", **kwargs):
        """
        初始化驗證代理人
        
        Args:
            name: 代理人名稱
        """
        super().__init__(name=name, **kwargs)
        
        # 設置動作
        self._init_actions([
            ValidationAction(),
            CrossValidationAction()
        ])
        
        self.logger = get_logger("ValidationAgent")
        self.config = get_config()
        
        # 驗證歷史記錄
        self.validation_history: Dict[str, List[ValidationResult]] = {}
        self.ocr_history: Dict[str, List[OCRResult]] = {}
        self.max_history_size = 100
    
    @log_agent_action("ValidationAgent")
    async def _act(self) -> Message:
        """執行代理人動作"""
        try:
            # 從訊息中取得 OCR 結果
            ocr_results = []
            frame_id = "unknown"
            
            for msg in self.rc.memory.get():
                if hasattr(msg, 'content') and 'ocr_results' in str(msg.content):
                    # 這裡應該解析訊息內容取得 OCR 結果
                    # 為了示範，我們跳過實際解析
                    pass
            
            if not ocr_results:
                return Message(
                    content="沒有收到 OCR 結果資料",
                    role=self.profile,
                    cause_by=ValidationAction
                )
            
            # 執行驗證
            validation_results = await self.validate_ocr_results(ocr_results, frame_id)
            
            # 建立回應訊息
            message_content = {
                'action': 'validation',
                'frame_id': frame_id,
                'validation_count': len(validation_results),
                'valid_count': len([r for r in validation_results if r.is_valid]),
                'timestamp': datetime.now().isoformat()
            }
            
            return Message(
                content=str(message_content),
                role=self.profile,
                cause_by=ValidationAction
            )
            
        except Exception as e:
            self.logger.error(f"代理人動作執行失敗: {e}")
            return Message(
                content=f"Error: {str(e)}",
                role=self.profile,
                cause_by=ValidationAction
            )
    
    async def validate_ocr_results(self, ocr_results: List[OCRResult], 
                                 frame_id: str) -> List[ValidationResult]:
        """
        驗證 OCR 結果
        
        Args:
            ocr_results: OCR 結果列表
            frame_id: 影像幀識別碼
            
        Returns:
            List[ValidationResult]: 驗證結果列表
        """
        try:
            # 準備上下文資訊
            context = self._prepare_validation_context(frame_id)
            
            # 執行驗證
            validation_action = ValidationAction()
            validation_results = await validation_action.run(ocr_results, context)
            
            # 執行交叉驗證
            cross_validation_action = CrossValidationAction()
            cross_validation_results = await cross_validation_action.run(ocr_results)
            
            # 更新歷史記錄
            self._update_history(ocr_results, validation_results, frame_id)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"OCR 結果驗證失敗: {e}")
            raise
    
    def _prepare_validation_context(self, frame_id: str) -> Dict[str, Any]:
        """準備驗證上下文"""
        try:
            camera_id = frame_id.split('_')[0] if '_' in frame_id else 'unknown'
            
            context = {
                'frame_id': frame_id,
                'camera_id': camera_id,
                'history': self.ocr_history.get(camera_id, [])
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"準備驗證上下文失敗: {e}")
            return {}
    
    def _update_history(self, ocr_results: List[OCRResult], 
                       validation_results: List[ValidationResult], frame_id: str):
        """更新歷史記錄"""
        try:
            camera_id = frame_id.split('_')[0] if '_' in frame_id else 'unknown'
            
            # 更新 OCR 歷史
            if camera_id not in self.ocr_history:
                self.ocr_history[camera_id] = []
            
            self.ocr_history[camera_id].extend(ocr_results)
            
            # 限制歷史記錄大小
            if len(self.ocr_history[camera_id]) > self.max_history_size:
                excess = len(self.ocr_history[camera_id]) - self.max_history_size
                self.ocr_history[camera_id] = self.ocr_history[camera_id][excess:]
            
            # 更新驗證歷史
            if camera_id not in self.validation_history:
                self.validation_history[camera_id] = []
            
            self.validation_history[camera_id].extend(validation_results)
            
            if len(self.validation_history[camera_id]) > self.max_history_size:
                excess = len(self.validation_history[camera_id]) - self.max_history_size
                self.validation_history[camera_id] = self.validation_history[camera_id][excess:]
                
        except Exception as e:
            self.logger.error(f"更新歷史記錄失敗: {e}")
    
    def get_validation_statistics(self, camera_id: str) -> Dict[str, Any]:
        """
        取得驗證統計資訊
        
        Args:
            camera_id: 攝影機 ID
            
        Returns:
            Dict[str, Any]: 統計資訊
        """
        try:
            history = self.validation_history.get(camera_id, [])
            if not history:
                return {'total_validations': 0, 'success_rate': 0.0, 'average_score': 0.0}
            
            valid_count = len([r for r in history if r.is_valid])
            total_score = sum(r.validation_score for r in history)
            
            return {
                'total_validations': len(history),
                'valid_results': valid_count,
                'success_rate': valid_count / len(history),
                'average_score': total_score / len(history),
                'latest_validation': history[-1].timestamp.isoformat() if history else None
            }
            
        except Exception as e:
            self.logger.error(f"取得驗證統計失敗: {e}")
            return {'error': str(e)}
    
    async def get_validation_summary(self) -> Dict[str, Any]:
        """取得驗證摘要"""
        try:
            summary = {
                'total_cameras': len(self.validation_history),
                'camera_statistics': {},
                'overall_success_rate': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            total_validations = 0
            total_valid = 0
            
            for camera_id in self.validation_history.keys():
                stats = self.get_validation_statistics(camera_id)
                summary['camera_statistics'][camera_id] = stats
                
                total_validations += stats['total_validations']
                total_valid += stats['valid_results']
            
            # 計算整體成功率
            if total_validations > 0:
                summary['overall_success_rate'] = total_valid / total_validations
            
            return summary
            
        except Exception as e:
            self.logger.error(f"取得驗證摘要失敗: {e}")
            return {'error': str(e)}


# 建立全域 ValidationAgent 實例的工廠函數
def create_validation_agent() -> ValidationAgent:
    """建立 ValidationAgent 實例"""
    return ValidationAgent()