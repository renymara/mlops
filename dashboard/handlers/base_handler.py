"""
Interface padrão para handlers de métricas de modelos
=====================================================
Todos os handlers devem implementar esta interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd


class ModelMetricsHandler(ABC):
    """
    Interface base para handlers de métricas de modelos
    """
    
    @abstractmethod
    def get_model_metrics(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extrai métricas do modelo a partir dos traces do MLflow
        
        Args:
            traces: Lista de traces do MLflow (já parseados)
            
        Returns:
            Lista de dicionários com formato:
            [
                {
                    'metric_name': str,          # Nome único da métrica
                    'metric_description': str,   # Descrição legível
                    'metric_value': float,       # Valor calculado
                    'metric_type': str          # 'percentage', 'continuous', 'count'
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    def prepare_for_evidently(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara DataFrame para análise com Evidently
        
        Args:
            df: DataFrame bruto dos traces
            
        Returns:
            DataFrame formatado com features apropriadas
        """
        pass
    
    def calculate_drift(self, baseline_value: float, current_value: float, 
                       threshold: float) -> Dict[str, Any]:
        """
        Calcula drift entre baseline e valor atual
        
        Args:
            baseline_value: Valor de baseline
            current_value: Valor atual
            threshold: Limite para alerta
            
        Returns:
            Dict com informações de drift
        """
        if baseline_value == 0:
            deviation_pct = 0
        else:
            deviation_pct = ((current_value - baseline_value) / baseline_value)
        
        drift_detected = abs(deviation_pct) > threshold
        
        return {
            'baseline': baseline_value,
            'current': current_value,
            'deviation': deviation_pct,
            'deviation_pct': deviation_pct * 100,
            'threshold': threshold,
            'drift_detected': drift_detected,
            'status': '🔴 DRIFT' if drift_detected else '🟢 OK'
        }
