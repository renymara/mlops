"""
Handler para modelo de Clustering de Regiões de Entrega
========================================================
Implementa extração de métricas específicas deste modelo
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from handlers.base_handler import ModelMetricsHandler


class DeliveryRegionHandler(ModelMetricsHandler):
    """
    Handler para o modelo de clustering K-means de regiões de entrega
    """
    
    def get_model_metrics(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extrai métricas do modelo de delivery region
        
        Métricas calculadas:
        - Taxa de cobertura (% de requisições dentro da área)
        - Distância média (distância média aos centros)
        - Latência média (tempo de resposta)
        - Total de requisições
        """
        if not traces:
            return []
        
        df = pd.DataFrame(traces)
        
        metrics = []
        
        # 1. Taxa de cobertura
        if 'is_covered' in df.columns:
            coverage = df['is_covered'].sum() / len(df)
            metrics.append({
                'metric_name': 'coverage_rate',
                'metric_description': 'Taxa de cobertura das entregas',
                'metric_value': coverage,
                'metric_type': 'percentage'
            })
        
        # 2. Distância média
        if 'distance_km' in df.columns:
            avg_distance = df['distance_km'].mean()
            metrics.append({
                'metric_name': 'avg_distance_km',
                'metric_description': 'Distância média aos centros',
                'metric_value': avg_distance,
                'metric_type': 'continuous'
            })
        
        # 3. Latência média
        if 'execution_time_ms' in df.columns:
            avg_latency = df['execution_time_ms'].mean()
            metrics.append({
                'metric_name': 'avg_latency_ms',
                'metric_description': 'Latência média de predição',
                'metric_value': avg_latency,
                'metric_type': 'continuous'
            })
        
        # 4. Total de requisições
        metrics.append({
            'metric_name': 'total_requests',
            'metric_description': 'Total de requisições',
            'metric_value': len(df),
            'metric_type': 'count'
        })
        
        return metrics
    
    def prepare_for_evidently(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara dados para análise com Evidently
        
        Seleciona apenas features relevantes e numéricas/categóricas
        """
        cols_to_keep = []
        
        # Features numéricas
        for col in ['lat', 'lng', 'distance_km', 'execution_time_ms']:
            if col in df.columns:
                cols_to_keep.append(col)
        
        # Features categóricas
        for col in ['cluster_id', 'is_covered']:
            if col in df.columns:
                cols_to_keep.append(col)
        
        return df[cols_to_keep].copy()
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Retorna descrições das features para documentação
        """
        return {
            'lat': 'Latitude do ponto de entrega',
            'lng': 'Longitude do ponto de entrega',
            'distance_km': 'Distância ao centro de distribuição mais próximo (km)',
            'cluster_id': 'ID do cluster/centro de distribuição atribuído',
            'is_covered': 'Se o ponto está dentro da área de cobertura',
            'execution_time_ms': 'Tempo de execução da predição (ms)'
        }
