"""
Controller para Dashboard de Monitoramento de Drift
===================================================
Gerencia configurações, carregamento de dados e handlers de modelos
"""
import yaml
import importlib
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

# Configura path para importar handlers
sys.path.insert(0, str(Path(__file__).parent.parent))

# Caminho fixo do arquivo de configuração (relativo a este arquivo)
CONFIG_PATH = Path(__file__).parent.parent / "config/baselines.yml"


class DriftMonitorController:
    """
    Controller central para o dashboard de drift monitoring
    """
    
    def __init__(self):
        """Inicializa o controller"""
        self.config_path = CONFIG_PATH
        self.config = self._load_config()
        self.handlers = {}
        self._load_handlers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Erro ao parsear YAML: {e}")
    
    def _load_handlers(self):
        """Carrega handlers dinâmicos baseado na configuração"""
        experiments = self.config.get('experiments', {})
        
        for exp_name, exp_config in experiments.items():
            handler_name = exp_config.get('handler')
            if handler_name:
                try:
                    # Importa módulo dinamicamente
                    module = importlib.import_module(f'handlers.{handler_name}')
                    
                    # Encontra a classe Handler (assume padrão CamelCase)
                    handler_class_name = ''.join(word.capitalize() for word in handler_name.split('_'))
                    handler_class = getattr(module, handler_class_name)
                    
                    # Instancia handler
                    self.handlers[exp_name] = handler_class()
                    
                except (ImportError, AttributeError) as e:
                    print(f"⚠️ Aviso: Não foi possível carregar handler '{handler_name}' para '{exp_name}': {e}")
    
    def get_experiments(self) -> List[str]:
        """Retorna lista de experimentos configurados"""
        return list(self.config.get('experiments', {}).keys())
    
    def get_experiment_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna configuração de um experimento específico
        
        Args:
            experiment_name: Nome do experimento
            
        Returns:
            Dicionário com configuração ou None se não encontrado
        """
        return self.config.get('experiments', {}).get(experiment_name)
    
    def get_baseline_metrics(self, experiment_name: str) -> List[Dict[str, Any]]:
        """
        Retorna métricas baseline de um experimento
        
        Args:
            experiment_name: Nome do experimento
            
        Returns:
            Lista de métricas baseline
        """
        exp_config = self.get_experiment_config(experiment_name)
        if not exp_config:
            return []
        
        return exp_config.get('baseline_metrics', [])
    
    def get_trace_name(self, experiment_name: str) -> Optional[str]:
        """
        Retorna o nome do trace para filtrar no MLflow
        
        Args:
            experiment_name: Nome do experimento
            
        Returns:
            Nome do trace ou None
        """
        exp_config = self.get_experiment_config(experiment_name)
        if not exp_config:
            return None
        
        return exp_config.get('trace_name')
    
    def get_handler(self, experiment_name: str):
        """
        Retorna handler de métricas para um experimento
        
        Args:
            experiment_name: Nome do experimento
            
        Returns:
            Instância do handler ou None
        """
        return self.handlers.get(experiment_name)
    
    def calculate_current_metrics(self, experiment_name: str, 
                                  traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calcula métricas atuais usando o handler apropriado
        
        Args:
            experiment_name: Nome do experimento
            traces: Lista de traces do MLflow
            
        Returns:
            Lista de métricas calculadas
        """
        handler = self.get_handler(experiment_name)
        if not handler:
            raise ValueError(f"Handler não encontrado para experimento '{experiment_name}'")
        
        return handler.get_model_metrics(traces)
    
    def compare_metrics(self, experiment_name: str, 
                       current_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compara métricas atuais com baseline e detecta drift
        
        Args:
            experiment_name: Nome do experimento
            current_metrics: Métricas calculadas de produção
            
        Returns:
            Lista de comparações com drift detection
        """
        baseline_metrics = self.get_baseline_metrics(experiment_name)
        handler = self.get_handler(experiment_name)
        
        # Cria lookup de métricas atuais
        current_lookup = {m['metric_name']: m for m in current_metrics}
        
        comparisons = []
        
        for baseline in baseline_metrics:
            metric_name = baseline['metric_name']
            current = current_lookup.get(metric_name)
            
            if not current:
                continue
            
            # Calcula drift
            drift_info = handler.calculate_drift(
                baseline_value=baseline['metric_value'],
                current_value=current['metric_value'],
                threshold=baseline.get('alert_threshold', 
                                      self.config['global_settings']['default_alert_threshold'])
            )
            
            comparisons.append({
                'metric_name': metric_name,
                'metric_description': baseline['metric_description'],
                'metric_type': baseline['metric_type'],
                'baseline_value': drift_info['baseline'],
                'current_value': drift_info['current'],
                'deviation': drift_info['deviation'],
                'deviation_pct': drift_info['deviation_pct'],
                'threshold': drift_info['threshold'],
                'drift_detected': drift_info['drift_detected'],
                'status': drift_info['status']
            })
        
        return comparisons
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Retorna configurações globais"""
        return self.config.get('global_settings', {})
    
    def prepare_for_evidently(self, experiment_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara dados para análise com Evidently usando handler apropriado
        
        Args:
            experiment_name: Nome do experimento
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame preparado
        """
        handler = self.get_handler(experiment_name)
        if not handler:
            return df
        
        return handler.prepare_for_evidently(df)
