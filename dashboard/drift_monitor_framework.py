import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from evidently import Report
from evidently.presets import DataDriftPreset
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Adiciona path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.drift_controller import DriftMonitorController

# Configuração
st.set_page_config(page_title="Drift Monitor", layout="wide")

# ============================================================================
# INICIALIZAÇÃO DO CONTROLLER
# ============================================================================
@st.cache_resource
def get_controller():
    """Inicializa controller (cached)"""
    return DriftMonitorController()

@st.cache_resource
def get_mlflow_client():
    """Inicializa cliente MLflow"""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()

try:
    controller = get_controller()
    global_settings = controller.get_global_settings()
except Exception as e:
    st.error(f"❌ Erro ao inicializar controller: {e}")
    st.info("Verifique se o arquivo dashboard/config/baselines.yml existe e está correto")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================
st.title("Monitor de Drift - Framework Genérico")
st.markdown("**Monitoramento extensível para múltiplos modelos ML**")

# ============================================================================
# SIDEBAR - SELEÇÃO DE EXPERIMENTO
# ============================================================================
st.sidebar.header("Configuração")

# Lista experimentos disponíveis
available_experiments = controller.get_experiments()
if not available_experiments:
    st.error("Nenhum experimento configurado em baselines.yml")
    st.stop()

experiment_name = st.sidebar.selectbox(
    "Selecione o Experimento",
    available_experiments,
    help="Experimentos configurados em baselines.yml"
)

# Configuração do experimento
exp_config = controller.get_experiment_config(experiment_name)
if exp_config:
    st.sidebar.info(f"""
    **{exp_config['name']}**
    
    {exp_config['description']}
    
    Tipo: `{exp_config['model_type']}`
    """)

# Parâmetros de busca
hours_back = st.sidebar.slider(
    "Janela temporal (horas)", 
    1, 168, 24
)

max_traces = st.sidebar.slider(
    "Máximo de traces", 
    50, 1000, 500
)

refresh = st.sidebar.button("Atualizar")

# ============================================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================================

@st.cache_data(ttl=global_settings.get('cache_ttl_seconds', 300))
def load_traces_data(experiment_name: str, max_results: int, hours_back: int):
    """Carrega traces do MLflow"""
    client = get_mlflow_client()
    ctrl = get_controller()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        st.error(f"Experimento '{experiment_name}' não encontrado no MLflow")
        return []
    
    # Obtém o nome do trace para filtrar da configuração
    trace_name_filter = ctrl.get_trace_name(experiment_name)
    
    traces = client.search_traces(
        locations=[experiment.experiment_id],
        max_results=max_results,
        order_by=["timestamp_ms DESC"]
    )
    
    st.info(f"Buscando traces com nome: **{trace_name_filter or 'Todos'}**")
    
    parsed_traces = []
    cutoff = datetime.now() - timedelta(hours=hours_back)
    
    for trace in traces:
        trace_name = trace.info.tags.get('mlflow.traceName', '')
        
        # Filtra por nome de trace se especificado
        if trace_name_filter and trace_name != trace_name_filter:
            continue
            
        trace_data = {
            'timestamp': datetime.fromtimestamp(trace.info.timestamp_ms / 1000),
            'execution_time_ms': trace.info.execution_time_ms
        }
        
        # Filtra por tempo
        if trace_data['timestamp'] < cutoff:
            continue
        
        for span in trace.data.spans:
            print(span.name)
            if span.name == 'predict':
                if span.inputs:
                    trace_data['lat'] = span.inputs.get('lat')
                    trace_data['lng'] = span.inputs.get('lng')
                if span.outputs:
                    result = span.outputs
                    trace_data['is_covered'] = result.get('is_region_covered')
                    
                    # Suporta diferentes formatos
                    closest_center = result.get('closest_center')
                    if isinstance(closest_center, dict):
                        trace_data['cluster_id'] = closest_center.get('id')
                        trace_data['distance_km'] = closest_center.get('distance_in_km')
                    else:
                        trace_data['cluster_id'] = result.get('cluster_id')
                        trace_data['distance_km'] = result.get('distance_km')
        
        if 'lat' in trace_data and 'is_covered' in trace_data:
            parsed_traces.append(trace_data)
    
    return parsed_traces

# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================
with st.spinner(f"📡 Carregando traces do experimento '{experiment_name}'..."):
    traces = load_traces_data(experiment_name, max_traces, hours_back)

# Debug: mostra informações sobre traces
with st.expander("🔍 Debug Info", expanded=not traces):
    client = get_mlflow_client()
    experiment = client.get_experiment_by_name(experiment_name)
    trace_name_filter = controller.get_trace_name(experiment_name)
    
    if experiment:
        st.success(f"✅ Experimento encontrado: **{experiment.name}**")
        st.write(f"- **ID**: {experiment.experiment_id}")
        st.write(f"- **Filtro de trace**: `{trace_name_filter or 'Nenhum (todos os traces)'}`")
        
        # Busca traces sem cache
        all_traces = client.search_traces(
            locations=[experiment.experiment_id],
            max_results=20,
            order_by=["timestamp_ms DESC"]
        )
        st.write(f"- **Total de traces no MLflow**: {len(all_traces)}")
        
        if all_traces:
            st.write("**Nomes de traces encontrados:**")
            trace_names = {}
            for t in all_traces:
                name = t.info.tags.get('mlflow.traceName', 'N/A')
                trace_names[name] = trace_names.get(name, 0) + 1
            
            for name, count in trace_names.items():
                icon = "✅" if name == trace_name_filter else "⚪"
                st.write(f"{icon} `{name}` ({count})")
            
            # Mostra estrutura do primeiro trace
            st.write("**Estrutura do primeiro trace:**")
            first = all_traces[0]
            st.write(f"- Nome: `{first.info.tags.get('mlflow.traceName', 'N/A')}`")
            st.write(f"- Timestamp: {datetime.fromtimestamp(first.info.timestamp_ms / 1000)}")
            st.write(f"- Spans: {len(first.data.spans)}")
            
            for span in first.data.spans[:2]:  # Primeiros 2 spans
                st.write(f"  **Span: {span.name}**")
                if span.inputs:
                    st.write("  Inputs:")
                    st.json(dict(list(span.inputs.items())[:5]))
                if span.outputs:
                    st.write("  Outputs:")
                    st.json(dict(list(span.outputs.items())[:5]))
    else:
        st.error(f"❌ Experimento '{experiment_name}' não encontrado")
        st.write("**Experimentos disponíveis:**")
        all_exps = client.search_experiments()
        for exp in all_exps:
            st.write(f"- {exp.name}")

if not traces:
    st.warning(f"⚠️ Nenhum trace encontrado para o experimento '{experiment_name}'")
    st.info("Execute algumas predições na API primeiro")
    st.stop()

# ============================================================================
# CÁLCULO DE MÉTRICAS
# ============================================================================
st.divider()
st.header("Análise de Métricas")

# Calcula métricas atuais usando handler
try:
    current_metrics = controller.calculate_current_metrics(experiment_name, traces)
except Exception as e:
    st.error(f"❌ Erro ao calcular métricas: {e}")
    st.stop()

# Compara com baseline
comparisons = controller.compare_metrics(experiment_name, current_metrics)

# ============================================================================
# SEÇÃO 1: MÉTRICAS PRINCIPAIS
# ============================================================================
st.subheader("1️⃣ Métricas Principais")

# Exibe KPIs principais (primeiras 4 métricas)
cols = st.columns(min(4, len(current_metrics)))
for idx, metric in enumerate(current_metrics[:4]):
    with cols[idx]:
        # Formata valor baseado no tipo
        if metric['metric_type'] == 'percentage':
            value_str = f"{metric['metric_value']:.1%}"
        elif metric['metric_type'] == 'continuous':
            value_str = f"{metric['metric_value']:.2f}"
        else:  # count
            value_str = f"{int(metric['metric_value'])}"
        
        st.metric(
            metric['metric_description'],
            value_str
        )

st.divider()

# ============================================================================
# SEÇÃO 2: COMPARAÇÃO COM BASELINE
# ============================================================================
st.subheader("2️⃣ Detecção de Performance Drift")
st.caption("Monitora se o desempenho do modelo degradou em produção")

if not comparisons:
    st.warning("Nenhuma métrica baseline configurada para este experimento")
else:
    # Formata dados para tabela
    comparison_table = []
    for comp in comparisons:
        # Formata valores baseado no tipo
        if comp['metric_type'] == 'percentage':
            baseline_str = f"{comp['baseline_value']:.1%}"
            current_str = f"{comp['current_value']:.1%}"
        elif comp['metric_type'] == 'continuous':
            baseline_str = f"{comp['baseline_value']:.2f}"
            current_str = f"{comp['current_value']:.2f}"
        else:
            baseline_str = f"{int(comp['baseline_value'])}"
            current_str = f"{int(comp['current_value'])}"
        
        comparison_table.append({
            'Métrica': comp['metric_description'],
            'Baseline': baseline_str,
            'Produção': current_str,
            'Desvio': f"{comp['deviation_pct']:+.1f}%",
            'Status': comp['status']
        })
    
    st.dataframe(
        pd.DataFrame(comparison_table),
        use_container_width=True,
        hide_index=True
    )
    
    # Alerta se drift detectado
    drift_detected = any(c['drift_detected'] for c in comparisons)
    if drift_detected:
        st.error("""
        ⚠️ **DRIFT DETECTADO!**
        
        Uma ou mais métricas desviaram significativamente do baseline.
        """)
        
        # Lista métricas com drift
        drifted = [c for c in comparisons if c['drift_detected']]
        st.markdown("**Métricas com drift:**")
        for d in drifted:
            st.markdown(f"- **{d['metric_description']}**: {d['deviation_pct']:+.1f}% (limite: {d['threshold']*100:.0f}%)")
    else:
        st.success("✅ **Sem drift detectado.** Modelo operando dentro do esperado.")

st.divider()

# ============================================================================
# SEÇÃO 3: EVIDENTLY REPORT
# ============================================================================
if global_settings.get('evidently_enabled', True):
    st.subheader("3️⃣ Análise de Data Drift")
    
    st.info("💡 Detecta mudanças na distribuição dos dados de entrada do modelo (testes PSI, KS e Chi²)")
    
    generate_report = st.button("🔬 Gerar Relatório", type="primary", use_container_width=False)
    
    min_samples = global_settings.get('min_samples_for_analysis', 40)
    
    if generate_report:
        if len(traces) < min_samples:
            st.warning(f"⚠️ São necessários pelo menos {min_samples} traces para análise")
        else:
            with st.spinner("Gerando relatório Evidently..."):
                try:
                    # Converte traces para DataFrame
                    df_prod = pd.DataFrame(traces)
                    
                    # Prepara dados usando handler
                    df_prepared = controller.prepare_for_evidently(experiment_name, df_prod)
                    
                    # Divide em referência e atual
                    split_idx = len(df_prepared) // 2
                    df_reference = df_prepared.iloc[:split_idx]
                    df_current = df_prepared.iloc[split_idx:]
                    
                    # Gera relatório
                    report = Report(metrics=[DataDriftPreset()])
                    result = report.run(reference_data=df_reference, current_data=df_current)

                    # Gerar o HTML
                    result.save_html('temp/teste.html')

                    with open("temp/teste.html", "r") as f:
                        report_html = f.read()

                    # Exibe
                    st.components.v1.html(report_html, height=1200, scrolling=True)
                    
                except Exception as e:
                    st.error(f"❌ Erro ao gerar relatório: {e}")

st.divider()

# ============================================================================
# FOOTER
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Informações")
    st.write(f"- **Experimento**: {experiment_name}")
    st.write(f"- **Total de traces**: {len(traces)}")
    st.write(f"- **Período**: Últimas {hours_back}h")

with col2:
    st.subheader("Configuração")
    st.write(f"- **Handler**: {exp_config.get('handler', 'N/A')}")
    st.write(f"- **Tipo**: {exp_config.get('model_type', 'N/A')}")
    st.write(f"- **Cache TTL**: {global_settings.get('cache_ttl_seconds', 300)}s")

mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
st.caption(f"🔗 [MLflow UI]({mlflow_uri}) | Framework extensível de drift monitoring")
