from pathlib import Path
import sys 
path = Path().joinpath().joinpath('..')
sys.path.append(str(path))

import configparser
import folium
import json 
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import random
from joblib import dump
from commons.utils import get_colors, add_clusters_to_markers, enrich_points_with_cluster_info, generate_sample_points, add_sample_points_to_markers, plot_points, get_drift_params
import geopy.distance
from typing import Union
import logging 

logger = logging.getLogger(__name__)

class ClusteringEstimator:

    # Construtor
    def __init__(self, env = 'PROD'):
        
        self.env = env
        self.app_config = self.get_app_config(env)

        logger.info('constructor executed')
    
    # Carrega das configurações da aplicação
    def get_app_config (self, env):

        app_config = configparser.ConfigParser()
        app_config.read('config/app_config.ini')
        return app_config[env]
    
    # Função para treinar o modelo
    def fit(self, points, input_points):

        #---------------------------------------------------------------
        # Treinando o modelo
        #---------------------------------------------------------------

        # Define o modelo base
        model = KMeans()

        # Define o número de clusters
        n_clusters = 5

        # Treina o modelo de clustering
        model = KMeans(n_clusters = n_clusters, random_state = 0).fit(input_points)

        #---------------------------------------------------------------
        # Guarda um mapa de exemplo do modelo
        #---------------------------------------------------------------

        # Seleciona algumas cores para o mapa
        colors = get_colors(n_clusters)

        # Agora vamos preparar um colormap, no qual vamos associar uma cor para cada cluster
        cm = dict()
        for cluster in np.unique(model.labels_):
            cm[cluster] = colors[cluster]

        # Preparando os marcadores do mapa
        markers = add_clusters_to_markers(model)

        # Preparando os pontos do mapa
        points = enrich_points_with_cluster_info(points, model, markers, cm)

        sample_points = generate_sample_points(points, markers)

        # Prepara um novo marcador para incluir os novos pontos
        markers_test = add_sample_points_to_markers (markers, sample_points)

        # Exibe todos os dados de treino no mapa
        m = plot_points(points = points, markers = markers_test)

        #---------------------------------------------------------------
        # Model drift params
        #---------------------------------------------------------------

        distances = dict()
        for p in points:

            # Seleciona a identificação do cluster do ponto p
            cluster = p['cluster']
            
            # Adiciona o resultado à lista
            try:
                distances[cluster] += [p['dist']]
            except:
                distances[cluster] = [p['dist']]

        drift_params = get_drift_params(distances)

        # Calcula a média de perc_inner_radius
        model_metric = [] 
        for k, v in drift_params.items():
            model_metric += [v['perc_inner_radius']] 
        model_metric = np.mean(model_metric)

        # Retorna artefato do modelo
        return model, m, drift_params, sample_points, model_metric