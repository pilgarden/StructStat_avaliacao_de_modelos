"""
Módulo de Configuração (config.py)
Centraliza constantes de unidades estruturais (Força, Deslocamento, Momento, Tensão).
"""

# Dicionários segmentados por grandeza física para Engenharia de Estruturas.
# Todas as conversões são baseadas na unidade base do SI internamente.
GRANDEZAS_UNIDADES = {
    'Força': {
        'N (Newton)': 1.0, 
        'kN (Quilonewton)': 1e3, 
        'MN (Meganewton)': 1e6,
        'kgf (Quilograma-força)': 9.80665, 
        'tf (Tonelada-força)': 9806.65
    },
    'Deslocamento': {
        'm (Metro)': 1.0, 
        'cm (Centímetro)': 1e-2, 
        'mm (Milímetro)': 1e-3, 
        'in (Polegada)': 0.0254
    },
    'Momento Fletor / Torsor': {
        'N.m (Newton-metro)': 1.0, 
        'kN.m (Quilonewton-metro)': 1e3,
        'kN.cm (Quilonewton-centímetro)': 10.0, # 1 kN * 0.01 m
        'tf.m (Tonelada-força-metro)': 9806.65
    },
    'Tensão / Pressão': {
        'Pa (Pascal)': 1.0, 
        'kPa (Quilopascal)': 1e3, 
        'MPa (Megapascal)': 1e6, 
        'GPa (Gigapascal)': 1e9
    }
}

def calcular_fator_conversao(grandeza: str, unidade_origem: str, unidade_destino: str) -> float:
    """Calcula o multiplicador escalar entre duas unidades da mesma grandeza."""
    dicionario_grandeza = GRANDEZAS_UNIDADES.get(grandeza, {})
    
    # Busca o fator ou assume 1.0 em caso de erro
    fator_origem = dicionario_grandeza.get(unidade_origem, 1.0)
    fator_destino = dicionario_grandeza.get(unidade_destino, 1.0)
    
    return float(fator_origem / fator_destino)