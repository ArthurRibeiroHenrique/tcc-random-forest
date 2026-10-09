# ==============================================================================
# MAPA MUNICÍPIO -> ESTAÇÃO INMET MAIS PRÓXIMA (TCC, seção 2.5.1)
# ==============================================================================
# Nem todo município tem estação meteorológica. Este script calcula a distância
# (Haversine) entre a sede de cada município do Brasil e cada estação do INMET
# que possui dados no arquivo de clima processado, e gera uma tabela dizendo
# qual estação usar para cada município, com a distância e a flag de qualidade.
#
# Saída: dados/mapa_municipio_estacao.csv (separador ';', decimal ',')
#   - ORDEM = 1 é a estação mais próxima; ORDEM 2 e 3 são as reservas, usadas
#     pelo cruzar_sicor_inmet.py quando a mais próxima não tem dado na data.

import sqlite3
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ==============================================================================
# CONFIGURAÇÕES DE CAMINHOS
# ==============================================================================
PASTA_DADOS = Path(__file__).resolve().parent / "dados"
CAMINHO_SAIDA = PASTA_DADOS / "mapa_municipio_estacao.csv"

# 🟢 Catálogo do INMET com latitude/longitude das estações automáticas
CAMINHO_CATALOGO_ESTACOES = (
    r"C:\Users\User\Desktop\Arquivos - INMET\CSVs que precisam ser utilzados"
    r"\Todas as Estações Automáticas - Inmet.xlsx"
)

# 🟢 Arquivo de clima processado (só a coluna COD_ESTACAO é lida aqui)
CAMINHO_CLIMA = (
    r"C:\Users\User\Downloads"
    r"\Cópia de clima_processado_BRASIL_2019_2025_desvio_climatologico.csv"
)

# 🟢 Localidades 2022 do IBGE (Brasil inteiro). É a mesma base dos arquivos .kml
# por município; a sede municipal traz LAT_LOCALIDADE e LONG_LOCALIDADE.
URL_LOCALIDADES_IBGE = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/"
    "localidades/Localidades_do_Brasil/2022/Localidades_Brasil_gpkg.zip"
)
PASTA_CACHE_IBGE = PASTA_DADOS / "_cache_ibge"

# ==============================================================================
# REGRAS DO TCC (seção 2.5.1)
# ==============================================================================
DIST_MAX_KM = 100       # acima disso o município fica SEM_COBERTURA
MAX_ESTACOES = 3        # estação mais próxima + reservas dentro do raio
RAIO_TERRA_KM = 6371.0088


def classificar_distancia(dist_km):
    """Flags de qualidade: <30 excelente | 30-60 boa | 60-100 regular."""
    return np.select(
        [dist_km < 30, dist_km <= 60, dist_km <= DIST_MAX_KM],
        ["EXCELENTE", "BOA", "REGULAR"],
        default="SEM_COBERTURA",
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em km entre pontos (graus decimais). Aceita arrays do NumPy."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


# ==============================================================================
# 1. MUNICÍPIOS (IBGE)
# ==============================================================================
def carregar_municipios() -> pd.DataFrame:
    """Uma linha por município, com a coordenada da sede municipal."""
    gpkgs = list(PASTA_CACHE_IBGE.glob("*.gpkg"))
    if not gpkgs:
        print("⏳ Baixando Localidades 2022 do IBGE...")
        PASTA_CACHE_IBGE.mkdir(parents=True, exist_ok=True)
        caminho_zip = PASTA_CACHE_IBGE / "Localidades_Brasil_gpkg.zip"
        urllib.request.urlretrieve(URL_LOCALIDADES_IBGE, caminho_zip)
        with zipfile.ZipFile(caminho_zip) as z:
            z.extractall(PASTA_CACHE_IBGE)
        gpkgs = list(PASTA_CACHE_IBGE.glob("*.gpkg"))

    # O GeoPackage é um banco SQLite: dá para ler os atributos sem geopandas
    with sqlite3.connect(gpkgs[0]) as con:
        camada = con.execute("SELECT table_name FROM gpkg_contents").fetchone()[0]
        df = pd.read_sql(
            f"""
            SELECT CD_MUN, NM_MUN, SIGLA_UF, LAT_LOCALIDADE, LONG_LOCALIDADE
            FROM "{camada}"
            WHERE CT_LOCALIDADE = 'Cidade'
               OR CT_LOCALIDADE LIKE 'Distrito Estadual%'
            """,
            con,
        )

    # As capitais aparecem duas vezes (Sede Municipal e Capital Estadual)
    df = df.drop_duplicates(subset="CD_MUN").rename(columns={
        "CD_MUN": "CD_IBGE_MUNICIPIO",
        "NM_MUN": "NM_MUNICIPIO",
        "SIGLA_UF": "UF",
        "LAT_LOCALIDADE": "LAT_MUNICIPIO",
        "LONG_LOCALIDADE": "LON_MUNICIPIO",
    })
    df["CD_IBGE_MUNICIPIO"] = df["CD_IBGE_MUNICIPIO"].astype(int)
    df[["LAT_MUNICIPIO", "LON_MUNICIPIO"]] = df[["LAT_MUNICIPIO", "LON_MUNICIPIO"]].round(4)
    return df.sort_values("CD_IBGE_MUNICIPIO").reset_index(drop=True)


# ==============================================================================
# 2. ESTAÇÕES (INMET)
# ==============================================================================
def carregar_estacoes() -> pd.DataFrame:
    """Estações que têm dados no arquivo de clima, com suas coordenadas."""
    catalogo = pd.read_excel(CAMINHO_CATALOGO_ESTACOES).rename(columns={
        "Código": "COD_ESTACAO",
        "Nome": "NOME_ESTACAO",
        "UF": "UF_ESTACAO",
        "Latitude": "LAT_ESTACAO",
        "Longitude": "LON_ESTACAO",
    })
    catalogo["COD_ESTACAO"] = catalogo["COD_ESTACAO"].astype(str).str.strip()
    catalogo = catalogo.drop_duplicates(subset="COD_ESTACAO")

    # Só interessa estação que realmente tem dado climático processado
    codigos_clima = (
        pl.scan_csv(CAMINHO_CLIMA, separator=";", infer_schema_length=0)
        .select("COD_ESTACAO")
        .unique()
        .collect()["COD_ESTACAO"]
        .to_list()
    )

    estacoes = catalogo[catalogo["COD_ESTACAO"].isin(codigos_clima)]
    estacoes = estacoes.dropna(subset=["LAT_ESTACAO", "LON_ESTACAO"])
    sem_coordenada = sorted(set(codigos_clima) - set(estacoes["COD_ESTACAO"]))
    if sem_coordenada:
        print(f"⚠️ {len(sem_coordenada)} estações do clima sem coordenada no catálogo: {sem_coordenada}")

    colunas = ["COD_ESTACAO", "NOME_ESTACAO", "UF_ESTACAO", "LAT_ESTACAO", "LON_ESTACAO"]
    return estacoes[colunas].reset_index(drop=True)


# ==============================================================================
# 3. DISTÂNCIAS E MONTAGEM DO MAPA
# ==============================================================================
def montar_mapa(municipios: pd.DataFrame, estacoes: pd.DataFrame) -> pd.DataFrame:
    # Matriz (municípios x estações) de distâncias em km
    dist = haversine_km(
        municipios["LAT_MUNICIPIO"].to_numpy()[:, None],
        municipios["LON_MUNICIPIO"].to_numpy()[:, None],
        estacoes["LAT_ESTACAO"].to_numpy()[None, :],
        estacoes["LON_ESTACAO"].to_numpy()[None, :],
    )
    mais_proximas = np.argsort(dist, axis=1)[:, :MAX_ESTACOES]

    partes = []
    for ordem in range(mais_proximas.shape[1]):
        idx_estacao = mais_proximas[:, ordem]
        parte = municipios.copy()
        parte["ORDEM"] = ordem + 1
        for coluna in ["COD_ESTACAO", "NOME_ESTACAO", "UF_ESTACAO"]:
            parte[coluna] = estacoes[coluna].to_numpy()[idx_estacao]
        parte["DIST_KM"] = dist[np.arange(len(municipios)), idx_estacao].round(1)
        partes.append(parte)

    mapa = pd.concat(partes, ignore_index=True)
    mapa["QUALIDADE_ESTACAO"] = classificar_distancia(mapa["DIST_KM"].to_numpy())

    # A mais próxima fica sempre (documenta a distância mesmo sem cobertura);
    # as reservas só entram se estiverem dentro do raio máximo.
    mapa = mapa[(mapa["ORDEM"] == 1) | (mapa["DIST_KM"] <= DIST_MAX_KM)]
    return mapa.sort_values(["CD_IBGE_MUNICIPIO", "ORDEM"]).reset_index(drop=True)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    municipios = carregar_municipios()
    print(f"🏙️ Municípios (IBGE): {len(municipios)}")

    estacoes = carregar_estacoes()
    print(f"📡 Estações INMET com dados e coordenadas: {len(estacoes)}")

    mapa = montar_mapa(municipios, estacoes)
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    mapa.to_csv(CAMINHO_SAIDA, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    print(f"💾 Mapa salvo em: {CAMINHO_SAIDA} ({len(mapa)} linhas)")

    principal = mapa[mapa["ORDEM"] == 1]
    print("\n" + "=" * 60)
    print("📋 COBERTURA DOS MUNICÍPIOS (estação mais próxima)")
    print("=" * 60)
    for flag in ["EXCELENTE", "BOA", "REGULAR", "SEM_COBERTURA"]:
        qtd = int((principal["QUALIDADE_ESTACAO"] == flag).sum())
        print(f"   {flag:<14} {qtd:>5} municípios ({qtd / len(principal):.1%})")
    print(f"\n   Distância mediana: {principal['DIST_KM'].median():.1f} km")
