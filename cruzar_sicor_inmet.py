# ==============================================================================
# CRUZAMENTO SICOR x INMET (via estação mais próxima do município)
# ==============================================================================
# Para cada operação do SICOR, busca as variáveis climáticas da estação INMET
# mais próxima do município (CD_IBGE_MUNICIPIO) na data de emissão do contrato.
# Se a estação mais próxima não tiver dado válido naquela data, usa a próxima
# estação dentro do raio de 100 km (colunas ORDEM 2 e 3 do mapa).
#
# Pré-requisito: rodar antes o mapa_municipio_estacao.py
#
# Uso:
#   python cruzar_sicor_inmet.py
#   python cruzar_sicor_inmet.py --sicor PASTA --clima ARQUIVO.csv --saida PASTA

import argparse
import glob
import os
from pathlib import Path

import polars as pl

# ==============================================================================
# CONFIGURAÇÕES DE CAMINHOS (podem ser trocados pela linha de comando)
# ==============================================================================
PASTA_PROJETO = Path(__file__).resolve().parent

# 🟢 Tabela gerada pelo mapa_municipio_estacao.py
CAMINHO_MAPA = PASTA_PROJETO / "dados" / "mapa_municipio_estacao.csv"

# 🟢 PASTA DE ENTRADA: Parquets limpos do SICOR
PASTA_SICOR = (
    "/content/drive/MyDrive/TCC - Ideias/"
    "Microdados do Crédito Rural e do Proagro - Sicor - Parquet/"
    "Parquet_Limpo_Features_TCC"
)

# 🟢 Arquivo de clima processado do INMET
CAMINHO_CLIMA = (
    r"C:\Users\User\Downloads"
    r"\Cópia de clima_processado_BRASIL_2019_2025_desvio_climatologico.csv"
)

# 🟢 PASTA DE SAÍDA: Parquets do SICOR já com as colunas climáticas
PASTA_SAIDA = PASTA_PROJETO / "dados" / "sicor_com_clima"

# ==============================================================================
# REGRAS DO CRUZAMENTO
# ==============================================================================
COLUNA_MUNICIPIO = "CD_IBGE_MUNICIPIO"
COLUNA_DATA = "DT_EMISSAO"
COLUNAS_CLIMA = ["PRECIP_90D", "DESVIO_%", "VERANICOS_90D"]

# Usa o clima do dia ANTERIOR à emissão: o modelo só pode enxergar o que já
# tinha acontecido antes da assinatura do contrato (evita data leakage).
DEFASAGEM_DIAS = 1

# As janelas de 90 dias só estão completas a partir do 90º dia de cada estação.
# Antes disso o PRECIP_90D é um acumulado parcial e não deve ser usado.
DIAS_MINIMOS_HISTORICO = 90


# ==============================================================================
# 1. CARGA DAS TABELAS DE APOIO
# ==============================================================================
def carregar_mapa(caminho) -> pl.DataFrame:
    mapa = pl.read_csv(caminho, separator=";", decimal_comma=True).select(
        pl.col("CD_IBGE_MUNICIPIO").cast(pl.Int64).alias("_MUN"),
        pl.col("ORDEM").alias("ORDEM_ESTACAO"),
        "COD_ESTACAO",
        pl.col("DIST_KM").alias("DIST_ESTACAO_KM"),
        "QUALIDADE_ESTACAO",
    )
    # Estações fora do raio não entram no cruzamento
    return mapa.filter(pl.col("QUALIDADE_ESTACAO") != "SEM_COBERTURA")


def carregar_clima(caminho) -> pl.DataFrame:
    """Lê só as colunas necessárias e mantém apenas os dias com dado utilizável."""
    tipos = {"COD_ESTACAO": pl.Utf8, "DATA": pl.Date}
    tipos.update({coluna: pl.Float64 for coluna in COLUNAS_CLIMA})

    clima = (
        pl.scan_csv(caminho, separator=";", decimal_comma=True, schema_overrides=tipos)
        .select("COD_ESTACAO", "DATA", *COLUNAS_CLIMA)
        .collect()
    )
    total = clima.height

    if clima.select("COD_ESTACAO", "DATA").is_duplicated().any():
        raise ValueError("❌ O arquivo de clima tem linhas repetidas para a mesma estação e data.")

    clima = (
        clima.with_columns(pl.col("DATA").min().over("COD_ESTACAO").alias("_INICIO"))
        .filter(
            (pl.col("DATA") >= pl.col("_INICIO") + pl.duration(days=DIAS_MINIMOS_HISTORICO - 1))
            & pl.col("PRECIP_90D").is_not_null()
        )
        .drop("_INICIO")
    )
    print(f"🌦️ Clima: {clima.height:,} dias-estação utilizáveis "
          f"({total - clima.height:,} descartados por janela de 90 dias incompleta)")
    return clima


# ==============================================================================
# 2. CRUZAMENTO DE UM ARQUIVO DO SICOR
# ==============================================================================
def cruzar_arquivo(caminho_entrada: str, caminho_saida: str,
                   mapa: pl.DataFrame, clima: pl.DataFrame) -> pl.DataFrame:
    lf = pl.scan_parquet(caminho_entrada)
    schema = lf.collect_schema()

    # No SICOR a data vem como texto "dd/mm/aaaa"; aceita também Date/Datetime
    tipo_data = schema[COLUNA_DATA]
    if tipo_data == pl.Date:
        data_emissao = pl.col(COLUNA_DATA)
    elif isinstance(tipo_data, pl.Datetime):
        data_emissao = pl.col(COLUNA_DATA).dt.date()
    else:
        data_emissao = pl.col(COLUNA_DATA).str.to_date("%d/%m/%Y", strict=False)

    if schema[COLUNA_MUNICIPIO].is_numeric():
        municipio = pl.col(COLUNA_MUNICIPIO).cast(pl.Int64, strict=False)
    else:
        municipio = pl.col(COLUNA_MUNICIPIO).str.strip_chars().cast(pl.Int64, strict=False)

    lf = lf.with_columns(
        municipio.alias("_MUN"),
        (data_emissao - pl.duration(days=DEFASAGEM_DIAS)).alias("DATA_CLIMA"),
    )

    # Resolve a estação uma única vez por (município, data), e não por operação:
    # são poucos pares distintos, então o cruzamento fica leve mesmo com milhões
    # de contratos.
    pares = lf.select("_MUN", "DATA_CLIMA").unique().collect()
    melhor_estacao = (
        pares.join(mapa, on="_MUN")
        .join(clima, left_on=["COD_ESTACAO", "DATA_CLIMA"], right_on=["COD_ESTACAO", "DATA"])
        .sort("ORDEM_ESTACAO")
        .unique(subset=["_MUN", "DATA_CLIMA"], keep="first", maintain_order=True)
    )
    municipios_cobertos = mapa.select("_MUN").unique().with_columns(pl.lit(True).alias("_COBERTO"))

    (
        lf.join(melhor_estacao.lazy(), on=["_MUN", "DATA_CLIMA"], how="left")
        .join(municipios_cobertos.lazy(), on="_MUN", how="left")
        .with_columns(
            pl.when(pl.col("QUALIDADE_ESTACAO").is_not_null())
            .then(pl.col("QUALIDADE_ESTACAO"))
            .when(pl.col("_COBERTO"))
            .then(pl.lit("SEM_DADO_NA_DATA"))
            .otherwise(pl.lit("SEM_COBERTURA"))
            .alias("QUALIDADE_ESTACAO")
        )
        .drop("_MUN", "_COBERTO")
        .sink_parquet(caminho_saida, compression="snappy")
    )

    return (
        pl.scan_parquet(caminho_saida)
        .group_by("QUALIDADE_ESTACAO")
        .agg(pl.len().alias("OPERACOES"))
        .collect()
    )


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cruza os Parquets do SICOR com o clima do INMET.")
    parser.add_argument("--sicor", default=str(PASTA_SICOR), help="pasta com os .parquet do SICOR")
    parser.add_argument("--clima", default=str(CAMINHO_CLIMA), help="CSV de clima processado do INMET")
    parser.add_argument("--mapa", default=str(CAMINHO_MAPA), help="CSV gerado pelo mapa_municipio_estacao.py")
    parser.add_argument("--saida", default=str(PASTA_SAIDA), help="pasta onde salvar os .parquet cruzados")
    args = parser.parse_args()

    arquivos = sorted(glob.glob(os.path.join(args.sicor, "*.parquet")))
    if not arquivos:
        raise FileNotFoundError(f"❌ Nenhum arquivo parquet encontrado em: {args.sicor}")

    mapa = carregar_mapa(args.mapa)
    print(f"🗺️ Mapa: {mapa['_MUN'].n_unique():,} municípios com estação a até 100 km")
    clima = carregar_clima(args.clima)

    os.makedirs(args.saida, exist_ok=True)
    resumos = []
    for idx, arq in enumerate(arquivos, 1):
        nome_base = os.path.basename(arq)
        resumo = cruzar_arquivo(arq, os.path.join(args.saida, nome_base), mapa, clima)
        resumos.append(resumo)
        print(f"[{idx}/{len(arquivos)}] ✅ {nome_base} ({resumo['OPERACOES'].sum():,} operações)")

    total = pl.concat(resumos).group_by("QUALIDADE_ESTACAO").agg(pl.col("OPERACOES").sum())
    total_operacoes = total["OPERACOES"].sum()
    print("\n" + "=" * 60)
    print("📋 QUALIDADE DO CRUZAMENTO (operações do SICOR)")
    print("=" * 60)
    for flag in ["EXCELENTE", "BOA", "REGULAR", "SEM_DADO_NA_DATA", "SEM_COBERTURA"]:
        qtd = total.filter(pl.col("QUALIDADE_ESTACAO") == flag)["OPERACOES"].sum()
        print(f"   {flag:<17} {qtd:>12,} ({qtd / total_operacoes:.1%})")
    print(f"\n📦 Arquivos cruzados salvos em: {args.saida}")
