# ==============================================================================
# ETAPA 03 - ADICIONA O NOME DO MUNICÍPIO (MunicipiosSicor.csv)
# ==============================================================================
# Liga o CD_IBGE_MUNICIPIO (vindo da etapa 02) à tabela auxiliar de municípios
# do SICOR, que traz o nome do município e da UF.
#
# Entrada: PASTA_COM_CODIGO_MUNICIPIO/*.parquet + CAMINHO_MUNICIPIOS_SICOR
# Saída:   PASTA_COM_NOME_MUNICIPIO/final_<nome>.parquet
#
# Uso:
#   python preparacao_sicor/03_adicionar_nome_municipio.py

import os

import polars as pl

from comum import (
    CAMINHO_MUNICIPIOS_SICOR,
    PASTA_COM_CODIGO_MUNICIPIO,
    PASTA_COM_NOME_MUNICIPIO,
    listar_parquets,
)

COLUNA_IBGE_SICOR = "CD_IBGE_MUNICIPIO"     # coluna vinda da planilha complementar
COLUNA_IBGE_AUXILIAR = "COD_MUNICIPIO_IBGE"  # coluna da tabela de municípios


if __name__ == "__main__":
    os.makedirs(PASTA_COM_NOME_MUNICIPIO, exist_ok=True)

    print("Preparando a tabela auxiliar de municípios do IBGE...")
    municipios = (
        pl.read_csv(
            CAMINHO_MUNICIPIOS_SICOR,
            separator=";",
            encoding="latin-1",
            infer_schema_length=10000,
        )
        .lazy()
        .with_columns(pl.col(COLUNA_IBGE_AUXILIAR).cast(pl.String))
    )

    arquivos = listar_parquets(PASTA_COM_CODIGO_MUNICIPIO)
    print(f"📂 Encontrados {len(arquivos)} arquivos para anexar as cidades.\n")

    for idx, caminho_arquivo in enumerate(arquivos, start=1):
        nome_arquivo = os.path.basename(caminho_arquivo)
        caminho_saida = os.path.join(PASTA_COM_NOME_MUNICIPIO, f"final_{nome_arquivo}")
        print(f"[{idx}/{len(arquivos)}] Cruzando municípios e gerando: {nome_arquivo}...")

        (
            pl.scan_parquet(caminho_arquivo)
            .with_columns(pl.col(COLUNA_IBGE_SICOR).cast(pl.String))
            .join(municipios, left_on=COLUNA_IBGE_SICOR, right_on=COLUNA_IBGE_AUXILIAR, how="left")
            .sink_parquet(caminho_saida, compression="snappy")
        )
        print(f"   ✅ Salvo com sucesso: {caminho_saida}")

    print("\n🎉 Todas as planilhas foram enriquecidas com os nomes das cidades!")
