# ==============================================================================
# ETAPA 02 - ADICIONA O CÓDIGO IBGE DO MUNICÍPIO (PLANILHA COMPLEMENTAR)
# ==============================================================================
# A operação básica do SICOR não traz o município. Ele vem da planilha
# complementar (SICOR_COMPLEMENTO_OPERACAO_BASICA), ligada pelas chaves
# #REF_BACEN + NU_ORDEM. O LEFT JOIN mantém todas as operações, mesmo as que
# não aparecem no complemento (essas ficam com CD_IBGE_MUNICIPIO nulo).
#
# Entrada: PASTA_PARQUET/*.parquet + CAMINHO_COMPLEMENTO
# Saída:   PASTA_COM_CODIGO_MUNICIPIO/enriquecido_<nome>.parquet
#
# Uso:
#   python preparacao_sicor/02_adicionar_codigo_municipio.py

import os

import polars as pl

from comum import CAMINHO_COMPLEMENTO, PASTA_COM_CODIGO_MUNICIPIO, PASTA_PARQUET, listar_parquets

CHAVES_CRUZAMENTO = ["#REF_BACEN", "NU_ORDEM"]


if __name__ == "__main__":
    os.makedirs(PASTA_COM_CODIGO_MUNICIPIO, exist_ok=True)

    print("Preparando a tabela auxiliar complementar...")
    complemento = pl.scan_parquet(CAMINHO_COMPLEMENTO)

    arquivos = listar_parquets(PASTA_PARQUET)
    print(f"📂 Encontrados {len(arquivos)} arquivos para enriquecer.\n")

    for idx, caminho_arquivo in enumerate(arquivos, start=1):
        nome_arquivo = os.path.basename(caminho_arquivo)
        caminho_saida = os.path.join(PASTA_COM_CODIGO_MUNICIPIO, f"enriquecido_{nome_arquivo}")
        print(f"[{idx}/{len(arquivos)}] Cruzando chaves e enriquecendo: {nome_arquivo}...")

        # O Polars faz o JOIN via streaming direto para o disco
        (
            pl.scan_parquet(caminho_arquivo)
            .join(complemento, on=CHAVES_CRUZAMENTO, how="left")
            .sink_parquet(caminho_saida, compression="snappy")
        )
        print(f"   ✅ Salvo com sucesso: {caminho_saida}")

    print("\n🎉 Todas as planilhas da pasta foram enriquecidas com sucesso!")
