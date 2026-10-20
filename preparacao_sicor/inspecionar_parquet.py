# ==============================================================================
# EXPLORAÇÃO - ESTRUTURA E AMOSTRA DE UM ARQUIVO PARQUET
# ==============================================================================
# Mostra as colunas e os tipos de um .parquet, as primeiras linhas e,
# opcionalmente, salva as N primeiras linhas (todas as colunas) em CSV para
# abrir no Excel. Só os blocos necessários para essas linhas são lidos do disco.
#
# Uso:
#   python preparacao_sicor/inspecionar_parquet.py ARQUIVO.parquet
#   python preparacao_sicor/inspecionar_parquet.py PASTA --linhas 100 --csv amostra.csv
#   (se receber uma pasta, usa o primeiro .parquet dela em ordem alfabética)

import argparse
import os

import polars as pl

from comum import listar_parquets


def resolver_arquivo(caminho: str) -> str:
    if os.path.isdir(caminho):
        arquivos = listar_parquets(caminho)
        if not arquivos:
            raise SystemExit(f"⚠️ Nenhum arquivo parquet encontrado em: {caminho}")
        return arquivos[0]
    if not os.path.exists(caminho):
        raise SystemExit(f"❌ Arquivo não encontrado: {caminho}")
    return caminho


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mostra a estrutura e uma amostra de um Parquet.")
    parser.add_argument("caminho", help="arquivo .parquet ou pasta com arquivos .parquet")
    parser.add_argument("--linhas", type=int, default=5, help="quantas linhas mostrar/salvar")
    parser.add_argument("--csv", help="se informado, salva a amostra neste CSV")
    args = parser.parse_args()

    arquivo = resolver_arquivo(args.caminho)
    print(f"📂 Arquivo selecionado: {os.path.basename(arquivo)}")

    esquema = pl.scan_parquet(arquivo).collect_schema()
    print(f"\n📊 Total de colunas: {len(esquema)}\n")
    print("--- ESTRUTURA DE COLUNAS E TIPOS DE DADOS ---")
    for idx, (nome_coluna, tipo_dado) in enumerate(esquema.items(), start=1):
        print(f"{idx:02d}. {nome_coluna} -> {tipo_dado}")

    amostra = pl.read_parquet(arquivo, n_rows=args.linhas)
    print(f"\n--- PRÉVIA DOS DADOS (primeiras {amostra.height} linhas) ---")
    with pl.Config(tbl_cols=-1):
        print(amostra)

    if args.csv:
        amostra.write_csv(args.csv)
        print(f"\n💾 Amostra de {amostra.height} linhas x {amostra.width} colunas salva em: {args.csv}")
