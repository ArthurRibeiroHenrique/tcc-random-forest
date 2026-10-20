# ==============================================================================
# ETAPA 01 - CONVERTE OS CSVs DO SICOR EM PARQUET
# ==============================================================================
# Para cada ano de 2013 a 2025, procura a subpasta "CSV <ano>" dentro da pasta
# do ano, lê o primeiro CSV encontrado e salva como .parquet.
#
# Entrada: PASTA_CSV_SICOR/<ano>/CSV <ano>/*.csv
# Saída:   PASTA_PARQUET/<nome do csv>.parquet
#
# Uso:
#   python preparacao_sicor/01_csv_para_parquet.py

import gc
import os

import pandas as pd

from comum import PASTA_CSV_SICOR, PASTA_PARQUET

# ==============================================================================
# CONFIGURAÇÕES DE LEITURA DO CSV
# ==============================================================================
ENCODING = "latin-1"
SEPARADOR = ";"
DECIMAL = ","
ANOS = range(2013, 2026)


def encontrar_pasta_csv(pasta_ano: str, ano: int) -> str | None:
    """Acha a subpasta do ano: "CSV 2013", "CSV_2013", "csv2013" etc."""
    subpastas = [d for d in os.listdir(pasta_ano) if os.path.isdir(os.path.join(pasta_ano, d))]

    for sub in subpastas:
        if "csv" in sub.lower() and str(ano) in sub:
            return sub

    # Se não encontrou com "CSV", aceita qualquer subpasta que tenha o ano
    for sub in subpastas:
        if str(ano) in sub:
            return sub
    return None


def converter_ano(ano: int) -> None:
    print(f"\n{'=' * 50}")
    print(f"📂 Processando ano: {ano}")

    pasta_ano = os.path.join(PASTA_CSV_SICOR, str(ano))
    if not os.path.exists(pasta_ano):
        print(f"⚠️  Pasta {pasta_ano} não encontrada. Pulando...")
        return

    pasta_csv = encontrar_pasta_csv(pasta_ano, ano)
    if pasta_csv is None:
        print(f"❌ Nenhuma subpasta com 'CSV' ou com o ano {ano} encontrada em {pasta_ano}. Pulando...")
        return
    print(f"   📁 Subpasta encontrada: {pasta_csv}")

    lista_csvs = sorted(
        os.path.join(pasta_ano, pasta_csv, f)
        for f in os.listdir(os.path.join(pasta_ano, pasta_csv))
        if f.lower().endswith(".csv")
    )
    if not lista_csvs:
        print(f"❌ Nenhum arquivo CSV encontrado em {pasta_csv}. Pulando...")
        return

    arquivo_csv = lista_csvs[0]
    print(f"   📄 Arquivo encontrado: {os.path.basename(arquivo_csv)}")

    try:
        print("   ⏳ Lendo CSV... (pode levar alguns segundos)")
        df = pd.read_csv(
            arquivo_csv,
            encoding=ENCODING,
            sep=SEPARADOR,
            decimal=DECIMAL,
            on_bad_lines="warn",
            low_memory=False,
        )
        print(f"   ✅ Leitura concluída! Linhas: {len(df)}, Colunas: {len(df.columns)}")
        print(f"   📋 Primeiras colunas: {df.columns[:5].tolist()}")
    except Exception as e:
        print(f"   ❌ ERRO ao ler o CSV: {e}")
        return

    nome_base = os.path.splitext(os.path.basename(arquivo_csv))[0]
    arquivo_parquet = os.path.join(PASTA_PARQUET, f"{nome_base}.parquet")

    try:
        print("   ⏳ Convertendo e salvando como Parquet...")
        df.to_parquet(arquivo_parquet, compression="snappy", index=False)
        tamanho_mb = os.path.getsize(arquivo_parquet) / (1024 * 1024)
        print(f"   ✅ Salvo com sucesso! Tamanho: {tamanho_mb:.2f} MB")
    except Exception as e:
        print(f"   ❌ ERRO ao salvar Parquet: {e}")
    finally:
        del df
        gc.collect()


if __name__ == "__main__":
    os.makedirs(PASTA_PARQUET, exist_ok=True)
    for ano in ANOS:
        converter_ano(ano)

    print(f"\n{'=' * 50}")
    print("🎉 PROCESSAMENTO CONCLUÍDO! Todos os anos foram convertidos.")
