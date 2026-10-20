# ==============================================================================
# ETAPA 06 - REMOVE AS COLUNAS QUE NÃO SERVEM PARA A IA
# ==============================================================================
# Monta a lista de exclusão a partir da auditoria da etapa 05 (tudo que foi
# marcado como REMOVER ou AVALIAR REMOÇÃO) e grava novos Parquets só com as
# colunas restantes. As colunas excluídas nem são lidas do disco, o que
# economiza RAM e tempo.
#
# Entrada: PASTA_ENRIQUECIDO/*.parquet + CAMINHO_AUDITORIA
# Saída:   PASTA_LIMPO/<nome>.parquet  (entrada do cruzar_sicor_inmet.py)
#
# Uso:
#   python preparacao_sicor/06_limpar_colunas.py

import os

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from comum import CAMINHO_AUDITORIA, PASTA_ENRIQUECIDO, PASTA_LIMPO, listar_parquets

# Sem as colunas inúteis, dá para usar blocos maiores sem estourar a RAM
TAMANHO_BLOCO = 50_000


def montar_blacklist_auditoria(caminho_csv) -> set:
    """Colunas marcadas para remoção (❌) ou avaliação crítica (⚠️) na auditoria."""
    print("📖 Lendo relatório de auditoria para montar a Blacklist...")
    df_audit = pl.read_csv(caminho_csv)

    blacklist = set(
        df_audit.filter(
            pl.col("Status").str.contains("REMOVER") | pl.col("Status").str.contains("AVALIAR REMOÇÃO")
        )["Coluna"].unique().to_list()
    )
    print(f"🚫 Blacklist montada com sucesso: {len(blacklist)} colunas serão descartadas.")

    for exemplo in ["VL_PRODUTIV_OBTIDA", "CNPJ_IF", "NOME_MUNICIPIO", "VL_AREA_INFORMADA"]:
        status = "✅ NA BLACKLIST" if exemplo in blacklist else "⚠️ ATENÇÃO: NÃO ENCONTRADA"
        print(f"   -> {exemplo}: {status}")

    return blacklist


def limpar_arquivo_parquet(caminho_origem: str, caminho_saida: str, blacklist: set) -> tuple[int, int]:
    """Copia o Parquet em blocos, lendo só as colunas fora da blacklist."""
    pf = pq.ParquetFile(caminho_origem)
    colunas_originais = pf.schema_arrow.names
    colunas_para_ler = [c for c in colunas_originais if c not in blacklist]

    if not colunas_para_ler:
        print(f"⚠️ {os.path.basename(caminho_origem)}: Nenhuma coluna útil encontrada. Pulando...")
        return 0, 0

    schema_saida = pa.schema([pf.schema_arrow.field(c) for c in colunas_para_ler])

    with pq.ParquetWriter(caminho_saida, schema_saida, compression="snappy") as writer:
        for batch in pf.iter_batches(batch_size=TAMANHO_BLOCO, columns=colunas_para_ler):
            writer.write_batch(batch)

    return len(colunas_originais), len(colunas_para_ler)


if __name__ == "__main__":
    if not os.path.exists(CAMINHO_AUDITORIA):
        raise FileNotFoundError(f"❌ CSV de auditoria não encontrado em: {CAMINHO_AUDITORIA}")

    blacklist = montar_blacklist_auditoria(CAMINHO_AUDITORIA)

    arquivos = listar_parquets(PASTA_ENRIQUECIDO)
    if not arquivos:
        raise SystemExit(f"⚠️ Nenhum arquivo parquet encontrado em: {PASTA_ENRIQUECIDO}")

    os.makedirs(PASTA_LIMPO, exist_ok=True)
    print(f"\n🚀 Iniciando limpeza de {len(arquivos)} arquivos (Recorte 2019-2025)...")
    print(f"📂 Salvando arquivos limpos em: {PASTA_LIMPO}\n")
    print("-" * 60)

    for idx, arq in enumerate(arquivos, 1):
        nome_base = os.path.basename(arq)
        try:
            cols_orig, cols_final = limpar_arquivo_parquet(arq, os.path.join(PASTA_LIMPO, nome_base), blacklist)
            print(f"[{idx}/{len(arquivos)}] ✅ {nome_base}")
            print(f"   📉 Colunas: {cols_orig} -> {cols_final} (Descartadas: {cols_orig - cols_final})")
        except Exception as e:
            print(f"[{idx}/{len(arquivos)}] ❌ Erro ao processar {nome_base}: {e}")

    print("-" * 60)
    print("🎉 LIMPEZA FINALIZADA!")
    print(f"📦 Dados prontos para o cruzamento com o clima em: {PASTA_LIMPO}")
