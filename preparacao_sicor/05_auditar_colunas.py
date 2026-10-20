# ==============================================================================
# ETAPA 05 - AUDITORIA DE SAÚDE DAS COLUNAS (TCC, seção 2.5.3)
# ==============================================================================
# Para cada coluna de cada arquivo: % de nulos, quantidade de valores únicos e
# a decisão de manter ou remover, conforme os critérios do TCC:
#   - identificadores únicos (CPFs, BACEN...)       -> remover
#   - informação geográfica redundante               -> remover
#   - vazamento de dados (só se sabe depois da safra) -> remover
#   - coluna constante ou com cardinalidade de ID     -> remover
#   - mais de 50% de nulos                            -> avaliar remoção
#
# Processa em blocos (PyArrow + Polars) para não estourar a RAM.
#
# Entrada: PASTA_ENRIQUECIDO/*.parquet
# Saída:   PASTA_RELATORIOS/auditoria_colunas_sicor.csv (e .parquet)
#
# Uso:
#   python preparacao_sicor/05_auditar_colunas.py

import gc
import os

import polars as pl
import pyarrow.parquet as pq

from comum import CAMINHO_AUDITORIA, PASTA_ENRIQUECIDO, PASTA_RELATORIOS, listar_parquets

# Tamanho do bloco de leitura
TAMANHO_BLOCO = 25_000

# Acima disso a contagem de únicos para (evita estourar a memória) e a coluna é
# marcada como de alta cardinalidade
MAX_UNICOS_TRACK = 500_000

# ------------------------------------------------------------------------------
# COLUNAS PROIBIDAS (justificativa no TCC)
# ------------------------------------------------------------------------------
COLUNAS_IDENTIFICADORES = {
    "#REF_BACEN", "NU_ORDEM", "CNPJ_IF", "CD_INST_CREDITO",
    "CNPJ_AGENTE_INVEST", "CD_EMPREENDIMENTO", "CD_CONTRATO_STN",
    "CD_CNPJ_CADASTRANTE", "NUM_CEDULA_IF", "AGENCIA_IF", "REF_BACEN_EFETIVO",
}

COLUNAS_REDUNDANTES_GEO = {
    "NOME_MUNICIPIO", "COD_UF", "NOME_UF", "COD_UF_IBGE",
    "#COD_MUNICIPIO_CADMU", "VL_AREA_INFORMADA",
}

COLUNAS_VAZAMENTO_DADOS = {
    "VL_PRODUTIV_OBTIDA",  # ⚠️ DATA LEAKAGE: só se conhece no fim da safra
}

ORDEM_STATUS = [
    "❌ REMOVER (Data Leakage)",
    "❌ REMOVER (Identificador)",
    "❌ REMOVER (Redundante)",
    "❌ REMOVER (Constante)",
    "❌ REMOVER (Alta Cardinalidade/ID)",
    "⚠️ AVALIAR REMOÇÃO (Muitos Nulos)",
    "✅ MANTER (Feature Útil)",
]


def classificar_coluna(col: str, n_unicos: int, pct_nulos: float, total_linhas: int) -> tuple[str, str]:
    if col in COLUNAS_VAZAMENTO_DADOS:
        return ORDEM_STATUS[0], "Vazamento de dados (ex: produtividade só se conhece no fim da safra)."
    if col in COLUNAS_IDENTIFICADORES:
        return ORDEM_STATUS[1], "Chave primária/ID. Não generaliza para novos pedidos."
    if col in COLUNAS_REDUNDANTES_GEO:
        return ORDEM_STATUS[2], "Informação geográfica duplicada (já temos CD_IBGE_MUNICIPIO)."
    if n_unicos == 1:
        return ORDEM_STATUS[3], "A coluna possui apenas 1 valor único em todo o dataset."
    if n_unicos >= total_linhas * 0.95:
        return ORDEM_STATUS[4], "Quase todos os valores são únicos. Provável ID disfarçado."
    if pct_nulos > 50.0:
        return ORDEM_STATUS[5], f"Mais de 50% de dados faltantes ({pct_nulos:.1f}%). Introduz ruído."
    return ORDEM_STATUS[6], "Coluna com variância e preenchimento adequados."


def analisar_saude_arquivo(caminho_arquivo: str) -> pl.DataFrame:
    nome_base = os.path.basename(caminho_arquivo)
    print(f"\n🔍 Analisando: {nome_base}")

    pf = pq.ParquetFile(caminho_arquivo)
    schema = pf.schema_arrow
    colunas = schema.names

    total_linhas = 0
    null_counts = {col: 0 for col in colunas}

    # Colunas de ponto flutuante: só min/max (para detectar constante), sem
    # guardar os valores únicos. Demais colunas: conjunto de únicos até o limite.
    numeric_stats = {col: {"min": float("inf"), "max": float("-inf")} for col in colunas}
    unique_sets = {}
    unique_overflow = {}
    for col in colunas:
        tipo = str(schema.field(col).type).lower()
        unique_sets[col] = None if ("float" in tipo or "double" in tipo) else set()
        unique_overflow[col] = False

    for batch in pf.iter_batches(batch_size=TAMANHO_BLOCO):
        df = pl.from_arrow(batch)
        total_linhas += df.height

        for i, nulos in enumerate(df.null_count().row(0)):
            null_counts[colunas[i]] += nulos

        for col in colunas:
            serie = df[col]

            if unique_sets[col] is None and not unique_overflow[col]:
                minimo, maximo = serie.min(), serie.max()
                if minimo is not None and minimo < numeric_stats[col]["min"]:
                    numeric_stats[col]["min"] = minimo
                if maximo is not None and maximo > numeric_stats[col]["max"]:
                    numeric_stats[col]["max"] = maximo

            elif not unique_overflow[col]:
                unicos_bloco = serie.drop_nulls().unique().to_list()
                if len(unique_sets[col]) + len(unicos_bloco) > MAX_UNICOS_TRACK:
                    unique_overflow[col] = True
                    unique_sets[col] = None  # libera a RAM na hora
                else:
                    unique_sets[col].update(unicos_bloco)

        del df, batch
    gc.collect()

    print(f"📊 Gerando relatório para {total_linhas:,} linhas...")
    resultados = []
    for col in colunas:
        pct_nulos = (null_counts[col] / total_linhas) * 100 if total_linhas > 0 else 0

        if unique_overflow[col]:
            n_unicos = f"> {MAX_UNICOS_TRACK:,}"
            n_unicos_int = MAX_UNICOS_TRACK + 1
        elif unique_sets[col] is None:
            # Coluna numérica: min == max indica constante
            min_val, max_val = numeric_stats[col]["min"], numeric_stats[col]["max"]
            if min_val == max_val and min_val != float("inf"):
                n_unicos, n_unicos_int = "1 (Constante)", 1
            else:
                n_unicos, n_unicos_int = "Numérico Contínuo", MAX_UNICOS_TRACK + 1
        else:
            n_unicos_int = len(unique_sets[col])
            n_unicos = f"{n_unicos_int:,}"

        status, motivo = classificar_coluna(col, n_unicos_int, pct_nulos, total_linhas)
        resultados.append({
            "Arquivo": nome_base,
            "Coluna": col,
            "Status": status,
            "Motivo": motivo,
            "Valores_Unicos": n_unicos,
            "Pct_Nulos": round(pct_nulos, 2),
        })

    return pl.DataFrame(resultados)


if __name__ == "__main__":
    os.makedirs(PASTA_RELATORIOS, exist_ok=True)

    arquivos = listar_parquets(PASTA_ENRIQUECIDO)
    if not arquivos:
        raise SystemExit(f"⚠️ Nenhum arquivo parquet encontrado em: {PASTA_ENRIQUECIDO}")

    print(f"🚀 Iniciando auditoria de {len(arquivos)} arquivos...")
    relatorios = []
    for idx, arq in enumerate(arquivos, 1):
        try:
            relatorios.append(analisar_saude_arquivo(arq))
            print(f"[{idx}/{len(arquivos)}] ✅ Concluído.")
        except Exception as e:
            print(f"[{idx}/{len(arquivos)}] ❌ Erro ao processar {arq}: {e}")

    if not relatorios:
        raise SystemExit("❌ Nenhum arquivo foi auditado.")

    # Problemas primeiro, depois as colunas mantidas
    ordem = {s: i for i, s in enumerate(ORDEM_STATUS)}
    df_final = (
        pl.concat(relatorios)
        .with_columns(pl.col("Status").replace_strict(ordem, return_dtype=pl.Int32).alias("_ordem"))
        .sort("_ordem", maintain_order=True)
        .drop("_ordem")
    )

    df_final.write_csv(CAMINHO_AUDITORIA)
    caminho_parquet = CAMINHO_AUDITORIA.with_suffix(".parquet")
    df_final.write_parquet(caminho_parquet)

    print("\n" + "=" * 60)
    print("🎉 AUDITORIA FINALIZADA!")
    print(f"📄 Relatório CSV salvo em: {CAMINHO_AUDITORIA}")
    print(f"📦 Relatório Parquet salvo em: {caminho_parquet}")
    print("=" * 60)

    print("\n📊 RESUMO GERAL DAS COLUNAS:")
    print(df_final.group_by("Status").len().sort("len", descending=True))
