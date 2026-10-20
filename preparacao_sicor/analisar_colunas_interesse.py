# ==============================================================================
# EXPLORAÇÃO - PRESENÇA E NULOS DAS COLUNAS DE INTERESSE
# ==============================================================================
# Verifica, arquivo por arquivo, se as colunas candidatas a feature existem e
# quantos nulos têm. Classificação: até 5% OK | até 20% Atenção | acima Crítico.
#
# Uso:
#   python preparacao_sicor/analisar_colunas_interesse.py
#   python preparacao_sicor/analisar_colunas_interesse.py --pasta CAMINHO

import argparse
import os

import polars as pl
import pyarrow.parquet as pq

from comum import PASTA_PARQUET, listar_parquets

COLUNAS_INTERESSE = [
    # Mutuários
    "CD_TIPO_BENEFICIARIO", "CD_SEXO", "CD_DAP",
    # Operações
    "VL_AREA_FINANC", "VL_PARC_CREDITO", "VL_REC_PROPRIO", "VL_PREV_PROD",
    "CD_PROGRAMA", "CD_SUBPROGRAMA", "CD_IBGE_MUNICIPIO",
    "DT_INICIO_PLANTIO", "DT_FIM_PLANTIO",
    "DT_INICIO_COLHEITA", "DT_FIM_COLHEITA",
    "CD_ZONEAMENTO", "VL_JUROS",
    "VL_PREC_CUSTO_EFET_TOTAL", "VL_RECEITA_BRUTA_ESPERADA",
]


def classificar_nulos(pct: float) -> str:
    if pct <= 5:
        return "🟢 OK"
    if pct <= 20:
        return "🟡 Atenção"
    return "🔴 Crítico"


def analisar_arquivo(caminho: str) -> dict:
    total_linhas = pq.ParquetFile(caminho).metadata.num_rows
    lf = pl.scan_parquet(caminho)
    colunas_existentes = lf.collect_schema().names()
    presentes = [c for c in COLUNAS_INTERESSE if c in colunas_existentes]

    nulos = {}
    if presentes:
        contagem = lf.select(pl.col(c).is_null().sum() for c in presentes).collect()
        nulos = {c: contagem[c].item() for c in presentes}

    return {
        "arquivo": os.path.basename(caminho),
        "total_linhas": total_linhas,
        "colunas_presentes": colunas_existentes,
        "colunas_interesse_presentes": presentes,
        "nulos": nulos,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Presença e nulos das colunas de interesse do SICOR.")
    parser.add_argument("--pasta", default=str(PASTA_PARQUET), help="pasta com os .parquet do SICOR")
    args = parser.parse_args()

    arquivos = listar_parquets(args.pasta)
    if not arquivos:
        raise SystemExit("❌ Nenhum arquivo .parquet encontrado!")

    print(f"🔍 Encontrei {len(arquivos)} arquivo(s):")
    for a in arquivos:
        print(f"   - {os.path.basename(a)}")

    print("\n⏳ Analisando os arquivos...\n")
    resultados = []
    for arq in arquivos:
        print(f"   📂 {os.path.basename(arq)}...")
        resultados.append(analisar_arquivo(arq))

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=300):
        print("\n" + "=" * 90)
        print("📋 MATRIZ DE PRESENÇA DAS COLUNAS DE INTERESSE")
        print("=" * 90)
        presenca = pl.DataFrame([
            {"Arquivo": r["arquivo"],
             **{c: "✅" if c in r["colunas_interesse_presentes"] else "❌" for c in COLUNAS_INTERESSE}}
            for r in resultados
        ])
        print(presenca)

        print("\n" + "=" * 90)
        print("📊 ANÁLISE DE NULOS PARA CADA COLUNA DE INTERESSE")
        print("=" * 90)
        dados_nulos = []
        for r in resultados:
            for col, n in r["nulos"].items():
                pct = n / r["total_linhas"] * 100 if r["total_linhas"] else 0
                dados_nulos.append({
                    "Arquivo": r["arquivo"], "Coluna": col, "Total_Linhas": r["total_linhas"],
                    "Nulos": n, "Percentual": round(pct, 2), "Status": classificar_nulos(pct),
                })

        if not dados_nulos:
            raise SystemExit("Nenhuma coluna de interesse encontrada. Verifique se os nomes estão corretos.")

        df_nulos = pl.DataFrame(dados_nulos).sort("Arquivo", "Coluna")
        print(df_nulos)

        print("\n" + "=" * 90)
        print("✅ RESUMO FINAL - COLUNAS RECOMENDADAS PARA O DATASET SINTÉTICO")
        print("=" * 90)
        recomendadas = df_nulos.filter(pl.col("Status") != "🔴 Crítico")
        print("\n📌 Colunas com menos de 20% de nulos (podem ser usadas):")
        if recomendadas.is_empty():
            print("   ⚠️ Nenhuma coluna com menos de 20% de nulos.")
        else:
            print(
                recomendadas.group_by("Coluna", maintain_order=True).agg(
                    pl.col("Arquivo").str.join(", "),
                    pl.format("mín: {}% | máx: {}%", pl.col("Percentual").min(), pl.col("Percentual").max())
                    .alias("Percentual"),
                )
            )

        criticas = df_nulos.filter(pl.col("Status") == "🔴 Crítico")
        if not criticas.is_empty():
            print("\n🔴 Colunas com MAIS de 20% de nulos (use com cuidado ou descarte):")
            print(criticas.select("Coluna", "Arquivo", "Percentual"))

    print("\n" + "=" * 90)
    print("📂 COLUNAS DISPONÍVEIS EM CADA ARQUIVO (TODAS)")
    print("=" * 90)
    for r in resultados:
        cols = r["colunas_presentes"]
        print(f"\n📄 {r['arquivo']} ({r['total_linhas']:,} linhas):")
        print(f"   {', '.join(cols[:10])}" + ("..." if len(cols) > 10 else ""))
