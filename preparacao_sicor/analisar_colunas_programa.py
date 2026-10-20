# ==============================================================================
# EXPLORAÇÃO - SAÚDE DAS COLUNAS DE PROGRAMA (CD_PROGRAMA, CD_SUBPROGRAMA...)
# ==============================================================================
# Para cada coluna cujo nome contém "programa": nulos, valores únicos e os
# 5 valores mais frequentes. Classificação: até 5% excelente | até 20%
# razoável | acima crítico.
#
# Uso:
#   python preparacao_sicor/analisar_colunas_programa.py
#   python preparacao_sicor/analisar_colunas_programa.py --pasta CAMINHO

import argparse
import os
import re

import polars as pl
import pyarrow.parquet as pq

from comum import PASTA_PARQUET, listar_parquets

# Só calcula o top 5 quando a coluna tem menos valores distintos que isso
LIMITE_UNICOS_TOP5 = 1000


def emoji_status(pct: float) -> str:
    return "🟢" if pct <= 5 else ("🟡" if pct <= 20 else "🔴")


def analisar_colunas_programa(caminho: str) -> dict:
    total_linhas = pq.ParquetFile(caminho).metadata.num_rows
    lf = pl.scan_parquet(caminho)
    cols_programa = [c for c in lf.collect_schema().names() if re.search(r"programa", c, re.IGNORECASE)]

    resultado = {"arquivo": os.path.basename(caminho), "total_linhas": total_linhas,
                 "cols_programa": cols_programa, "analise": {}}
    if not cols_programa:
        return resultado

    # Nulos e distintos de todas as colunas numa única passada
    exprs = []
    for col in cols_programa:
        exprs.append(pl.col(col).is_null().sum().alias(f"{col}_nulos"))
        exprs.append(pl.col(col).n_unique().alias(f"{col}_unicos"))
    agregados = lf.select(exprs).collect()

    for col in cols_programa:
        nulos = agregados[f"{col}_nulos"][0]
        unicos = agregados[f"{col}_unicos"][0]

        if 0 < unicos < LIMITE_UNICOS_TOP5:
            top = lf.group_by(col).len().sort("len", descending=True).head(5).collect()
            top5 = dict(zip(top[col].to_list(), top["len"].to_list()))
        else:
            top5 = {"(many distinct)": unicos}

        resultado["analise"][col] = {
            "total": total_linhas,
            "nulos": nulos,
            "pct_nulos": round(nulos / total_linhas * 100, 2) if total_linhas else 0,
            "unicos": unicos,
            "top5": top5,
        }
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saúde das colunas de programa do SICOR.")
    parser.add_argument("--pasta", default=str(PASTA_PARQUET), help="pasta com os .parquet do SICOR")
    args = parser.parse_args()

    arquivos = listar_parquets(args.pasta)
    if not arquivos:
        raise SystemExit("❌ Nenhum arquivo .parquet encontrado!")

    print(f"🔍 Encontrei {len(arquivos)} arquivo(s):")
    for a in arquivos:
        print(f"   - {os.path.basename(a)}")

    print("\n⏳ Analisando colunas com 'PROGRAMA' em cada arquivo...\n")
    resultados = []
    for arq in arquivos:
        print(f"   📂 {os.path.basename(arq)}...")
        resultados.append(analisar_colunas_programa(arq))

    print("\n" + "=" * 90)
    print("📋 COLUNAS COM 'PROGRAMA' ENCONTRADAS EM CADA ARQUIVO")
    print("=" * 90)
    for r in resultados:
        print(f"\n📄 {r['arquivo']} ({r['total_linhas']:,} linhas):")
        if not r["cols_programa"]:
            print("   ⚠️ Nenhuma coluna com 'PROGRAMA'.")
        for col, a in r["analise"].items():
            print(f"   {emoji_status(a['pct_nulos'])} {col}: nulos={a['nulos']:,} "
                  f"({a['pct_nulos']:.2f}%), únicos={a['unicos']:,}")
            print(f"      Top 5: {a['top5']}")

    # Consolidado por coluna, ponderado pelo total de linhas de cada arquivo
    consolidado = {}
    for r in resultados:
        for col, a in r["analise"].items():
            c = consolidado.setdefault(col, {"arquivos": [], "unicos": [], "linhas": 0, "nulos": 0})
            c["arquivos"].append(r["arquivo"])
            c["unicos"].append(a["unicos"])
            c["linhas"] += a["total"]
            c["nulos"] += a["nulos"]

    linhas = []
    for col, c in consolidado.items():
        pct = round(c["nulos"] / c["linhas"] * 100, 2) if c["linhas"] else 0
        status = "🟢 Excelente" if pct <= 5 else ("🟡 Razoável" if pct <= 20 else "🔴 Crítico")
        linhas.append({"Coluna": col, "Arquivos": len(c["arquivos"]), "Pct_Nulos_Medio": pct,
                       "Status": status, "Unicos (min/max)": f"{min(c['unicos'])} / {max(c['unicos'])}"})

    print("\n" + "=" * 90)
    print("📊 RELATÓRIO CONSOLIDADO POR COLUNA 'PROGRAMA' (agregado)")
    print("=" * 90)
    if not linhas:
        raise SystemExit("Nenhuma coluna com 'PROGRAMA' encontrada.")
    df_consolidado = pl.DataFrame(linhas).sort("Pct_Nulos_Medio")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        print(df_consolidado)

    print("\n" + "=" * 90)
    print("✅ RECOMENDAÇÃO")
    print("=" * 90)
    por_status = {s: df_consolidado.filter(pl.col("Status") == s)["Coluna"].to_list()
                  for s in ("🟢 Excelente", "🟡 Razoável", "🔴 Crítico")}
    if por_status["🟢 Excelente"]:
        print(f"\n🟢 Colunas com ótima saúde (≤5% nulos): {', '.join(por_status['🟢 Excelente'])}")
    if por_status["🟡 Razoável"]:
        print(f"\n🟡 Colunas com saúde razoável (5-20% nulos): {', '.join(por_status['🟡 Razoável'])}"
              " - Avalie se compensa usar (perda de até 20% das observações).")
    if por_status["🔴 Crítico"]:
        print(f"\n🔴 Colunas com muitos nulos (>20%): {', '.join(por_status['🔴 Crítico'])}"
              " - Evite usar ou faça imputação com cautela.")

    print("\n🏁 Análise concluída!")
