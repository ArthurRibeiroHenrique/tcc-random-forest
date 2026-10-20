# ==============================================================================
# EXPLORAÇÃO - TOTAL DE CRÉDITO RURAL POR ANO, ESTADO E PROGRAMA
# ==============================================================================
# Junta as duas células de exploração do notebook (a que lia os Parquets
# originais e a que lia os Parquets com nome de município). Gera:
#   1. Total de crédito por ano (bilhões de R$)
#   2. Top 10 estados do ano mais recente
#   3. Top 10 programas do ano mais recente
#   4. Juros médio por estado, comparando os anos
#   5. Tabela-resumo por ano no console
# Os gráficos são salvos em HTML (Plotly) em resultados/exploracao_sicor.
#
# Uso:
#   python preparacao_sicor/explorar_credito.py
#   python preparacao_sicor/explorar_credito.py --pasta CAMINHO --inspecionar

import argparse
import os
import re

import plotly.express as px
import polars as pl

from comum import PASTA_COM_NOME_MUNICIPIO, PASTA_SAIDA_EXPLORACAO, listar_parquets, para_numero

COLUNA_VALOR = "VL_PARC_CREDITO"
COLUNA_JUROS = "VL_JUROS"
COLUNA_ESTADO = "CD_ESTADO"
COLUNA_PROGRAMA = "CD_PROGRAMA"

# Linhas por arquivo carregadas no modo inspeção
AMOSTRA_INSPECAO = 1000


def extrair_ano(nome_arquivo: str) -> int | None:
    match = re.search(r"(\d{4})", nome_arquivo)
    return int(match.group(1)) if match else None


def inspecionar_todas_colunas(caminho: str, n_linhas: int = AMOSTRA_INSPECAO) -> pl.DataFrame:
    """Carrega uma amostra com TODAS as colunas para validar schema e valores brutos."""
    df = pl.read_parquet(caminho, n_rows=n_linhas)
    print(f"\n🔍 INSPEÇÃO: {os.path.basename(caminho)}")
    print(f"   Shape: {df.shape}")
    print(f"   Colunas ({len(df.columns)}): {df.columns}")
    print(f"   Schema:\n{df.schema}")
    print("\n   Primeiras 5 linhas:")
    print(df.head(5))
    return df


def processar_arquivo(caminho: str, ano: int) -> dict | None:
    lf = pl.scan_parquet(caminho)
    colunas = set(lf.collect_schema().names())

    if COLUNA_VALOR not in colunas:
        print(f"⚠️  {os.path.basename(caminho)}: '{COLUNA_VALOR}' não encontrada. Pulando.")
        return None

    lf = lf.with_columns(para_numero(COLUNA_VALOR))
    tem_juros = COLUNA_JUROS in colunas
    if tem_juros:
        lf = lf.with_columns(para_numero(COLUNA_JUROS))

    def agregados(*extras):
        exprs = [pl.sum(COLUNA_VALOR).alias("TOTAL_CREDITO"), pl.len().alias("OPERACOES"), *extras]
        if tem_juros:
            exprs.append(pl.mean(COLUNA_JUROS).alias("JUROS_MEDIO"))
        return exprs

    resultado = {"ano": ano, "global": lf.select(agregados()).collect().to_dicts()[0]}

    if COLUNA_ESTADO in colunas:
        resultado["estado"] = (
            lf.group_by(COLUNA_ESTADO).agg(agregados()).collect().with_columns(pl.lit(ano).alias("ANO"))
        )

    if COLUNA_PROGRAMA in colunas:
        resultado["programa"] = (
            lf.group_by(COLUNA_PROGRAMA)
            .agg(pl.sum(COLUNA_VALOR).alias("TOTAL_CREDITO"), pl.len().alias("OPERACOES"))
            .sort("TOTAL_CREDITO", descending=True)
            .head(10)
            .collect()
            .with_columns(pl.lit(ano).alias("ANO"))
        )

    return resultado


def salvar(fig, nome: str) -> None:
    caminho = PASTA_SAIDA_EXPLORACAO / nome
    fig.write_html(caminho)
    print(f"📈 Gráfico salvo em: {caminho}")


def gerar_graficos(agregados: list[dict]) -> None:
    os.makedirs(PASTA_SAIDA_EXPLORACAO, exist_ok=True)

    df_evolucao = pl.DataFrame([
        {"ANO": a["ano"],
         "TOTAL_CREDITO_BI": a["global"]["TOTAL_CREDITO"] / 1e9,
         "OPERACOES": a["global"]["OPERACOES"]}
        for a in agregados
    ]).sort("ANO")

    fig1 = px.bar(df_evolucao.to_pandas(), x="ANO", y="TOTAL_CREDITO_BI",
                  title="💰 Total de Crédito por Ano (Bilhões de R$)")
    fig1.update_traces(texttemplate="%{y:.1f} Bi", textposition="outside")
    fig1.update_layout(yaxis_title="Bilhões de R$")
    salvar(fig1, "01_credito_por_ano.html")

    ano_recente = max(a["ano"] for a in agregados)
    recente = next(a for a in agregados if a["ano"] == ano_recente)

    if "estado" in recente:
        top_estados = (
            recente["estado"].sort("TOTAL_CREDITO", descending=True).head(10)
            .with_columns((pl.col("TOTAL_CREDITO") / 1e9).alias("TOTAL_CREDITO_BI"))
        )
        fig2 = px.bar(top_estados.to_pandas(), x=COLUNA_ESTADO, y="TOTAL_CREDITO_BI",
                      title=f"🏆 Top 10 Estados - {ano_recente} (Bilhões de R$)")
        fig2.update_traces(texttemplate="%{y:.1f} Bi", textposition="outside")
        fig2.update_layout(yaxis_title="Bilhões de R$")
        salvar(fig2, "02_top10_estados.html")

    if "programa" in recente:
        fig3 = px.pie(recente["programa"].to_pandas(), values="OPERACOES", names=COLUNA_PROGRAMA,
                      title=f"📊 Top 10 Programas - {ano_recente}", hole=0.3)
        salvar(fig3, "03_top10_programas.html")

    dfs_juros = [
        a["estado"].select(COLUNA_ESTADO, "JUROS_MEDIO", "ANO")
        for a in agregados
        if "estado" in a and "JUROS_MEDIO" in a["estado"].columns
    ]
    if len(dfs_juros) >= 2 and "estado" in recente:
        top10 = recente["estado"].sort("OPERACOES", descending=True).head(10)[COLUNA_ESTADO].to_list()
        df_juros = pl.concat(dfs_juros).filter(pl.col(COLUNA_ESTADO).is_in(top10))
        fig4 = px.bar(df_juros.to_pandas(), x=COLUNA_ESTADO, y="JUROS_MEDIO", color="ANO", barmode="group",
                      title="📉 Juros Médios - Top 10 Estados (Comparativo Anual)",
                      labels={"JUROS_MEDIO": "Taxa Média (%)"})
        salvar(fig4, "04_juros_por_estado.html")

    print("\n📋 RESUMO EXECUTIVO:")
    for linha in df_evolucao.iter_rows(named=True):
        print(f"   {linha['ANO']}  R$ {linha['TOTAL_CREDITO_BI']:>8,.2f} Bi  {linha['OPERACOES']:>12,} operações")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploração do crédito rural do SICOR.")
    parser.add_argument("--pasta", default=str(PASTA_COM_NOME_MUNICIPIO), help="pasta com os .parquet do SICOR")
    parser.add_argument("--inspecionar", action="store_true",
                        help="antes, mostra uma amostra com todas as colunas dos 3 primeiros arquivos")
    args = parser.parse_args()

    arquivos = listar_parquets(args.pasta)
    print(f"📂 Encontrados {len(arquivos)} arquivos parquet.")

    if args.inspecionar:
        print("\n" + "=" * 60)
        print("🔎 MODO INSPEÇÃO: CARREGANDO TODAS AS COLUNAS (AMOSTRA)")
        print("=" * 60)
        for arq in arquivos[:3]:
            inspecionar_todas_colunas(arq)

    print("\n" + "=" * 60)
    print("📊 PROCESSANDO AGREGADOS PARA VISUALIZAÇÕES")
    print("=" * 60)
    agregados = []
    for arq in arquivos:
        ano = extrair_ano(os.path.basename(arq))
        if ano is None:
            continue
        print(f"⚙️  Processando {os.path.basename(arq)} ({ano})...")
        resultado = processar_arquivo(arq, ano)
        if resultado:
            agregados.append(resultado)

    if not agregados:
        raise SystemExit("❌ Nenhum dado válido processado.")
    gerar_graficos(agregados)
