# ==============================================================================
# MODELO RANDOM FOREST DE INADIMPLÊNCIA (TCC, seções 2.3, 2.5 e 2.6)
# ==============================================================================
# 1. Base de validação (seção 2.5): sorteia 5.000 operações reais de LAVOURA
#    que receberam clima no cruzamento (dados/sicor_com_clima) e gera para elas
#    o alvo sintético de inadimplência: combinação ponderada de fatores
#    financeiros e climáticos + ruído gaussiano, com corte no percentil da
#    taxa de inadimplência alvo (3% a 7%).
# 2. Holdout estratificado 80/20 (seção 2.6.1).
# 3. Para cada cenário (seção 2.6.4) - BASELINE (só SICOR) e APRIMORADA
#    (SICOR + INMET) - GridSearchCV com k-fold estratificado, k=5, só dentro
#    dos 80% de treino, otimizando o F1 (seção 2.6.2).
# 4. Avaliação nos 20% de teste: matriz de confusão, precisão, recall, F1,
#    AUC-ROC (seção 2.3), OOB (seção 2.6.3), Brier score e curva de
#    calibração (seção 2.6.5) e importância das variáveis.
# 5. Faixas de roteamento das operações de teste (seção 2.5.4).
#
# ATENÇÃO: o SICOR não informa inadimplência. O alvo é gerado a partir das
# próprias variáveis, com os pesos de PESOS_RISCO; por isso o ganho do cenário
# APRIMORADA sobre o BASELINE é consequência desses pesos. O experimento valida
# a esteira de modelagem, não o efeito real do clima sobre a inadimplência.
#
# Os nulos das variáveis climáticas não são imputados: o RandomForestClassifier
# do scikit-learn (>= 1.4) trata valores ausentes nativamente.
#
# Pré-requisito: rodar antes o cruzar_sicor_inmet.py
#
# Uso:
#   python random_forest.py
#   python random_forest.py --operacoes 5000 --taxa-inadimplencia 0.05 --semente 42

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # só salva as figuras em arquivo, sem abrir janela

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ==============================================================================
# CONFIGURAÇÕES DE CAMINHOS
# ==============================================================================
PASTA_PROJETO = Path(__file__).resolve().parent

# 🟢 Parquets do SICOR já com as colunas climáticas (saída do cruzar_sicor_inmet.py)
PASTA_SICOR_CLIMA = PASTA_PROJETO / "dados" / "sicor_com_clima"

# 🟢 Métricas, gráficos, base sintética e modelos treinados
PASTA_RESULTADOS = PASTA_PROJETO / "resultados" / "random_forest"

# ==============================================================================
# REGRAS DO TCC
# ==============================================================================
N_OPERACOES = 5_000          # tamanho da base de validação (seção 1.4)
TAXA_INADIMPLENCIA = 0.05    # dentro da faixa de 3% a 7% (seção 2.5)
PROPORCAO_TESTE = 0.20       # holdout 80/20 (seção 2.6.1)
K_FOLDS = 5                  # validação cruzada (seção 2.6.2)
SEMENTE = 42

# MODALIDADE_Empreendimento = "LAVOURA" é o filtro da base (seção 2.5); depois
# dele a coluna fica constante e por isso não entra como feature.
FILTRO_MODALIDADE = "LAVOURA"

# Variáveis do SICOR (Tabela 1). CD_IBGE_MUNICIPIO entra como número: o código
# começa pelo código da UF, então as árvores conseguem separar regiões.
FEATURES_SICOR = ["VL_JUROS", "VL_AREA_FINANC", "VL_PARC_CREDITO", "CD_ESTADO", "CD_IBGE_MUNICIPIO"]
FEATURES_CATEGORICAS = ["CD_ESTADO"]

# Variáveis climáticas do INMET (Tabela 2)
FEATURES_CLIMA = ["PRECIP_90D", "DESVIO_%", "VERANICOS_90D"]

CENARIOS = {
    "BASELINE": FEATURES_SICOR,
    "APRIMORADA": FEATURES_SICOR + FEATURES_CLIMA,
}

# ------------------------------------------------------------------------------
# ALVO SINTÉTICO (seção 2.5)
# ------------------------------------------------------------------------------
# Cada fator vira um percentil de 0 a 1 dentro da base (robusto aos valores
# extremos do DESVIO_%, que passam de 100.000%). Fator nulo vale 0,5 (neutro).
#   JUROS               -> VL_JUROS alto (custo do crédito; o SICOR não traz o CET)
#   CREDITO_POR_HECTARE -> VL_PARC_CREDITO / VL_AREA_FINANC alto (endividamento)
#   DEFICIT_CHUVA       -> DESVIO_% negativo (chuva abaixo da média histórica)
#   VERANICOS           -> VERANICOS_90D alto
PESOS_RISCO = {
    "JUROS": 0.30,
    "CREDITO_POR_HECTARE": 0.20,
    "DEFICIT_CHUVA": 0.30,
    "VERANICOS": 0.20,
}
DESVIO_RUIDO = 0.10  # desvio padrão do ruído gaussiano somado à probabilidade

# ------------------------------------------------------------------------------
# HIPERPARÂMETROS TESTADOS NO GridSearchCV
# ------------------------------------------------------------------------------
GRADE_HIPERPARAMETROS = {
    "rf__n_estimators": [300],
    "rf__max_depth": [None, 8, 16],
    "rf__min_samples_leaf": [1, 5, 20],
    "rf__max_features": ["sqrt", 0.5],
    "rf__class_weight": [None, "balanced_subsample"],
}

# Faixas de roteamento (seção 2.5.4)
LIMIAR_APROVACAO = 0.30
LIMIAR_REPROVACAO = 0.65

# Cores dos gráficos: uma por cenário, sempre a mesma
CORES_CENARIO = {"BASELINE": "#2a78d6", "APRIMORADA": "#eb6834"}
TINTA_PRIMARIA, TINTA_SECUNDARIA, TINTA_SUAVE = "#0b0b0b", "#52514e", "#898781"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA_SECUNDARIA, "axes.titlecolor": TINTA_PRIMARIA,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE, "text.color": TINTA_PRIMARIA,
    "legend.frameon": False, "font.size": 10,
})


# ==============================================================================
# 1. BASE DE VALIDAÇÃO
# ==============================================================================
def carregar_operacoes_lavoura(pasta) -> pl.DataFrame:
    """Operações de lavoura que receberam variáveis climáticas no cruzamento."""
    return (
        pl.scan_parquet(Path(pasta) / "*.parquet")
        .filter(
            (pl.col("MODALIDADE_Empreendimento") == FILTRO_MODALIDADE)
            & pl.col("PRECIP_90D").is_not_null()
        )
        .select(
            "DT_EMISSAO",
            "CD_ESTADO",
            pl.col("CD_IBGE_MUNICIPIO").cast(pl.Int64, strict=False),
            # No SICOR os valores vêm como texto com ponto decimal ("83130.00")
            *[pl.col(c).cast(pl.Float64, strict=False) for c in ["VL_JUROS", "VL_AREA_FINANC", "VL_PARC_CREDITO"]],
            "COD_ESTACAO",
            "QUALIDADE_ESTACAO",
            *FEATURES_CLIMA,
        )
        .collect()
    )


def percentil(serie: pd.Series) -> pd.Series:
    """Posição de cada valor na base, de 0 a 1; nulo fica neutro (0,5)."""
    return serie.rank(pct=True).fillna(0.5)


def gerar_alvo_sintetico(df: pd.DataFrame, taxa: float, semente: int) -> pd.DataFrame:
    area = df["VL_AREA_FINANC"].where(df["VL_AREA_FINANC"] > 0)
    fatores = {
        "JUROS": percentil(df["VL_JUROS"]),
        "CREDITO_POR_HECTARE": percentil(df["VL_PARC_CREDITO"] / area),
        "DEFICIT_CHUVA": percentil(-df["DESVIO_%"]),
        "VERANICOS": percentil(df["VERANICOS_90D"]),
    }
    rng = np.random.default_rng(semente)
    probabilidade = sum(PESOS_RISCO[nome] * valor for nome, valor in fatores.items())
    probabilidade = probabilidade + rng.normal(0, DESVIO_RUIDO, len(df))

    # Inadimplente = probabilidade acima do percentil da taxa alvo
    corte = np.quantile(probabilidade, 1 - taxa)
    return df.assign(PROB_SINTETICA=probabilidade, INADIMPLENTE=(probabilidade > corte).astype(int))


# ==============================================================================
# 2. MODELO
# ==============================================================================
def montar_modelo(features: list[str], semente: int) -> Pipeline:
    # O one-hot fica dentro do pipeline, então é ajustado só nos dados de
    # treino de cada fold (evita vazamento)
    categoricas = [c for c in features if c in FEATURES_CATEGORICAS]
    preparo = ColumnTransformer(
        [("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    floresta = RandomForestClassifier(oob_score=True, random_state=semente, n_jobs=1)
    return Pipeline([("preparo", preparo), ("rf", floresta)])


def treinar_cenario(features, X_treino, y_treino, semente) -> GridSearchCV:
    busca = GridSearchCV(
        montar_modelo(features, semente),
        GRADE_HIPERPARAMETROS,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1",  # F1 é a métrica primária (seção 2.3)
        cv=StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=semente),
        n_jobs=-1,
    )
    busca.fit(X_treino[features], y_treino)
    return busca


def classificar_roteamento(probabilidades: np.ndarray) -> pd.Categorical:
    return pd.cut(
        probabilidades,
        bins=[-np.inf, LIMIAR_APROVACAO, LIMIAR_REPROVACAO, np.inf],
        labels=["Aprovação imediata (<30%)", "Análise humana (30% a 65%)", "Reprovação defensiva (>65%)"],
        right=False,
    )


def avaliar_cenario(nome, busca, features, X_treino, y_treino, X_teste, y_teste, semente) -> dict:
    modelo = busca.best_estimator_
    floresta = modelo.named_steps["rf"]
    proba = modelo.predict_proba(X_teste[features])[:, 1]
    previsto = (proba >= 0.5).astype(int)

    # OOB: cada árvore é avaliada nas operações que ficaram fora do seu bootstrap
    proba_oob = floresta.oob_decision_function_[:, 1]

    # Importância por permutação, no teste e nas variáveis originais
    permutacao = permutation_importance(
        modelo, X_teste[features], y_teste, scoring="roc_auc", n_repeats=10, random_state=semente, n_jobs=-1
    )
    # Importância de Gini (MDI); as colunas do one-hot voltam para CD_ESTADO
    nomes = modelo.named_steps["preparo"].get_feature_names_out()
    gini = pd.Series(floresta.feature_importances_, index=nomes)
    gini = gini.groupby(lambda n: next((c for c in FEATURES_CATEGORICAS if n.startswith(f"{c}_")), n)).sum()

    importancia = pd.DataFrame({
        "CENARIO": nome,
        "VARIAVEL": features,
        "IMPORTANCIA_PERMUTACAO_AUC": permutacao.importances_mean,
        "DESVIO_PERMUTACAO": permutacao.importances_std,
        "IMPORTANCIA_GINI": [gini[f] for f in features],
    }).sort_values("IMPORTANCIA_PERMUTACAO_AUC", ascending=False)

    roteamento = (
        pd.DataFrame({"FAIXA": classificar_roteamento(proba), "INADIMPLENTE": y_teste.to_numpy()})
        .groupby("FAIXA", observed=False)
        .agg(OPERACOES=("INADIMPLENTE", "size"), INADIMPLENTES=("INADIMPLENTE", "sum"))
        .reset_index()
        .assign(CENARIO=nome)
    )

    metricas = {
        "CENARIO": nome,
        "F1_CV": busca.cv_results_["mean_test_f1"][busca.best_index_],
        "AUC_CV": busca.cv_results_["mean_test_roc_auc"][busca.best_index_],
        "OOB_ACURACIA": floresta.oob_score_,
        "OOB_AUC": roc_auc_score(y_treino, proba_oob),
        "PRECISAO": precision_score(y_teste, previsto, zero_division=0),
        "RECALL": recall_score(y_teste, previsto),
        "F1": f1_score(y_teste, previsto),
        "AUC_ROC": roc_auc_score(y_teste, proba),
        "BRIER": brier_score_loss(y_teste, proba),
    }
    return {
        "metricas": metricas,
        "melhores_parametros": busca.best_params_,
        "matriz_confusao": confusion_matrix(y_teste, previsto),
        "proba": proba,
        "importancia": importancia,
        "roteamento": roteamento,
        "modelo": modelo,
    }


# ==============================================================================
# 3. GRÁFICOS
# ==============================================================================
def grafico_roc(resultados, y_teste, caminho):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=TINTA_SUAVE, linewidth=1, linestyle="--", label="Aleatório")
    for nome, r in resultados.items():
        fpr, tpr, _ = roc_curve(y_teste, r["proba"])
        ax.plot(fpr, tpr, color=CORES_CENARIO[nome], linewidth=2,
                label=f"{nome} (AUC = {r['metricas']['AUC_ROC']:.3f})")
    ax.set(title="Curva ROC no conjunto de teste", xlabel="Taxa de falsos positivos",
           ylabel="Taxa de verdadeiros positivos", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def grafico_calibracao(resultados, y_teste, caminho):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=TINTA_SUAVE, linewidth=1, linestyle="--", label="Calibração perfeita")
    for nome, r in resultados.items():
        # Faixas por quantil: com ~5% de inadimplentes, faixas de largura fixa
        # deixariam quase todas as operações na primeira
        real, previsto = calibration_curve(y_teste, r["proba"], n_bins=10, strategy="quantile")
        ax.plot(previsto, real, color=CORES_CENARIO[nome], linewidth=2, marker="o", markersize=5,
                label=f"{nome} (Brier = {r['metricas']['BRIER']:.4f})")
    ax.set(title="Curva de calibração no conjunto de teste", xlabel="Probabilidade prevista (média da faixa)",
           ylabel="Proporção real de inadimplentes", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def grafico_matriz_confusao(nome, matriz, caminho):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    exibicao = ConfusionMatrixDisplay(matriz, display_labels=["Adimplente", "Inadimplente"])
    exibicao.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.grid(False)
    ax.set(title=f"Matriz de confusão - {nome}", xlabel="Previsto", ylabel="Real")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def grafico_importancia(importancia: pd.DataFrame, caminho):
    variaveis = (
        importancia.groupby("VARIAVEL")["IMPORTANCIA_PERMUTACAO_AUC"].max().sort_values().index.tolist()
    )
    y = np.arange(len(variaveis))
    altura = 0.38
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(variaveis) + 1.2))
    for i, nome in enumerate(CENARIOS):
        valores = (
            importancia[importancia["CENARIO"] == nome].set_index("VARIAVEL")["IMPORTANCIA_PERMUTACAO_AUC"]
            .reindex(variaveis)
        )
        ax.barh(y + (0.5 - i) * (altura + 0.02), valores.fillna(0), height=altura,
                color=CORES_CENARIO[nome], label=nome)
    ax.set_yticks(y, variaveis)
    ax.tick_params(axis="y", colors=TINTA_SECUNDARIA)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set(title="Importância por permutação (queda da AUC no teste)", xlabel="Queda média da AUC-ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina e avalia o Random Forest de inadimplência do TCC.")
    parser.add_argument("--entrada", default=str(PASTA_SICOR_CLIMA), help="pasta com os .parquet cruzados")
    parser.add_argument("--saida", default=str(PASTA_RESULTADOS), help="pasta dos resultados")
    parser.add_argument("--operacoes", type=int, default=N_OPERACOES, help="tamanho da base de validação")
    parser.add_argument("--taxa-inadimplencia", type=float, default=TAXA_INADIMPLENCIA,
                        help="proporção de inadimplentes no alvo sintético (0.03 a 0.07)")
    parser.add_argument("--semente", type=int, default=SEMENTE, help="semente aleatória")
    args = parser.parse_args()

    pasta_saida = Path(args.saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # --- 1. Base de validação ---
    lavoura = carregar_operacoes_lavoura(args.entrada)
    print(f"🌾 Operações de {FILTRO_MODALIDADE} com clima: {lavoura.height:,}")
    base = lavoura.sample(n=args.operacoes, seed=args.semente).to_pandas()
    base = gerar_alvo_sintetico(base, args.taxa_inadimplencia, args.semente)
    print(f"🎲 Base de validação: {len(base):,} operações sorteadas, "
          f"{base['INADIMPLENTE'].sum()} inadimplentes ({base['INADIMPLENTE'].mean():.1%}) no alvo sintético")
    base.to_csv(pasta_saida / "base_validacao.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

    # --- 2. Holdout estratificado 80/20 ---
    X = base[FEATURES_SICOR + FEATURES_CLIMA]
    y = base["INADIMPLENTE"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=PROPORCAO_TESTE, stratify=y, random_state=args.semente
    )
    print(f"✂️  Treino: {len(X_treino):,} ({y_treino.mean():.1%} inadimplentes) | "
          f"Teste: {len(X_teste):,} ({y_teste.mean():.1%} inadimplentes)")

    # --- 3 e 4. Treino e avaliação de cada cenário ---
    resultados = {}
    for nome, features in CENARIOS.items():
        print(f"\n⏳ {nome}: GridSearchCV ({K_FOLDS} folds) com {len(features)} variáveis...")
        busca = treinar_cenario(features, X_treino, y_treino, args.semente)
        resultados[nome] = avaliar_cenario(nome, busca, features, X_treino, y_treino, X_teste, y_teste,
                                           args.semente)
        print(f"   ✅ Melhores hiperparâmetros: {busca.best_params_}")
        joblib.dump(resultados[nome]["modelo"], pasta_saida / f"modelo_{nome.lower()}.joblib")

    # --- Arquivos de resultado ---
    metricas = pd.DataFrame([r["metricas"] for r in resultados.values()])
    metricas.to_csv(pasta_saida / "metricas.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    importancia = pd.concat([r["importancia"] for r in resultados.values()])
    importancia.to_csv(pasta_saida / "importancia_variaveis.csv", sep=";", decimal=",", index=False,
                       encoding="utf-8-sig")
    roteamento = pd.concat([r["roteamento"] for r in resultados.values()])
    roteamento.to_csv(pasta_saida / "roteamento.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    with open(pasta_saida / "configuracao.json", "w", encoding="utf-8") as f:
        json.dump({
            "operacoes": args.operacoes,
            "taxa_inadimplencia": args.taxa_inadimplencia,
            "semente": args.semente,
            "pesos_risco": PESOS_RISCO,
            "desvio_ruido": DESVIO_RUIDO,
            "cenarios": CENARIOS,
            "melhores_parametros": {n: r["melhores_parametros"] for n, r in resultados.items()},
        }, f, ensure_ascii=False, indent=2)

    grafico_roc(resultados, y_teste, pasta_saida / "curva_roc.png")
    grafico_calibracao(resultados, y_teste, pasta_saida / "curva_calibracao.png")
    for nome, r in resultados.items():
        grafico_matriz_confusao(nome, r["matriz_confusao"], pasta_saida / f"matriz_confusao_{nome.lower()}.png")
    grafico_importancia(importancia, pasta_saida / "importancia_variaveis.png")

    # --- Resumo no console ---
    print("\n" + "=" * 60)
    print("📋 COMPARATIVO DOS CENÁRIOS (conjunto de teste)")
    print("=" * 60)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(metricas.set_index("CENARIO").T)

    for nome, r in resultados.items():
        tn, fp, fn, tp = r["matriz_confusao"].ravel()
        print(f"\n🔢 {nome} - matriz de confusão: VN={tn} FP={fp} FN={fn} VP={tp}")
        print("   Roteamento das operações de teste:")
        for linha in r["roteamento"].itertuples():
            print(f"     {linha.FAIXA:<30} {linha.OPERACOES:>5} operações, {linha.INADIMPLENTES:>3} inadimplentes")
        print("   Importância por permutação (queda da AUC):")
        for linha in r["importancia"].itertuples():
            print(f"     {linha.VARIAVEL:<18} {linha.IMPORTANCIA_PERMUTACAO_AUC:+.4f}")

    print("\n⚠️  Alvo sintético: o ganho do APRIMORADA reflete os pesos de PESOS_RISCO, "
          "não um efeito observado em dados reais de inadimplência.")
    print(f"📦 Resultados salvos em: {pasta_saida}")
