# ==============================================================================
# CAMINHOS E FUNÇÕES COMPARTILHADAS DA PREPARAÇÃO DO SICOR
# ==============================================================================
# Antes estes passos rodavam no Google Colab, lendo do Google Drive. A estrutura
# de pastas foi mantida igual à do Drive, dentro de PASTA_TCC.
#
# Por padrão PASTA_TCC é dados/sicor (dentro do projeto). Para apontar para
# outra pasta, por exemplo o Google Drive sincronizado no PC, defina a variável
# de ambiente TCC_PASTA_DADOS antes de rodar os scripts:
#   PowerShell:  $env:TCC_PASTA_DADOS = "G:\Meu Drive\TCC - Ideias"

import os
from pathlib import Path

import polars as pl

PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_TCC = Path(os.environ.get("TCC_PASTA_DADOS", PASTA_PROJETO / "dados" / "sicor"))

# 🟢 CSVs originais do SICOR: uma pasta por ano, com uma subpasta "CSV <ano>"
PASTA_CSV_SICOR = PASTA_TCC / "Microdados do Crédito Rural e do Proagro - Sicor"

# 🟢 Parquets de cada etapa (mesmos nomes de pasta usados no Drive)
_PASTA_PARQUETS = PASTA_TCC / "Microdados do Crédito Rural e do Proagro - Sicor - Parquet"
PASTA_PARQUET = _PASTA_PARQUETS / "Parquet"
PASTA_COM_CODIGO_MUNICIPIO = _PASTA_PARQUETS / "Parquet_Codigo_Municipios_IBGE"
PASTA_COM_NOME_MUNICIPIO = _PASTA_PARQUETS / "Parquet_Final_Cidades_IBGE"
PASTA_ENRIQUECIDO = _PASTA_PARQUETS / "Parquet_Enriquecido_Completo"
PASTA_LIMPO = _PASTA_PARQUETS / "Parquet_Limpo_Features_TCC"

# 🟢 Tabelas auxiliares do SICOR
PASTA_AUXILIARES = PASTA_TCC / "Planilhas para Relacionar - SICOR"
CAMINHO_COMPLEMENTO = (
    PASTA_PARQUET / "Planilha_Complementar_Parquet" / "SICOR_COMPLEMENTO_OPERACAO_BASICA.parquet"
)
CAMINHO_MUNICIPIOS_SICOR = PASTA_AUXILIARES / "MunicipiosSicor.csv"

# 🟢 Relatório de auditoria das colunas (gerado pelo 05, lido pelo 06)
PASTA_RELATORIOS = PASTA_TCC / "Relatorios_Saude_Colunas"
CAMINHO_AUDITORIA = PASTA_RELATORIOS / "auditoria_colunas_sicor.csv"

# 🟢 Gráficos e CSVs dos scripts de exploração
PASTA_SAIDA_EXPLORACAO = PASTA_PROJETO / "resultados" / "exploracao_sicor"

# Recorte de anos do TCC (os anos de 2013 a 2018 ficam de fora a partir do 04)
ANOS_ESCOPO = range(2019, 2026)


def listar_parquets(pasta) -> list[str]:
    """Lista os .parquet de uma pasta, em ordem alfabética (= ordem dos anos)."""
    return sorted(str(p) for p in Path(pasta).glob("*.parquet"))


def para_numero(coluna: str) -> pl.Expr:
    """Converte uma coluna de valor do SICOR para Float64.

    Nos Parquets os valores vêm como texto com ponto decimal ("83130.00").
    O notebook removia o ponto e dividia por 100, o que só funciona porque
    todos os valores têm exatamente duas casas decimais. Aqui o texto é
    convertido direto; se aparecer vírgula (formato brasileiro, "1.234,56"),
    o ponto de milhar é removido antes.
    """
    texto = pl.col(coluna).cast(pl.String)
    return (
        pl.when(texto.str.contains(","))
        .then(texto.str.replace_all(r"\.", "").str.replace(",", "."))
        .otherwise(texto)
        .cast(pl.Float64, strict=False)
        .alias(coluna)
    )
