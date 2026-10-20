# ==============================================================================
# ETAPA 04 - ENRIQUECE COM AS TABELAS AUXILIARES DO SICOR (2019 A 2025)
# ==============================================================================
# Traz as descrições de fonte de recursos, subprograma, programa, modalidade,
# empreendimento (finalidade, atividade, MODALIDADE_Empreendimento, produto...),
# instrumento de crédito e tipo de garantia.
#
# O arquivo é processado em blocos (PyArrow) para não carregar tudo na RAM.
# As tabelas auxiliares são deduplicadas pela chave: sem isso, o LEFT JOIN
# multiplicaria as linhas da base principal.
#
# Entrada: PASTA_COM_NOME_MUNICIPIO/*_<ano>.parquet + PASTA_AUXILIARES/*.csv
# Saída:   PASTA_ENRIQUECIDO/full_<nome>.parquet
#
# Uso:
#   python preparacao_sicor/04_enriquecer_tabelas_auxiliares.py

import gc
import os

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from comum import ANOS_ESCOPO, PASTA_AUXILIARES, PASTA_COM_NOME_MUNICIPIO, PASTA_ENRIQUECIDO, listar_parquets

# ==============================================================================
# PARÂMETROS
# ==============================================================================
# Se faltar RAM, reduza para 10_000 ou 5_000; com RAM folgada, pode aumentar.
TAMANHO_BLOCO = 25_000
COMPRESSION = "snappy"

# Tabela auxiliar -> chave na base do SICOR
MAPA_AUXILIARES = {
    "FonteRecursos.csv": "CD_FONTE_RECURSO",
    "Subprogramas.csv": "CD_SUBPROGRAMA",
    "ProgramaModalidade.csv": "CD_PROGRAMA",
    "Programa.csv": "CD_PROGRAMA",
    "Empreendimento.csv": "CD_EMPREENDIMENTO",
    "InstrumentoCredito.csv": "CD_INST_CREDITO",
    "TipoGarantiaEmpreendimento.csv": "CD_TIPO_SEGURO",
}


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def normalizar_chave_expr(nome_coluna: str) -> pl.Expr:
    """Converte um código para Int64 extraindo a primeira sequência de dígitos.

    Evita problemas com códigos como "00123", "CODIGO 123", 123.0 e "123/45".
    """
    return pl.col(nome_coluna).cast(pl.String).str.extract(r"(\d+)").cast(pl.Int64, strict=False)


def tipo_efetivo(tipo_arrow):
    """Se a coluna for dictionary/categorical no Arrow, pega o tipo real dos valores."""
    while pa.types.is_dictionary(tipo_arrow):
        tipo_arrow = tipo_arrow.value_type
    return tipo_arrow


def expr_chave_base(nome_coluna: str, tipo_arrow) -> pl.Expr:
    """Normaliza a chave no arquivo principal: numérica vira Int64; texto tem os dígitos extraídos."""
    if pa.types.is_floating(tipo_arrow):
        return pl.col(nome_coluna).fill_nan(None).cast(pl.Int64, strict=False).alias(nome_coluna)
    if pa.types.is_integer(tipo_arrow):
        return pl.col(nome_coluna).cast(pl.Int64, strict=False).alias(nome_coluna)
    return normalizar_chave_expr(nome_coluna).alias(nome_coluna)


def encontrar_coluna_origem(cols: list[str], chave_alvo: str) -> str:
    """Coluna da chave na tabela auxiliar: #CODIGO, CODIGO ou a própria chave; senão, a primeira."""
    alvo = chave_alvo.upper()
    for c in cols:
        if c.upper() in ("#CODIGO", "CODIGO", alvo):
            return c
    return cols[0]


def carregar_auxiliar(nome_arquivo: str, nova_chave: str) -> pl.DataFrame | None:
    """Carrega a tabela auxiliar, normaliza a chave, remove nulos e deduplica."""
    caminho = os.path.join(PASTA_AUXILIARES, nome_arquivo)
    if not os.path.exists(caminho):
        print(f"⚠️ Arquivo auxiliar não encontrado: {nome_arquivo}")
        return None

    # Alguns arquivos estão em latin-1; se o Polars recusar, tenta utf8-lossy
    try:
        df = pl.read_csv(caminho, separator=";", encoding="latin-1", infer_schema_length=0, ignore_errors=True)
    except Exception:
        df = pl.read_csv(caminho, separator=";", encoding="utf8-lossy", infer_schema_length=0, ignore_errors=True)

    if not df.columns or df.is_empty():
        print(f"⚠️ Tabela auxiliar vazia: {nome_arquivo}")
        return None

    # Remove BOM e espaços nas bordas dos nomes de colunas
    df = df.rename({c: c.replace("﻿", "").strip() for c in df.columns})

    col_origem = encontrar_coluna_origem(df.columns, nova_chave)

    # Nome temporário para não conflitar quando a coluna de origem já tem o nome da chave final
    temp_chave = f"__{nova_chave}_aux__"
    colunas_para_remover = [c for c in {col_origem, nova_chave} if c in df.columns]

    df_aux = (
        df.lazy()
        .with_columns(normalizar_chave_expr(col_origem).alias(temp_chave))
        .drop(colunas_para_remover)
        .rename({temp_chave: nova_chave})
        .filter(pl.col(nova_chave).is_not_null())
        .unique(subset=[nova_chave], keep="first")
        .collect()
    )

    if df_aux.is_empty():
        print(f"⚠️ Tabela auxiliar sem chaves válidas: {nome_arquivo}")
        return None
    return df_aux


def enriquecer(df: pl.DataFrame, exprs_chave: list[pl.Expr],
               joins_info: list[tuple[str, pl.DataFrame, str]]) -> pl.DataFrame:
    """Aplica a normalização das chaves e os joins auxiliares em um bloco."""
    lf = df.lazy()
    if exprs_chave:
        lf = lf.with_columns(exprs_chave)
    for chave, df_aux, sufixo in joins_info:
        lf = lf.join(df_aux.lazy(), on=chave, how="left", suffix=sufixo)
    return lf.collect()


def processar_arquivo(pf: pq.ParquetFile, caminho_saida: str, exprs_chave: list[pl.Expr],
                      joins_info: list[tuple[str, pl.DataFrame, str]]) -> int:
    """Lê e escreve o arquivo em blocos, sem materializar tudo na RAM."""
    writer = None
    total_linhas = 0

    try:
        for batch in pf.iter_batches(batch_size=TAMANHO_BLOCO):
            df = enriquecer(pl.from_arrow(batch), exprs_chave, joins_info)
            tabela = df.to_arrow()

            if writer is None:
                writer = pq.ParquetWriter(caminho_saida, tabela.schema, compression=COMPRESSION)
            writer.write_table(tabela, row_group_size=TAMANHO_BLOCO)
            total_linhas += df.height

        if writer is None:
            # Arquivo de origem vazio: grava só o schema enriquecido
            df_vazio = enriquecer(pl.from_arrow(pf.schema_arrow.empty_table()), exprs_chave, joins_info)
            pq.write_table(df_vazio.to_arrow(), caminho_saida, compression=COMPRESSION)
        else:
            writer.close()
        return total_linhas

    except Exception:
        # Se falhar, fecha e remove o arquivo parcial
        if writer is not None:
            writer.close()
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)
        raise


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(PASTA_ENRIQUECIDO, exist_ok=True)

    arquivos = [
        a for a in listar_parquets(PASTA_COM_NOME_MUNICIPIO)
        if any(a.endswith(f"_{ano}.parquet") for ano in ANOS_ESCOPO)
    ]
    if not arquivos:
        raise SystemExit(f"⚠️ Nenhum arquivo encontrado em: {PASTA_COM_NOME_MUNICIPIO}")

    # 1) Descobre quais chaves existem em cada arquivo, para não fazer joins à toa
    print(f"🔎 Lendo schemas de {len(arquivos)} arquivos...")
    meta_arquivos = {}
    todas_chaves = set()
    for arq in arquivos:
        schema = pq.ParquetFile(arq).schema_arrow
        chaves_presentes = sorted({c for c in MAPA_AUXILIARES.values() if c in schema.names})
        meta_arquivos[arq] = (schema, chaves_presentes)
        todas_chaves.update(chaves_presentes)

    print(f"🚀 Iniciando processamento de {len(arquivos)} arquivos...")
    print(f"📦 Tamanho do bloco: {TAMANHO_BLOCO:,} linhas")

    # 2) Carrega só as tabelas auxiliares cujas chaves são usadas
    print("⏳ Carregando tabelas auxiliares necessárias...")
    dfs_aux = {}
    for nome_csv, chave in MAPA_AUXILIARES.items():
        if chave not in todas_chaves:
            continue
        df_aux = carregar_auxiliar(nome_csv, chave)
        if df_aux is not None:
            dfs_aux[nome_csv] = df_aux
            print(f"📄 {nome_csv}: {df_aux.height:,} linhas únicas")

    if not dfs_aux:
        print("⚠️ Nenhuma tabela auxiliar foi carregada.")

    # 3) Processa arquivo por arquivo, em blocos
    for idx, arq in enumerate(arquivos, 1):
        nome_base = os.path.basename(arq)
        caminho_saida = os.path.join(PASTA_ENRIQUECIDO, f"full_{nome_base}")
        schema, chaves_presentes = meta_arquivos[arq]

        exprs_chave = [expr_chave_base(c, tipo_efetivo(schema.field(c).type)) for c in chaves_presentes]

        # O sufixo (ex.: "_Empreendimento") diferencia colunas de mesmo nome
        # vindas de tabelas diferentes, como MODALIDADE_Empreendimento
        joins_info = [
            (MAPA_AUXILIARES[nome_csv], df_aux, f"_{os.path.splitext(nome_csv)[0]}")
            for nome_csv, df_aux in dfs_aux.items()
            if MAPA_AUXILIARES[nome_csv] in chaves_presentes
        ]

        try:
            total_linhas = processar_arquivo(pq.ParquetFile(arq), caminho_saida, exprs_chave, joins_info)
            print(f"[{idx}/{len(arquivos)}] ✅ {nome_base} | {total_linhas:,} linhas")
        except Exception as e:
            print(f"[{idx}/{len(arquivos)}] ❌ {nome_base} | Erro: {e}")
        finally:
            gc.collect()

    print("\n🎉 Processamento finalizado!")
