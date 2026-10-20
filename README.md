# TCC - Random Forest de inadimplência no crédito rural

Esteira em Python que integra o SICOR (Banco Central) e o clima do INMET e treina o modelo Random Forest descrito no TCC.

```
pip install -r requirements.txt
```

## Ordem de execução

| Passo | Script | Saída |
|---|---|---|
| 1 | `preparacao_sicor/01_csv_para_parquet.py` | Parquets do SICOR, um por ano |
| 2 | `preparacao_sicor/02_adicionar_codigo_municipio.py` | + `CD_IBGE_MUNICIPIO` (planilha complementar) |
| 3 | `preparacao_sicor/03_adicionar_nome_municipio.py` | + nome do município |
| 4 | `preparacao_sicor/04_enriquecer_tabelas_auxiliares.py` | + descrições (programa, empreendimento...), 2019 a 2025 |
| 5 | `preparacao_sicor/05_auditar_colunas.py` | relatório de saúde das colunas |
| 6 | `preparacao_sicor/06_limpar_colunas.py` | Parquets só com as colunas úteis |
| 7 | `mapa_municipio_estacao.py` | `dados/mapa_municipio_estacao.csv` |
| 8 | `cruzar_sicor_inmet.py --sicor <pasta do passo 6>` | `dados/sicor_com_clima/` |
| 9 | `random_forest.py` | `resultados/random_forest/` |

Os passos 1 a 6 leem e gravam em `dados/sicor/`, com a mesma estrutura de pastas usada no Google Drive. Para usar outra pasta, defina `TCC_PASTA_DADOS` (veja `preparacao_sicor/comum.py`).

Scripts de exploração, independentes da sequência: `explorar_credito.py`, `analisar_colunas_interesse.py`, `analisar_colunas_programa.py` e `inspecionar_parquet.py`, todos em `preparacao_sicor/`.

O notebook `Explorando_Dados_Sicor.ipynb` é a versão original (Google Colab) dos passos 1 a 6 e dos scripts de exploração.
