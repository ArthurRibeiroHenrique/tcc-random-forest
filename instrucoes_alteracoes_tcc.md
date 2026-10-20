# Alterações a fazer no TCC (cruzamento SICOR x INMET)

## O que é este pedido

O arquivo `.docx` anexado a esta conversa é o meu TCC: "Otimização da Concessão de Crédito Rural em Cooperativas: um Modelo Preditivo de Inadimplência com Random Forest e Integração de Dados Climáticos e Financeiros". Ele usa dados de crédito rural do SICOR (Banco Central) e dados climáticos do INMET.

Desde a última revisão do texto, o cruzamento entre as duas bases foi implementado e executado em Python sobre a base completa. Parte do texto ainda descreve esse cruzamento como um plano e, em alguns pontos, contradiz o que foi feito ou o que os dados mostraram. Preciso que você aplique as seis alterações listadas abaixo, e somente elas.

Todos os números deste arquivo foram medidos na base real. Use-os exatamente como estão e não calcule, estime nem arredonde outros. Se alguma alteração exigir um número que não está aqui, deixe uma marcação `[CONFIRMAR]` no texto em vez de preencher.

## Como trabalhar

- Mantenha o estilo do documento: português formal, impessoal, verbos no passado para o que já foi feito, números no formato brasileiro (vírgula decimal, ponto de milhar).
- Escreva os nomes das variáveis exatamente como aparecem aqui, em maiúsculas e com os mesmos sublinhados (por exemplo, `MODALIDADE_Empreendimento`, `DESVIO_%`).
- Não altere nenhuma outra seção, as referências já existentes, a descrição da base sintética nem os resultados esperados do modelo.
- Onde eu forneço o texto pronto, use-o como está; ajuste apenas o que for necessário para encaixar na frase vizinha.
- Se, ao aplicar uma alteração, você notar outra parte do texto que passe a contradizê-la, não corrija por conta própria: liste no final, em "Pontos para o autor conferir".

Se você conseguir editar o `.docx`, devolva o arquivo alterado. Se não, devolva cada trecho alterado por inteiro, identificado pelo número da seção, pronto para colar. Nos dois casos, termine com uma lista curta do que foi mudado em cada seção.

## O que foi feito na prática

Este é o método real, que o texto precisa refletir:

1. As coordenadas da sede de cada município vieram da base Localidades 2022 do IBGE (5.571 municípios).
2. As coordenadas das estações vieram do catálogo de estações automáticas do INMET. Foram consideradas as 584 estações que têm dados no arquivo climático de 2019 a 2025.
3. A distância entre cada sede municipal e cada estação foi calculada pela fórmula de Haversine.
4. Cada distância recebeu uma faixa de qualidade: excelente (inferior a 30 km), boa (de 30 a 60 km), regular (acima de 60 e até 100 km) e sem cobertura (acima de 100 km).
5. Para cada município foram guardadas até três estações dentro do raio de 100 km. Quando a mais próxima não tem `PRECIP_90D` válida na data, usa-se a seguinte.
6. Cada operação do SICOR recebe as variáveis climáticas do dia anterior à data de emissão do contrato (`DT_EMISSAO`), para que o modelo só veja o que já tinha acontecido antes da assinatura.
7. A ligação entre a operação e a estação é feita pelo código do município (`CD_IBGE_MUNICIPIO`) e pela distância. Os nomes de município e UF do arquivo do INMET não participam do cruzamento; servem apenas para identificar a estação.

## Alteração 1: reescrever a seção 2.5.1

A seção "2.5.1 Seleção de Estação Meteorológica (INMET)" ainda está no futuro ("serão definidas"), não diz de onde vêm as coordenadas, não classifica as distâncias de exatamente 30 km e 100 km e não menciona as estações reserva nem o dia anterior. Mantenha o título e substitua todo o conteúdo (os dois parágrafos atuais) por este:

> Como nem todos os municípios possuem estação meteorológica, cada município foi associado à estação automática do INMET mais próxima. As coordenadas das sedes municipais foram obtidas da base Localidades 2022 do IBGE, e as das estações, do catálogo de estações automáticas do INMET, considerando as 584 estações com dados no período de 2019 a 2025. A distância foi calculada em Python pela fórmula de Haversine e classificada em quatro faixas: excelente (inferior a 30 km), boa (de 30 a 60 km), regular (acima de 60 e até 100 km) e sem cobertura (acima de 100 km). Para cada município foram mantidas até três estações dentro do raio de 100 km, de modo que, quando a mais próxima não possui PRECIP_90D válida na data (seção 2.5.2), utiliza-se a seguinte. As variáveis climáticas atribuídas a cada operação são as do dia anterior à emissão do contrato, para evitar vazamento de informação.

Acrescente às referências a entrada da base do IBGE, no mesmo padrão ABNT das demais. Dados disponíveis: autor institucional IBGE; título "Localidades do Brasil 2022"; endereço `https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/localidades/Localidades_do_Brasil/2022/`; acesso em 8 out. 2026. O ano de publicação não foi confirmado: marque com `[CONFIRMAR]`.

## Alteração 2: corrigir o filtro de agricultura na seção 2.5

O parágrafo que começa com "E, com o intuito de que essa análise dê certo" diz que a agricultura foi isolada com três condições: `FINALIDADE` = "Custeio", `ATIVIDADE` = "Agrícola" e `MODALIDADE_Empreendimento` = "LAVOURA". Na base completa, essa combinação não retorna nenhuma operação. As colunas `FINALIDADE` e `ATIVIDADE` não descrevem a operação individual: das 3.404.713 operações de lavoura, nenhuma aparece nelas como custeio agrícola. Já pelas colunas `FINALIDADE_Empreendimento` e `ATIVIDADE_Empreendimento`, todas as 3.404.713 são custeio e agrícola.

Substitua o parágrafo por:

> E, com o intuito de que essa análise dê certo, é necessário que sejam isolados os dados referentes à agricultura, exclusivamente. Para tanto, foi utilizada a variável cadastral MODALIDADE_Empreendimento com o valor "LAVOURA". Na base de 2019 a 2025, esse filtro seleciona 3.404.713 operações, todas elas de custeio e de atividade agrícola segundo as colunas FINALIDADE_Empreendimento e ATIVIDADE_Empreendimento. As colunas FINALIDADE e ATIVIDADE não foram utilizadas, pois não descrevem a operação individual: nenhuma das operações de lavoura aparece nelas como custeio agrícola.

No parágrafo seguinte (o que cita `CD_ESTADO` e `CD_IBGE_MUNICIPIO`), acrescente que o `CD_IBGE_MUNICIPIO` é também a chave usada para associar cada operação à estação meteorológica (seção 2.5.1).

## Alteração 3: ajustar a Tabela 1 e a contagem de variáveis

Em consequência da alteração 2:

- Na Tabela 1, remova as linhas `FINALIDADE` e `ATIVIDADE`. A tabela fica com seis variáveis.
- Ainda na Tabela 1, troque `VL_AREA_FINANCIADA` por `VL_AREA_FINANC`, que é o nome real da coluna (o texto acima da tabela já usa o nome correto).
- Na seção 2.5.3, no item "Retenção", troque "Mantidas as 8 variáveis do SICOR" por "Mantidas as 6 variáveis do SICOR".

## Alteração 4: corrigir a linha "MUNICIPIO e UF" da Tabela 2

A Tabela 2 diz que `MUNICIPIO` e `UF` são "usados no cruzamento com o SICOR". Isso não é verdade (ver item 7 do método). Troque a linha para:

| Variável | Uso no modelo | Descrição |
|---|---|---|
| MUNICIPIO e UF | Identificação | Município e sigla do estado da estação; identificam a estação associada a cada operação (seção 2.5.1) |

## Alteração 5: esclarecer os percentuais de nulos na seção 2.5.2

O parágrafo sobre dados faltantes informa que a `PRECIP_90D` tem 31,58% de valores nulos e o `DESVIO_%` tem 32,51%. Esses percentuais estão corretos, mas são medidos no arquivo climático, que tem uma linha por estação e por dia. Depois do cruzamento, o efeito sobre as operações é bem menor, por causa das estações reserva. Logo depois da frase que traz esses dois percentuais, acrescente:

> Esses percentuais referem-se ao arquivo climático, com uma linha por estação e por dia. Após o cruzamento com o SICOR, que recorre a estações reserva (seção 2.5.1), o DESVIO_% fica nulo em no máximo 3,3% das operações que receberam variáveis climáticas em cada ano.

## Alteração 6: incluir os resultados do cruzamento na seção 3

A seção "3 Resultados e Discussões" hoje só traz resultados esperados. Insira, como primeiros parágrafos da seção, antes do texto que já existe, os dois parágrafos e a tabela abaixo. Numere a tabela na sequência do documento (deve ser a Tabela 3) e use a mesma legenda de fonte das outras, "Fonte: o autor (2026).".

> A associação entre municípios e estações cobriu a maior parte do território: dos 5.571 municípios, 41,5% ficaram na faixa excelente, 45,4% na boa e 11,5% na regular, com distância mediana de 34,4 km até a estação mais próxima. Apenas 88 municípios (1,6%) ficaram sem cobertura, principalmente no Amazonas, em Rondônia, em Roraima e em Mato Grosso.
>
> O cruzamento foi executado sobre as 14.667.090 operações do SICOR registradas de 2019 a 2025, das quais 68,9% receberam variáveis climáticas. Entre as 3.404.713 operações de lavoura, esse percentual foi de 67,2% e chega a 95,1% quando se consideram apenas as que possuem código de município. A estação mais próxima foi utilizada em 74,2% das operações cruzadas; nas demais 25,8%, recorreu-se à segunda ou à terceira estação, em razão das falhas de leitura descritas na seção 2.5.2. A principal perda não decorre dos dados climáticos, mas da ausência do código de município em 20,2% das operações da base enriquecida do SICOR, limitação cuja causa ainda não foi investigada. A Tabela 3 detalha a distribuição.

Tabela 3 – Situação das operações do SICOR após o cruzamento com o INMET (2019 a 2025)

| Situação | Todas as operações | Operações de lavoura |
|---|---|---|
| Excelente (inferior a 30 km) | 20,6% | 25,5% |
| Boa (de 30 a 60 km) | 30,4% | 32,8% |
| Regular (acima de 60 e até 100 km) | 17,8% | 8,9% |
| Sem dado válido na data | 10,0% | 3,4% |
| Sem cobertura (acima de 100 km) | 0,9% | 0,1% |
| Sem código de município | 20,2% | 29,3% |
| Total de operações | 14.667.090 | 3.404.713 |

Não proponha uma causa para a ausência do código de município: ela ainda não foi apurada.

## Números de referência

Para conferência; todos já aparecem nos textos acima.

| Item | Valor |
|---|---|
| Municípios (IBGE, Localidades 2022) | 5.571 |
| Estações do INMET com dados (2019 a 2025) | 584 |
| Municípios por faixa (estação mais próxima) | excelente 2.314 (41,5%); boa 2.530 (45,4%); regular 639 (11,5%); sem cobertura 88 (1,6%) |
| Distância mediana até a estação mais próxima | 34,4 km |
| Operações do SICOR, 2019 a 2025 | 14.667.090 |
| Operações que receberam clima | 10.105.739 (68,9%) |
| Operações sem código de município | 2.966.560 (20,2%) |
| Operações de lavoura | 3.404.713 |
| Operações de lavoura que receberam clima | 2.288.457 (67,2%; 95,1% das que têm município) |
| Estação utilizada nas operações cruzadas | 1ª: 74,2%; 2ª: 19,4%; 3ª: 6,4% |
| `DESVIO_%` nulo entre as operações cruzadas, por ano | 2019: 3,3%; 2020: 2,6%; 2021: 0,2%; 2022: 0,7%; 2023: 0,3%; 2024: 0,0%; 2025: 0,1% |
