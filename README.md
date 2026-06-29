# Análise Sphera
https://sphera-novo.streamlit.app/

Aplicação Streamlit para análise da base de eventos offshore Sphera com a nova planilha `data/Sphera.xlsx`.

## Base de dados

A aplicação lê a aba `Ontologia_Eventos` da planilha local:

```text
data/Sphera.xlsx
```

Também usa a aba `Variaveis_Utilidade` para apoiar a visão de qualidade dos dados.

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run analise_eventos_sphera.py
```

Senha local padrão:

```text
cdshell
```

Para alterar a senha no Streamlit Cloud, configure o segredo `APP_PASSWORD`.

## Dashboards

- Visão Executiva da Ontologia
- FPI/SIF e Severidade Potencial
- Sinais Fracos, Precursores e Incidentes
- Barreiras Críticas e Estado das Barreiras
- Cenários, Mecanismos e Pareto 80/20
- Curadoria e Qualidade da Classificação
- Qualidade dos Dados
- Modelos preditivos

## Modelos preditivos

A página de modelos permite treinar baselines de classificação para:

- Potencial FPI/SIF
- Tipo Ontológico
- Cenário Acidental
- Barreira Crítica
- Incidente Futuro
- Potential Severity - Pessoas
- Tipo de Dano FPI/SIF
- Escopo de Risco

Cada modelo aparece em uma aba própria, com explicação do objetivo, pergunta respondida, variável-alvo, uso esperado e cuidados de interpretação. O treino é independente por modelo e mostra métricas de teste, matriz de confusão, relatório por classe, variáveis influentes e simulação com eventos existentes.

Aplicação original publicada em:

https://analysissphera.streamlit.app/
