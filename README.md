# Análise Sphera

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
- Modelos preditivos - preparação

Aplicação original publicada em:

https://analysissphera.streamlit.app/
