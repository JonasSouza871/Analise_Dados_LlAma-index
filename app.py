from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import base64

# Configuração inicial
api_key = os.getenv("secret_key")
llm = Groq(model="llama3-70b-8192", api_key=api_key)

# Tema personalizado para a interface
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="gray",
    neutral_hue="slate",
).set(
    body_background_fill="white",
    body_background_fill_dark="#1a1a1a",
    block_background_fill="#f9fafb",
    block_background_fill_dark="#2d2d2d",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_200",
    button_secondary_text_color="*neutral_800",
)

# Função para descrição das colunas
def descricao_colunas(df):
    desc = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Detalhes das colunas do dataframe:\n" + desc

# Função para plotar gráficos
def plotar_grafico(df, tipo_grafico, x_col=None, y_col=None, hue_col=None, title="Gráfico"):
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    if tipo_grafico == "bar":
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
    elif tipo_grafico == "scatter":
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, size=hue_col)
    elif tipo_grafico == "line":
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
    elif tipo_grafico == "box":
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
    else:
        plt.close()
        return None, "Tipo de gráfico não suportado. Use 'bar', 'scatter', 'line' ou 'box'."

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    
    # Salvar o gráfico em um buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    
    # Converter a imagem para base64 para exibir no Gradio
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}", None

# Configuração do pipeline de consulta
def pipeline_consulta(df):
    instruction_str = (
        "1. Converta a consulta para código Python executável usando Pandas.\n"
        "2. A linha final do código deve ser uma expressão Python que possa ser chamada com `eval()`.\n"
        "3. O código deve representar uma solução para a consulta.\n"
        "4. IMPRIMA APENAS A EXPRESSÃO.\n"
        "5. Não coloque a expressão entre aspas.\n"
        "6. Se a consulta pedir um gráfico, retorne a expressão para calcular os dados necessários e mencione o tipo de gráfico (ex.: 'bar', 'scatter', 'line', 'box') e as colunas envolvidas (ex.: x='col1', y='col2', hue='col3').\n"
    )

    pandas_prompt_str = (
        "Você está trabalhando com um dataframe do pandas chamado `df`.\n"
        "{colunas_detalhes}\n\n"
        "Resultado de `print(df.head())`:\n"
        "{df_str}\n\n"
        "Siga estas instruções:\n"
        "{instruction_str}\n"
        "Consulta: {query_str}\n\n"
        "Expressão:"
    )

    response_synthesis_prompt_str = (
        "Dada uma pergunta de entrada, atue como analista de dados e elabore uma resposta a partir dos resultados da consulta.\n"
        "Responda de forma natural, sem introduções como 'A resposta é:' ou algo semelhante.\n"
        "Consulta: {query_str}\n\n"
        "Instruções do Pandas (opcional):\n{pandas_instructions}\n\n"
        "Saída do Pandas: {pandas_output}\n\n"
        "Resposta: \n\n"
        "Se a consulta pedir um gráfico, inclua na resposta as informações do gráfico (tipo, colunas usadas) no formato: 'Gráfico: tipo={tipo}, x={x}, y={y}, hue={hue}'.\n"
        "Ao final, exibir o código usado para gerar a resposta, no formato: O código utilizado foi `{pandas_instructions}`"
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str,
        df_str=df.head(5),
        colunas_detalhes=descricao_colunas(df)
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links([
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
        Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
    ])
    qp.add_link("response_synthesis_prompt", "llm2")
    return qp

# Função para carregar dados
def carregar_dados(caminho_arquivo, df_estado):
    if not caminho_arquivo:
        return "Por favor, faça o upload de um arquivo CSV ou Excel.", pd.DataFrame(), df_estado
    
    try:
        ext = os.path.splitext(caminho_arquivo)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(caminho_arquivo)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(caminho_arquivo)
        else:
            return "Formato de arquivo não suportado. Use CSV ou Excel.", pd.DataFrame(), df_estado
        return "Arquivo carregado com sucesso!", df.head(), df
    except Exception as e:
        return f"Erro ao carregar arquivo: {str(e)}", pd.DataFrame(), df_estado

# Função para processar pergunta
def processar_pergunta(pergunta, df_estado):
    if df_estado is not None and pergunta:
        try:
            qp = pipeline_consulta(df_estado)
            resposta = qp.run(query_str=pergunta)
            resposta_texto = resposta.message.content

            # Verificar se a resposta menciona um gráfico
            if "Gráfico:" in resposta_texto:
                # Extrair informações do gráfico
                import re
                grafico_info = re.search(r"Gráfico: tipo=(\w+), x=(\w+), y=(\w+), hue=(\w+)", resposta_texto)
                if grafico_info:
                    tipo, x_col, y_col, hue_col = grafico_info.groups()
                    # Se hue for "None", ajustamos para None
                    hue_col = None if hue_col == "None" else hue_col
                    # Gerar o gráfico
                    img_data, erro = plotar_grafico(df_estado, tipo, x_col, y_col, hue_col, title=pergunta)
                    if erro:
                        return f"Erro ao gerar gráfico: {erro}", None
                    return resposta_texto, img_data
            return resposta_texto, None
        except Exception as e:
            return f"Erro no pipeline: {str(e)}", None
    return "Por favor, carregue um arquivo e faça uma pergunta.", None

# Função para adicionar ao histórico
def add_historico(pergunta, resposta, historico_estado):
    if pergunta and resposta:
        historico_estado.append((pergunta, resposta))
        return historico_estado, gr.Info("Adicionado ao histórico do PDF!")
    return historico_estado, gr.Info("Nenhuma pergunta/resposta para adicionar.")

# Função para gerar PDF
def gerar_pdf(historico_estado):
    if not historico_estado:
        return None, "Nenhum dado para gerar o PDF."

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    caminho_pdf = f"relatorio_analise_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório de Análise de Dados", ln=True, align='C')
    pdf.ln(10)
    
    for i, (pergunta, resposta) in enumerate(historico_estado, 1):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Pergunta {i}: {pergunta}", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 6, resposta)
        pdf.ln(5)

    pdf.output(caminho_pdf)
    return caminho_pdf, "PDF gerado com sucesso!"

# Função para limpar
def limpar_pergunta_resposta():
    return "", "", None

# Função para resetar
def resetar_aplicacao():
    return None, "Aplicação resetada. Faça upload de um novo arquivo.", pd.DataFrame(), "", None, [], "", None

# Interface Gradio aprimorada
with gr.Blocks(theme=THEME, css="""
    .gr-button {margin: 5px;}
    .gr-textbox {border-radius: 5px;}
    #title {text-align: center; padding: 20px; color: #1e3a8a;}
    #subtitle {text-align: center; color: #4b5563; margin-bottom: 20px;}
    .gr-box {border-radius: 10px; padding: 15px;}
""") as app:
    
    gr.Markdown(
        "# DataInsight Pro 📈🔍",
        elem_id="title"
    )
    gr.Markdown(
        "Explore seus dados com perguntas em linguagem natural e visualize os resultados com gráficos interativos. Salve suas análises em um PDF profissional!",
        elem_id="subtitle"
    )

    with gr.Tabs():
        # Aba para carregar dados
        with gr.Tab("Carregar Dados"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_arquivo = gr.File(
                        file_types=[".csv", ".xlsx", ".xls"],
                        label="Upload de Arquivo (CSV ou Excel)"
                    )
                    upload_status = gr.Textbox(label="Status do Upload", interactive=False)
                with gr.Column(scale=2):
                    tabela_dados = gr.DataFrame(label="Prévia dos Dados", interactive=False)

        # Aba para análise
        with gr.Tab("Análise de Dados"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Faça sua Pergunta")
                    input_pergunta = gr.Textbox(
                        label="Pergunta",
                        placeholder="Ex: 'Qual a média de vendas por região?' ou 'Plote um gráfico de barras de vendas por região'",
                        lines=2
                    )
                    botao_submeter = gr.Button("Analisar", variant="primary")
                    gr.Markdown("""
                        **Dicas para Perguntas:**
                        - Use linguagem natural, como "Qual a média de vendas por região?"
                        - Para gráficos, peça explicitamente, como "Plote um gráfico de barras de vendas por região"
                        - Tipos de gráficos suportados: bar, scatter, line, box
                    """)
                with gr.Column(scale=2):
                    gr.Markdown("### Resposta")
                    output_resposta = gr.Textbox(label="Resposta", lines=5, interactive=False)
                    output_grafico = gr.Image(label="Gráfico", type="filepath")

            with gr.Row():
                botao_limpeza = gr.Button("Limpar", variant="secondary")
                botao_add_pdf = gr.Button("Adicionar ao Histórico", variant="secondary")
                botao_gerar_pdf = gr.Button("Gerar PDF", variant="primary")

            arquivo_pdf = gr.File(label="Download do PDF")

    # Botão de reset
    botao_resetar = gr.Button("Novo Conjunto de Dados", variant="secondary")

    # Estados
    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])

    # Conexões
    input_arquivo.change(
        fn=carregar_dados,
        inputs=[input_arquivo, df_estado],
        outputs=[upload_status, tabela_dados, df_estado]
    )
    botao_submeter.click(
        fn=processar_pergunta,
        inputs=[input_pergunta, df_estado],
        outputs=[output_resposta, output_grafico]
    )
    botao_limpeza.click(
        fn=limpar_pergunta_resposta,
        inputs=[],
        outputs=[input_pergunta, output_resposta, output_grafico]
    )
    botao_add_pdf.click(
        fn=add_historico,
        inputs=[input_pergunta, output_resposta, historico_estado],
        outputs=[historico_estado]
    )
    botao_gerar_pdf.click(
        fn=gerar_pdf,
        inputs=[historico_estado],
        outputs=[arquivo_pdf]
    )
    botao_resetar.click(
        fn=resetar_aplicacao,
        inputs=[],
        outputs=[input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta, output_grafico]
    )

if __name__ == "__main__":
    app.launch()