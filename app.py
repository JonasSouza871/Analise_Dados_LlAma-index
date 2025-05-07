# ========================== DEPENDÊNCIAS ==========================
# Execute no Colab:
# !pip install llama-index llama-index-llms-groq gradio pandas openpyxl fpdf matplotlib seaborn

from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile, os
from datetime import datetime
import base64
import io
import re

# ========================== LLM (Groq) ============================
os.environ["secret_key"] = os.getenv("secret_key", "")
llm = Groq(model="llama3-70b-8192", api_key=os.environ["secret_key"])

# ========================== UTILIDADES ============================
def descricao_colunas(df):
    return "Detalhes das colunas do dataframe:\n" + "\n".join(f"`{c}`: {t}" for c, t in zip(df.columns, df.dtypes))

def plotar_grafico(df, tipo, x_col=None, y_col=None, hue_col=None):
    """Função para plotar gráficos com base nos parâmetros"""
    plt.figure(figsize=(8, 6))

    if tipo == "bar":
        if hue_col:
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
        else:
            sns.barplot(data=df, x=x_col, y=y_col)
    elif tipo == "line":
        if hue_col:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
        else:
            sns.lineplot(data=df, x=x_col, y=y_col)
    elif tipo == "scatter":
        if hue_col:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col)
    elif tipo == "hist":
        sns.histplot(data=df, x=y_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col)

    plt.title(f"{tipo} de {y_col} por {x_col}")
    plt.tight_layout()

    # Salvar gráfico em buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close()

    # Converter para base64 para exibição no Gradio
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def pipeline_consulta(df):
    instruction_str = (
        "1. Analise a consulta do usuário.\n"
        "2. Se a consulta NÃO pedir um gráfico:\n"
        "   a. Converta a consulta para uma ÚNICA linha de código Python executável usando Pandas.\n"
        "   b. A linha final DEVE ser uma expressão Python que possa ser chamada com `eval()`.\n"
        "   c. IMPRIMA APENAS ESTA EXPRESSÃO. Não adicione comentários Python (#) nem texto explicativo.\n"
        "3. Se a consulta PEDIR um gráfico:\n"
        "   a. Inclua a linha 'Gráfico: tipo={tipo}, x={coluna_x}, y={coluna_y}' antes do código\n"
        "   b. O código deve gerar os dados necessários para o gráfico\n"
        "4. Certifique-se que os nomes das colunas correspondem EXATAMENTE aos do dataframe.\n"
    )

    pandas_prompt = PromptTemplate(
        "Você está trabalhando com um dataframe do pandas chamado `df`.\n"
        "{colunas}\n\nprint(df.head()):\n{head}\n\n{instr}\nConsulta: {query}\n\nResposta:"
    ).partial_format(
        colunas=descricao_colunas(df),
        head=df.head(5),
        instr=instruction_str,
    )

    rsp_prompt = PromptTemplate(
        "Dada uma pergunta, atue como analista de dados e elabore uma resposta clara.\n"
        "Consulta: {query}\n\nCódigo Pandas:\n{pandas_instructions}\n\nSaída:\n{pandas_output}\n\n"
        "Resposta (cite o código usado no final)."
    )

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": PandasInstructionParser(df),
            "rsp_prompt": rsp_prompt,
            "llm2": llm,
        },
        verbose=False,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links([
        Link("input", "rsp_prompt", dest_key="query"),
        Link("llm1", "rsp_prompt", dest_key="pandas_instructions"),
        Link("pandas_output_parser", "rsp_prompt", dest_key="pandas_output"),
    ])
    qp.add_link("rsp_prompt", "llm2")
    return qp

# ========================== CALLBACKS =============================
def carregar_dados(fp, df_state):
    if not fp:
        return "Faça upload de CSV/Excel.", pd.DataFrame(), df_state
    try:
        ext = os.path.splitext(fp)[1].lower()
        df = pd.read_csv(fp) if ext == ".csv" else pd.read_excel(fp)
        return "Arquivo carregado!", df.head(), df
    except Exception as e:
        return f"Erro: {e}", pd.DataFrame(), df_state

def processar_pergunta(q, df_state):
    if df_state is None or q.strip() == "":
        return "Carregue um arquivo e faça uma pergunta.", None, None

    try:
        # Executar o pipeline
        ans = pipeline_consulta(df_state).run(query=q)
        resposta = ans.message.content

        img_data = None

        # Verificar se a resposta contém solicitação de gráfico
        grafico_match = re.search(r"Gráfico:\s*tipo=([\w]+),\s*x=([\w]+),\s*y=([\w]+)", resposta)
        if grafico_match:
            tipo = grafico_match.group(1)
            x_col = grafico_match.group(2)
            y_col = grafico_match.group(3)

            # Extrair o código Python da resposta
            codigo_match = re.search(r"`([^`]+)`", resposta)
            if codigo_match:
                codigo = codigo_match.group(1)
                # Executar o código para obter os dados
                dados = eval(codigo, {"df": df_state, "pd": pd})

                # Plotar o gráfico
                img_data = plotar_grafico(dados, tipo, x_col, y_col)

        return resposta, img_data, img_data
    except Exception as e:
        return f"Erro no pipeline: {e}", None, None

def add_historico(perg, resp, img_path, hist):
    hist.append((perg, resp, img_path))
    return hist

def gerar_pdf(hist):
    if not hist:
        return None, "Histórico vazio."

    try:
        data_atual = datetime.now().strftime("%d-%m-%Y")
        nome_pdf = f"relatorio_{data_atual}.pdf"
        caminho_pdf = os.path.join(tempfile.gettempdir(), nome_pdf)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Relatório de Análise de Dados", ln=True, align="C")
        pdf.ln(8)

        for i, (p, r, img) in enumerate(hist, 1):
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 7, f"Pergunta {i}: {p}")
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 6, r)

            if img and isinstance(img, str) and img.startswith("data:image/png;base64,"):
                # Decodificar imagem base64 e salvar temporariamente
                img_data = base64.b64decode(img.split(",")[1])
                temp_img = os.path.join(tempfile.gettempdir(), f"temp_img_{i}.png")
                with open(temp_img, "wb") as f:
                    f.write(img_data)

                # Adicionar imagem ao PDF
                pdf.ln(2)
                pdf.image(temp_img, w=160)
                pdf.ln(4)

                # Remover arquivo temporário
                os.remove(temp_img)

        pdf.output(caminho_pdf)
        return caminho_pdf, "PDF gerado com sucesso!"
    except Exception as e:
        return None, f"Erro ao gerar o PDF: {e}"

def limpar():
    return "", "", None

def reset():
    return None, "Upload novo.", pd.DataFrame(), "", None, [], "", None

# ========================== INTERFACE GRADIO =======================
with gr.Blocks(theme="Soft") as app:
    gr.Markdown("# Analisando os dados 🔎🎲")

    with gr.Tab("Carregar Dados"):
        with gr.Row():
            f_upload = gr.File(label="Upload CSV/Excel", type="filepath")
            up_status = gr.Textbox(label="Status upload")
            df_head = gr.DataFrame(label="Pré-visualização dos dados")

    with gr.Tab("Análise de Dados"):
        with gr.Row():
            pergunta = gr.Textbox(label="Pergunta", placeholder="Ex: 'Qual a média de vendas?' ou 'Plote um gráfico de barras de vendas por região'")
            btn_send = gr.Button("Enviar")

        with gr.Row():
            resp = gr.Textbox(label="Resposta")
            grafico = gr.Image(label="Gráfico Gerado")

        with gr.Row():
            btn_clr = gr.Button("Limpar pergunta e resultado")
            btn_hist = gr.Button("Adicionar ao histórico do PDF")
            btn_pdf = gr.Button("Gerar PDF")

    pdf_file = gr.File(label="Download do PDF")
    pdf_status = gr.Textbox(label="Status do PDF")
    btn_reset = gr.Button("Quero analisar outro dataset!")

    df_state = gr.State(None)
    hist_state = gr.State([])
    img_state = gr.State(None)

    f_upload.change(carregar_dados, [f_upload, df_state], [up_status, df_head, df_state])
    btn_send.click(processar_pergunta, [pergunta, df_state], [resp, grafico, img_state])
    btn_clr.click(limpar, [], [pergunta, resp, grafico])
    btn_hist.click(add_historico, [pergunta, resp, img_state, hist_state], [hist_state])
    btn_pdf.click(gerar_pdf, [hist_state], [pdf_file, pdf_status])
    btn_reset.click(reset, [], [f_upload, up_status, df_head, resp, pdf_file, hist_state, pergunta, grafico])

if __name__ == "__main__":
    app.launch()
