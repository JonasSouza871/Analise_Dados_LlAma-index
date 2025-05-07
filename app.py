"""
Dash de Perguntas a CSV/Excel ✨
--------------------------------
• Lê CSV, XLSX/XLS, ODS
• Consulta via LlamaIndex (Groq)
• Histórico → PDF
"""

from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

# ⏬  Config ----------------------------------------------------------
LLM_MODEL = "llama3-70b-8192"
API_KEY   = os.getenv("secret_key")          # defina no ambiente

llm = Groq(model=LLM_MODEL, api_key=API_KEY)


# 📊  Helpers ---------------------------------------------------------
def describe_cols(df: pd.DataFrame) -> str:
    col_info = "\n".join(f"`{c}`: {t}" for c, t in zip(df.columns, df.dtypes))
    return "Colunas do dataframe:\n" + col_info


def build_pipeline(df: pd.DataFrame) -> QP:
    instruction_str = (
        "1. Converta a consulta para código Python executável usando Pandas.\n"
        "2. A linha final deve ser uma **expressão**, sem `print`, capaz de ser avaliada por `eval()`.\n"
        "3. Não inclua aspas ao redor da expressão.\n"
    )
    pandas_prompt_tmpl = PromptTemplate(
        """Você trabalha com um dataframe Pandas chamado `df`.
{colunas}
Primeiras linhas:
{df_head}

{instruction}
Pergunta: {query_str}

Expressão:"""
    ).partial_format(
        colunas=describe_cols(df),
        df_head=df.head(),
        instruction=instruction_str,
    )

    response_prompt_tmpl = PromptTemplate(
        """Responda como analista de dados.
Pergunta: {query_str}

Código Pandas gerado: {pandas_instructions}

Saída: {pandas_output}

Resposta:

O código utilizado foi `{pandas_instructions}`"""
    )

    pipeline = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt_tmpl,
            "llm1": llm,
            "parser": PandasInstructionParser(df),
            "resp_prompt": response_prompt_tmpl,
            "llm2": llm,
        },
        verbose=False,
    )
    pipeline.add_chain(["input", "pandas_prompt", "llm1", "parser"])
    pipeline.add_links(
        [
            Link("input", "resp_prompt", dest_key="query_str"),
            Link("llm1", "resp_prompt", dest_key="pandas_instructions"),
            Link("parser", "resp_prompt", dest_key="pandas_output"),
        ]
    )
    pipeline.add_link("resp_prompt", "llm2")
    return pipeline


# 📂  Upload ----------------------------------------------------------
def load_file(file_path, df_state):
    if not file_path:
        return "Faça upload de um CSV ou Excel.", pd.DataFrame(), df_state, ""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".csv", ".txt"}:
            df = pd.read_csv(file_path)
        elif ext in {".xlsx", ".xls", ".ods"}:
            # Se houver várias sheets, pega a primeira por padrão
            with pd.ExcelFile(file_path) as xls:
                sheet = xls.sheet_names[0]
                df = pd.read_excel(xls, sheet_name=sheet)
        else:
            return "Formato não suportado.", pd.DataFrame(), df_state, ""

        status = f"✔️ Arquivo carregado! Shape: {df.shape[0]} linhas × {df.shape[1]} colunas."
        return status, df.head(15), df, ""
    except Exception as e:
        return f"Erro ao carregar: {e}", pd.DataFrame(), df_state, ""


# ❓  Pergunta ---------------------------------------------------------
def ask_query(question, df_state):
    if df_state is None or df_state.empty:
        return "Faça upload do dataset primeiro."
    if not question.strip():
        return "Digite uma pergunta."
    pipeline = build_pipeline(df_state)
    result = pipeline.run(query_str=question)
    return result.message.content.strip()


# 📝  Histórico / PDF --------------------------------------------------
def add_to_history(q, a, hist):
    if q and a:
        hist.append((q, a))
        return hist
    return hist


def make_pdf(hist):
    if not hist:
        return None, "Nenhum item no histórico."
    name = f"relatorio_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    for q, a in hist:
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, q)
        pdf.ln(1)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, a)
        pdf.ln(4)
    pdf.output(name)
    return name, "PDF gerado!"

def reset_app():
    return (
        None,               # file upload
        "",                 # status
        pd.DataFrame(),     # preview
        "",                 # resposta
        None,               # pdf
        [],                 # histórico
        ""                  # pergunta
    )


# 🎨  Interface -------------------------------------------------------
soft = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(theme=soft, title="Analisador de Dados") as demo:
    gr.HTML("<h1 style='text-align:center'>🔎 Analisador de Dados CSV/Excel</h1>")
    estado_df = gr.State()
    estado_hist = gr.State([])

    with gr.Tabs():
        # TAB 1 – Dados
        with gr.TabItem("📂 Dados"):
            arquivo = gr.File(label="Upload CSV ou Excel", file_count="single", type="filepath")
            status = gr.Markdown()
            preview = gr.DataFrame(interactive=True, max_rows=15)
            arquivo.change(load_file, [arquivo, estado_df],
                           [status, preview, estado_df, preview])
        # TAB 2 – Perguntas
        with gr.TabItem("❓ Perguntas"):
            pergunta = gr.Textbox(label="Pergunta")
            btn_enviar = gr.Button("Enviar consulta", variant="primary", icon="🚀")
            resposta = gr.Markdown()
            btn_enviar.click(ask_query, [pergunta, estado_df], resposta)

            gr.HorizontalRule()
            with gr.Row():
                btn_clear = gr.Button("Limpar", icon="🧹")
                btn_add_hist = gr.Button("Adicionar ao histórico", icon="➕")
            btn_clear.click(lambda: ("", ""), None, [pergunta, resposta])
            btn_add_hist.click(add_to_history, [pergunta, resposta, estado_hist], estado_hist)

        # TAB 3 – Histórico / PDF
        with gr.TabItem("📝 Histórico / PDF"):
            hist_comp = gr.HighlightedText(label="Histórico (Q → A)")
            btn_pdf = gr.Button("Gerar PDF", icon="📄")
            output_pdf = gr.File()
            btn_pdf.click(make_pdf, estado_hist, [output_pdf, status])
            estado_hist.change(lambda h: {"text": "\n\n".join(f"➡️ {q}\n{a}" for q, a in h)}, estado_hist, hist_comp)

    gr.Button("🔄 Reiniciar aplicação", variant="stop", icon="♻️").click(
        reset_app,
        None,
        [arquivo, status, preview, resposta, output_pdf, estado_hist, pergunta],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch()
