from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

# Configuração inicial
api_key = os.getenv("secret_key")
llm = Groq(model="llama3-70b-8192", api_key=api_key)

# Estilização da interface
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

# Função para descrição das colunas
def descricao_colunas(df):
    desc = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Detalhes das colunas do dataframe:\n" + desc

# Configuração do pipeline de consulta
def pipeline_consulta(df):
    instruction_str = (
        "1. Converta a consulta para código Python executável usando Pandas.\n"
        "2. A linha final do código deve ser uma expressão Python que possa ser chamada com `eval()`.\n"
        "3. O código deve representar uma solução para a consulta.\n"
        "4. IMPRIMA APENAS A EXPRESSÃO.\n"
        "5. Não coloque a expressão entre aspas.\n"
    )

    pandas_prompt_str = (
        "Você está trabalhando com um dataframe do pandas chamado `df`.\n"
        "{colunas_detalhes}\n\n"
        "Resultado de `print(df.head())`:\n"
        "{df_str}\n\n"
        "Instruções:\n"
        "{instruction_str}\n"
        "Consulta: {query_str}\n\n"
        "Expressão:"
    )

    response_synthesis_prompt_str = (
        "Dada uma pergunta, atue como analista de dados e elabore uma resposta clara e concisa.\n"
        "Consulta: {query_str}\n\n"
        "Instruções Pandas:\n{pandas_instructions}\n\n"
        "Saída Pandas: {pandas_output}\n\n"
        "Resposta:\n\n"
        "Código utilizado: `{pandas_instructions}`"
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str,
        df_str=df.head(5).to_string(),
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

# Função para carregar dados (agora suporta CSV e Excel)
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
        qp = pipeline_consulta(df_estado)
        resposta = qp.run(query_str=pergunta)
        return resposta.message.content
    return "Por favor, carregue um arquivo e faça uma pergunta."

# Função para adicionar ao histórico
def add_historico(pergunta, resposta, historico_estado):
    if pergunta and resposta:
        historico_estado.append((pergunta, resposta))
        return historico_estado, gr.Info("Adicionado ao histórico do PDF!")
    return historico_estado, gr.Info("Nenhuma pergunta/resposta para adicionar.")

# Função para gerar PDF com layout aprimorado
def gerar_pdf(historico_estado):
    if not historico_estado:
        return None, "Nenhum dado para gerar o PDF."

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    caminho_pdf = f"relatorio_analise_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    
    # Adicionar título
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório de Análise de Dados", ln=True, align='C')
    pdf.ln(10)
    
    # Adicionar perguntas e respostas
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
    return "", ""

# Função para resetar
def resetar_aplicacao():
    return None, "Aplicação resetada. Faça upload de um novo arquivo.", pd.DataFrame(), "", None, [], ""

# Interface Gradio aprimorada
with gr.Blocks(theme=THEME, css="""
    .gr-button {margin: 5px;}
    .gr-textbox {border-radius: 5px;}
    #title {text-align: center; padding: 20px;}
""") as app:
    
    gr.Markdown(
        "# Análise Inteligente de Dados 📊",
        elem_id="title"
    )
    
    gr.Markdown("""
        Faça upload de um arquivo CSV ou Excel e explore seus dados com perguntas em linguagem natural.
        Salve suas análises em um PDF profissional com um clique!
    """)

    with gr.Tabs():
        with gr.Tab("Carregar e Visualizar"):
            input_arquivo = gr.File(
                file_types=[".csv", ".xlsx", ".xls"],
                label="Upload de Arquivo (CSV ou Excel)"
            )
            upload_status = gr.Textbox(label="Status")
            tabela_dados = gr.DataFrame(label="Visualização dos Dados", interactive=False)

        with gr.Tab("Análise"):
            gr.Markdown("### Faça sua pergunta")
            input_pergunta = gr.Textbox(
                label="Pergunta",
                placeholder="Ex: Qual é a média das vendas por região?"
            )
            botao_submeter = gr.Button("Analisar", variant="primary")
            output_resposta = gr.Textbox(label="Resposta", lines=10)

            with gr.Row():
                botao_limpeza = gr.Button("Limpar")
                botao_add_pdf = gr.Button("Adicionar ao Histórico")
                botao_gerar_pdf = gr.Button("Gerar PDF")

            arquivo_pdf = gr.File(label="Download do PDF")

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
        outputs=output_resposta
    )
    botao_limpeza.click(
        fn=limpar_pergunta_resposta,
        inputs=[],
        outputs=[input_pergunta, output_resposta]
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
        outputs=[input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta]
    )

if __name__ == "__main__":
    app.launch()