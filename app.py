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

# Função para descrição das colunas
def descricao_colunas(df):
    desc = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Detalhes das colunas do dataframe:\n" + desc

# Configuração do pipeline de consulta
def pipeline_consulta(df):
    print("Colunas do DataFrame:", df.columns)  # Depuração
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
        df_str=df.head(5),  # Revertido para o original
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
    print("Caminho do arquivo recebido:", caminho_arquivo)  # Depuração
    if not caminho_arquivo:
        return "Por favor, faça o upload de um arquivo CSV ou Excel.", pd.DataFrame(), df_estado
    
    try:
        ext = os.path.splitext(caminho_arquivo)[1].lower()
        print("Extensão do arquivo:", ext)  # Depuração
        if ext == '.csv':
            df = pd.read_csv(caminho_arquivo)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(caminho_arquivo)
        else:
            return "Formato de arquivo não suportado. Use CSV ou Excel.", pd.DataFrame(), df_estado
        
        print("DataFrame carregado:", df.head())  # Depuração
        return "Arquivo carregado com sucesso!", df.head(), df
    except Exception as e:
        print("Erro ao carregar arquivo:", str(e))  # Depuração
        return f"Erro ao carregar arquivo: {str(e)}", pd.DataFrame(), df_estado

# Função para processar pergunta
def processar_pergunta(pergunta, df_estado):
    print("Pergunta recebida:", pergunta)  # Depuração
    print("Estado do DataFrame:", df_estado)  # Depuração
    if df_estado is not None and pergunta:
        try:
            qp = pipeline_consulta(df_estado)
            print("Pipeline criado com sucesso")  # Depuração
            resposta = qp.run(query_str=pergunta)
            print("Resposta do pipeline:", resposta)  # Depuração
            return resposta.message.content
        except Exception as e:
            print("Erro no pipeline:", str(e))  # Depuração
            return f"Erro no pipeline: {str(e)}"
    return "Por favor, carregue um arquivo e faça uma pergunta."

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
    return "", ""

# Função para resetar
def resetar_aplicacao():
    return None, "Aplicação resetada. Faça upload de um novo arquivo.", pd.DataFrame(), "", None, [], ""

# Interface Gradio (revertida para o layout original)
with gr.Blocks(theme='Soft') as app:
    gr.Markdown("# Analisando os dados🔎🎲")
    gr.Markdown('''
        Carregue um arquivo CSV e faça perguntas sobre os dados. A cada pergunta, você poderá
        visualizar a resposta e, se desejar, adicionar essa interação ao PDF final, basta clicar
        em "Adicionar ao histórico do PDF". Para fazer uma nova pergunta, clique em "Limpar pergunta e resultado".
        Após definir as perguntas e respostas no histórico, clique em "Gerar PDF". Assim, será possível
        baixar um PDF com o registro completo das suas interações. Se você quiser analisar um novo dataset,
        basta clicar em "Quero analisar outro dataset" ao final da página.
    ''')

    input_arquivo = gr.File(file_count="single", type="filepath", label="Upload CSV")
    upload_status = gr.Textbox(label="Status do Upload:")
    tabela_dados = gr.DataFrame()

    gr.Markdown("""
        Exemplos de perguntas:
        1. Qual é o número de registros no arquivo?
        2. Quais são os tipos de dados das colunas?
        3. Quais são as estatísticas descritivas das colunas numéricas?
    """)

    input_pergunta = gr.Textbox(label="Digite sua pergunta sobre os dados")
    botao_submeter = gr.Button("Enviar")
    output_resposta = gr.Textbox(label="Resposta")

    with gr.Row():
        botao_limpeza = gr.Button("Limpar pergunta e resultado")
        botao_add_pdf = gr.Button("Adicionar ao histórico do PDF")
        botao_gerar_pdf = gr.Button("Gerar PDF")

    arquivo_pdf = gr.File(label="Download do PDF")
    botao_resetar = gr.Button("Quero analisar outro dataset!")

    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])

    input_arquivo.change(fn=carregar_dados, inputs=[input_arquivo, df_estado], outputs=[upload_status, tabela_dados, df_estado])
    botao_submeter.click(fn=processar_pergunta, inputs=[input_pergunta, df_estado], outputs=output_resposta)
    botao_limpeza.click(fn=limpar_pergunta_resposta, inputs=[], outputs=[input_pergunta, output_resposta])
    botao_add_pdf.click(fn=add_historico, inputs=[input_pergunta, output_resposta, historico_estado], outputs=[historico_estado])
    botao_gerar_pdf.click(fn=gerar_pdf, inputs=[historico_estado], outputs=[arquivo_pdf])
    botao_resetar.click(fn=resetar_aplicacao, inputs=[], outputs=[input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta])

if __name__ == "__main__":
    app.launch()