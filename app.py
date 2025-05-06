from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import (QueryPipeline as QP, Link, InputComponent)
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API KEY do GROQ
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    logger.warning("GROQ_API_KEY não encontrada no ambiente. Configure a variável de ambiente.")

# Configuração inicial do LLM
def get_llm():
    try:
        return Groq(model="llama3-70b-8192", api_key=api_key)
    except Exception as e:
        logger.error(f"Erro ao inicializar o LLM: {str(e)}")
        raise

# Função para descrever as colunas do dataframe
def descrição_colunas(df):
    try:
        descrição = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
        return "Aqui estão os detalhes das colunas do dataframe:\n" + descrição
    except Exception as e:
        logger.error(f"Erro ao descrever colunas: {str(e)}")
        return "Erro ao descrever colunas do dataframe."

# Pipeline de consulta
def pipeline_consulta(df):
    try:
        llm = get_llm()
        
        instruction_str = (
            "1. Converta a consulta para código Python executável usando Pandas.\n"
            "2. A linha final do código deve ser uma expressão Python que possa ser chamada com a função `eval()`.\n"
            "3. O código deve representar uma solução para a consulta.\n"
            "4. IMPRIMA APENAS A EXPRESSÃO.\n"
            "5. Não coloque a expressão entre aspas.\n")

        pandas_prompt_str = (
            "Você está trabalhando com um dataframe do pandas em Python chamado `df`.\n"
            "{colunas_detalhes}\n\n"
            "Este é o resultado de `print(df.head())`:\n"
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
           "Ao final, exibir o código usado para gerar a resposta, no formato: O código utilizado foi `{pandas_instructions}`"
        )

        pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
            instruction_str=instruction_str,
            df_str=df.head(5),
            colunas_detalhes=descrição_colunas(df)
        )

        pandas_output_parser = PandasInstructionParser(df)
        response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

        # Criação do QueryPipeline
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
        qp.add_links(
            [
                Link("input", "response_synthesis_prompt", dest_key="query_str"),
                Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
                Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
            ]
        )
        qp.add_link("response_synthesis_prompt", "llm2")
        return qp
    except Exception as e:
        logger.error(f"Erro ao criar pipeline de consulta: {str(e)}")
        raise

# Função para carregar os dados
def carregar_dados(caminho_arquivo, df_estado):
    if caminho_arquivo is None or caminho_arquivo == "":
        return "Por favor, faça o upload de um arquivo CSV para analisar.", None, df_estado
    
    try:
        df = pd.read_csv(caminho_arquivo)
        logger.info(f"Arquivo carregado: {caminho_arquivo}, {len(df)} linhas, {len(df.columns)} colunas")
        return "Arquivo carregado com sucesso!", df.head(), df
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo {caminho_arquivo}: {str(e)}")
        return f"Erro ao carregar arquivo: {str(e)}", None, df_estado

# Função para processar a pergunta
def processar_pergunta(pergunta, df_estado):
    if df_estado is None:
        return "Por favor, carregue um arquivo CSV primeiro."
    
    if not pergunta.strip():
        return "Por favor, digite uma pergunta."
        
    try:
        logger.info(f"Processando pergunta: {pergunta}")
        qp = pipeline_consulta(df_estado)
        resposta = qp.run(query_str=pergunta)
        return resposta.message.content
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}")
        return f"Erro ao processar a pergunta: {str(e)}"

# Função para adicionar a pergunta e a resposta ao histórico
def add_historico(pergunta, resposta, historico_estado):
    if not pergunta.strip() or not resposta.strip():
        gr.Warning("Pergunta ou resposta vazia, não adicionada ao histórico.")
        return historico_estado
    
    historico_estado.append((pergunta, resposta))
    gr.Info("Adicionado ao histórico do PDF!", duration=2)
    logger.info(f"Adicionado ao histórico: {pergunta[:30]}...")
    return historico_estado

# Função para gerar o PDF
def gerar_pdf(historico_estado):
    if not historico_estado:
        return "Nenhum dado para adicionar ao PDF. Adicione perguntas e respostas ao histórico primeiro.", None

    try:
        # Gerar nome de arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        caminho_pdf = f"relatorio_analise_dados_{timestamp}.pdf"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Adicionar título
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Relatório de Análise de Dados", 0, 1, 'C')
        pdf.ln(5)
        
        # Adicionar data
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 5, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1, 'R')
        pdf.ln(10)

        for i, (pergunta, resposta) in enumerate(historico_estado, 1):
            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 8, txt=f"Pergunta {i}: {pergunta}")
            pdf.ln(2)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, txt=resposta)
            pdf.ln(6)

        pdf.output(caminho_pdf)
        logger.info(f"PDF gerado: {caminho_pdf}")
        return caminho_pdf
    except Exception as e:
        logger.error(f"Erro ao gerar PDF: {str(e)}")
        return f"Erro ao gerar PDF: {str(e)}", None

# Função para limpar a pergunta e a resposta
def limpar_pergunta_resposta():
    return "", ""

# Função para visualizar o histórico
def visualizar_historico(historico_estado):
    if not historico_estado:
        return "Histórico vazio. Adicione perguntas e respostas primeiro."
    
    resultado = ""
    for i, (pergunta, resposta) in enumerate(historico_estado, 1):
        resultado += f"### Pergunta {i}: {pergunta}\n\n{resposta}\n\n---\n\n"
    
    return resultado

# Função para resetar a aplicação
def resetar_aplicação():
    return None, "A aplicação foi resetada. Por favor, faça upload de um novo arquivo CSV.", None, "", None, [], "", ""

# Criação da interface gradio
with gr.Blocks(theme='soft') as app:
    # Título da app com estilo melhorado
    gr.Markdown(
        """
        # 🔎 Analisador de Dados Inteligente 🎲
        ### Analise seus dados CSV com perguntas em linguagem natural
        """
    )

    # Descrição da aplicação
    with gr.Accordion("ℹ️ Como usar", open=False):
        gr.Markdown('''
        **Instruções:**
        
        1. **Carregue um arquivo CSV** usando o botão de upload abaixo
        2. **Faça perguntas** sobre os dados em linguagem natural
        3. **Adicione perguntas e respostas** ao histórico do PDF se quiser salvar
        4. **Gere um PDF** com todas as perguntas e respostas quando finalizar
        
        Para analisar um novo conjunto de dados, clique em "Resetar Aplicação" no final da página.
        ''')

    # Campo de entrada de arquivos com feedback visual
    with gr.Row():
        with gr.Column(scale=4):
            input_arquivo = gr.File(
                file_count="single", 
                type="filepath", 
                label="Upload do arquivo CSV"
            )
        with gr.Column(scale=1):
            upload_status = gr.Textbox(label="Status", value="Aguardando upload...")

    # Preview dos dados
    with gr.Row():
        tabela_dados = gr.DataFrame(label="Visualização dos Dados")

    # Seção de exemplos de perguntas
    with gr.Accordion("🔍 Exemplos de perguntas", open=True):
        gr.Markdown("""
        Exemplos do que você pode perguntar:
        
        * "Quantas linhas e colunas existem no dataset?"
        * "Quais são os valores mínimos e máximos da coluna X?"
        * "Qual é a média da coluna Y agrupada por Z?"
        * "Mostre a distribuição dos valores da coluna X"
        * "Quais são os registros onde o valor da coluna X é maior que 100?"
        * "Existem valores nulos no dataset? Onde estão localizados?"
        * "Qual é a correlação entre as colunas numéricas?"
        """)

    # Interface de perguntas
    gr.Markdown("## 💬 Faça sua pergunta")
    with gr.Row():
        input_pergunta = gr.Textbox(
            label="Digite sua pergunta sobre os dados",
            placeholder="Ex: Qual é a média da coluna X?",
            lines=2
        )
        botao_submeter = gr.Button("📊 Analisar", variant="primary")

    # Exibição da resposta
    output_resposta = gr.Textbox(
        label="Resposta",
        lines=10
    )

    # Botões de ação
    with gr.Row():
        botao_limpeza = gr.Button("🧹 Limpar pergunta e resultado")
        botao_add_pdf = gr.Button("➕ Adicionar ao histórico", variant="secondary")
        
    # Visualização do histórico
    with gr.Accordion("📜 Histórico de Perguntas e Respostas", open=False):
        output_historico = gr.Markdown()
        visualizar_btn = gr.Button("🔄 Atualizar histórico")
    
    # Geração do PDF
    gr.Markdown("## 📄 Exportar Resultados")
    with gr.Row():
        botao_gerar_pdf = gr.Button("📑 Gerar PDF", variant="primary")
        arquivo_pdf = gr.File(label="Download do PDF")

    # Botão para resetar a aplicação
    gr.Markdown("---")
    botao_resetar = gr.Button("🔄 Resetar Aplicação", variant="stop")

    # Gerenciamento de estados
    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])

    # Conectando funções aos componentes
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
        outputs=historico_estado
    )
    
    visualizar_btn.click(
        fn=visualizar_historico,
        inputs=[historico_estado],
        outputs=[output_historico]
    )
    
    botao_gerar_pdf.click(
        fn=gerar_pdf, 
        inputs=[historico_estado], 
        outputs=arquivo_pdf
    )
    
    botao_resetar.click(
        fn=resetar_aplicação, 
        inputs=[], 
        outputs=[input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta, output_historico]
    )

if __name__ == "__main__":
    try:
        logger.info("Iniciando a aplicação")
        app.launch(share=False)
    except Exception as e:
        logger.error(f"Erro ao iniciar a aplicação: {str(e)}")