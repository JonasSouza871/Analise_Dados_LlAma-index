from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import (QueryPipeline as QP, Link, InputComponent)
import gradio as gr
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# API KEY do GROQ
api_key = os.getenv("secret_key")

# Configuração inicial do QP
llm = Groq(model="llama3-70b-8192", api_key=api_key)

# Funções de análise de dados
def obter_estatisticas_basicas(df):
    """Retorna estatísticas básicas do dataframe"""
    return df.describe().transpose()

def obter_tipos_dados(df):
    """Retorna os tipos de dados de cada coluna"""
    tipos = pd.DataFrame({
        'Coluna': df.columns,
        'Tipo': df.dtypes.astype(str),
        'Valores Nulos': df.isnull().sum(),
        'Valores Únicos': [df[col].nunique() for col in df.columns]
    })
    return tipos

def gerar_grafico_distribuicao(df, coluna):
    """Gera um gráfico de distribuição para uma coluna numérica"""
    plt.figure(figsize=(10, 6))
    if np.issubdtype(df[coluna].dtype, np.number):
        sns.histplot(df[coluna].dropna(), kde=True)
        plt.title(f'Distribuição de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
    else:
        counts = df[coluna].value_counts().head(15)
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f'Distribuição de {coluna} (Top 15)')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(coluna)
        plt.ylabel('Contagem')
    plt.tight_layout()
    
    # Salvar a figura em um buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Pipeline de consulta
def descrição_colunas(df):
    descrição = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Aqui estão os detalhes das colunas do dataframe:\n" + descrição

def pipeline_consulta(df):
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

# Função para carregar os dados
def carregar_dados(caminho_arquivo, df_estado):
    if caminho_arquivo is None or caminho_arquivo == "":
        return "Por favor, faça o upload de um arquivo CSV ou Excel para analisar.", pd.DataFrame(), df_estado, None, None, None
    
    try:
        # Verificar extensão do arquivo
        nome_arquivo = os.path.basename(caminho_arquivo)
        extensao = os.path.splitext(nome_arquivo)[1].lower()
        
        if extensao == '.csv':
            df = pd.read_csv(caminho_arquivo)
        elif extensao in ['.xlsx', '.xls']:
            df = pd.read_excel(caminho_arquivo)
        else:
            return f"Formato de arquivo não suportado: {extensao}. Por favor, utilize CSV ou Excel.", pd.DataFrame(), df_estado, None, None, None
        
        # Gerar resumo e estatísticas
        resumo = {
            "Linhas": len(df),
            "Colunas": len(df.columns),
            "Colunas numéricas": len(df.select_dtypes(include=['number']).columns),
            "Colunas categóricas": len(df.select_dtypes(include=['object', 'category']).columns),
            "Valores ausentes": df.isnull().sum().sum()
        }
        
        estatisticas = obter_estatisticas_basicas(df)
        tipos_dados = obter_tipos_dados(df)
        
        return f"Arquivo '{nome_arquivo}' carregado com sucesso!", df.head(), df, resumo, estatisticas, tipos_dados
    except Exception as e:
        return f"Erro ao carregar arquivo: {str(e)}", pd.DataFrame(), df_estado, None, None, None

# Função para processar a pergunta
def processar_pergunta(pergunta, df_estado):
    if df_estado is not None and pergunta:
        qp = pipeline_consulta(df_estado)
        resposta = qp.run(query_str=pergunta)
        return resposta.message.content
    return ""

# Função para adicionar a pergunta e a resposta ao histórico
def add_historico(pergunta, resposta, historico_estado):
    if pergunta and resposta:
        historico_estado.append((pergunta, resposta))
        gr.Info("Adicionado ao PDF!", duration=2)
        return historico_estado

# Função para gerar o PDF
def gerar_pdf(historico_estado, resumo_dados=None):
    if not historico_estado and not resumo_dados:
        return "Nenhum dado para adicionar ao PDF.", None

    # Gerar nome de arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    caminho_pdf = f"relatorio_analise_dados_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Adicionar título e data
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório de Análise de Dados", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, f"Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}", ln=True, align='C')
    pdf.ln(5)
    
    # Adicionar resumo dos dados se disponível
    if resumo_dados:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Resumo do Dataset", ln=True)
        pdf.ln(2)
        
        pdf.set_font("Arial", '', 11)
        for chave, valor in resumo_dados.items():
            pdf.cell(60, 8, f"{chave}:", 0)
            pdf.cell(0, 8, f"{valor}", ln=True)
        
        pdf.ln(5)
    
    # Adicionar perguntas e respostas
    if historico_estado:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Perguntas e Respostas", ln=True)
        pdf.ln(2)
        
        for i, (pergunta, resposta) in enumerate(historico_estado):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Pergunta {i+1}:", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 8, txt=pergunta)
            pdf.ln(2)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Resposta {i+1}:", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 8, txt=resposta)
            pdf.ln(6)

    pdf.output(caminho_pdf)
    return caminho_pdf

# Função para gerar gráfico com base na coluna selecionada
def atualizar_grafico(coluna_selecionada, df_estado):
    if df_estado is None or not coluna_selecionada:
        return None
    try:
        return gerar_grafico_distribuicao(df_estado, coluna_selecionada)
    except Exception as e:
        print(f"Erro ao gerar gráfico: {str(e)}")
        return None

# Função para limpar a pergunta e a resposta
def limpar_pergunta_resposta():
    return "", ""

# Função para resetar a aplicação
def resetar_aplicação():
    return None, "A aplicação foi resetada. Por favor, faça upload de um novo arquivo.", pd.DataFrame(), "", None, [], "", None, None, None, None

# Definição de CSS personalizado
css = """
.gradio-container {
    background-color: #f8f9fa;
}
.header-text {
    text-align: center;
    font-size: 2.5rem;
    color: #1a5276;
    font-weight: bold;
    margin-bottom: 1rem;
}
.subheader-text {
    text-align: center;
    font-size: 1.2rem;
    color: #566573;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2874a6;
    font-weight: bold;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}
.info-text {
    background-color: #ebf5fb;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #3498db;
    margin-bottom: 1rem;
}
.example-text {
    background-color: #e8f8f5;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1abc9c;
    margin-bottom: 1rem;
}
.submit-btn {
    background-color: #2874a6 !important;
}
.add-btn {
    background-color: #27ae60 !important;
}
.clear-btn {
    background-color: #e67e22 !important;
}
.pdf-btn {
    background-color: #c0392b !important;
}
.reset-btn {
    background-color: #7d3c98 !important;
}
"""

# Criação da interface gradio
with gr.Blocks(theme=gr.themes.Soft(), css=css) as app:
    # Cabeçalho da aplicação
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div class="header-text">📊 DataExplorer Pro 📈</div>
                <div class="subheader-text">Ferramenta interativa para análise de dados com IA</div>
                """
            )
    
    # Seção de upload de arquivos
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="section-header">1. Upload de Dados</div>')
            gr.HTML(
                """
                <div class="info-text">
                Carregue seu arquivo de dados (CSV ou Excel) para começar a análise. O sistema processará 
                automaticamente os dados e disponibilizará estatísticas básicas.
                </div>
                """
            )
            input_arquivo = gr.File(
                file_count="single", 
                type="filepath", 
                label="Upload do Arquivo (CSV ou Excel)",
                file_types=[".csv", ".xlsx", ".xls"]
            )
            upload_status = gr.Textbox(label="Status do Upload:", interactive=False)
    
    # Seção de visualização e análise exploratória
    with gr.Tabs() as tabs:
        with gr.TabItem("Visualização de Dados"):
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">2. Pré-visualização dos Dados</div>')
                    tabela_dados = gr.DataFrame(label="Primeiras linhas do dataset")
            
            with gr.Row():
                with gr.Column():
                    resumo_dados = gr.JSON(label="Resumo do Dataset", interactive=False)
                with gr.Column():
                    tipos_dados = gr.DataFrame(label="Tipos de Dados e Informações das Colunas")
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">3. Visualização Gráfica</div>')
                    coluna_selecionada = gr.Dropdown(label="Selecione uma coluna para visualizar")
                    grafico = gr.Image(label="Gráfico de Distribuição")
            
        with gr.TabItem("Estatísticas Avançadas"):
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">Estatísticas Descritivas</div>')
                    estatisticas_df = gr.DataFrame(label="Estatísticas descritivas das colunas numéricas")
    
    # Seção de perguntas e respostas
    with gr.Tab("Perguntas e Respostas"):
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-header">4. Faça perguntas sobre seus dados</div>')
                gr.HTML(
                    """
                    <div class="example-text">
                    <b>Exemplos de perguntas:</b>
                    <ul>
                      <li>Qual é o número total de registros no arquivo?</li>
                      <li>Qual é a média da coluna [nome_da_coluna]?</li>
                      <li>Quais são os 5 valores mais frequentes na coluna [nome_da_coluna]?</li>
                      <li>Qual é a correlação entre as colunas numéricas?</li>
                      <li>Como se comporta a distribuição dos dados na coluna [nome_da_coluna]?</li>
                    </ul>
                    </div>
                    """
                )
            
        with gr.Row():
            with gr.Column():
                input_pergunta = gr.Textbox(
                    label="Digite sua pergunta sobre os dados", 
                    placeholder="Ex: Qual é a média da coluna idade?"
                )
                with gr.Row():
                    botao_submeter = gr.Button("Enviar Pergunta", elem_classes="submit-btn")
                
        with gr.Row():
            with gr.Column():
                output_resposta = gr.Textbox(
                    label="Resposta", 
                    lines=10,
                    interactive=False
                )
                
        with gr.Row():
            botao_limpeza = gr.Button("Limpar pergunta e resultado", elem_classes="clear-btn")
            botao_add_pdf = gr.Button("Adicionar ao histórico do PDF", elem_classes="add-btn")
        
    # Seção de geração de relatórios
    with gr.Tab("Geração de Relatório"):
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-header">5. Gerar relatório em PDF</div>')
                gr.HTML(
                    """
                    <div class="info-text">
                    Gere um relatório em PDF contendo todas as perguntas e respostas adicionadas ao histórico,
                    além de estatísticas básicas sobre os dados.
                    </div>
                    """
                )
                
        with gr.Row():
            with gr.Column():
                botao_gerar_pdf = gr.Button("Gerar PDF", elem_classes="pdf-btn")
                arquivo_pdf = gr.File(label="Download do Relatório")
        
    # Botão para resetar a aplicação
    with gr.Row():
        with gr.Column():
            botao_resetar = gr.Button("Reiniciar Análise (Novo Dataset)", elem_classes="reset-btn")
    
    # Footer
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                    <p>DataExplorer Pro v2.0 © 2025</p>
                </div>
                """
            )
            
    # Gerenciamento de estados
    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])
    resumo_state = gr.State(value=None)
    
    # Conectando funções aos componentes
    input_arquivo.change(
        fn=carregar_dados, 
        inputs=[input_arquivo, df_estado], 
        outputs=[upload_status, tabela_dados, df_estado, resumo_dados, estatisticas_df, tipos_dados]
    )
    
    # Atualizar dropdown de colunas quando o dataframe mudar
    def atualizar_colunas(df_estado):
        if df_estado is None:
            return []
        return list(df_estado.columns)
    
    input_arquivo.change(
        fn=atualizar_colunas,
        inputs=[df_estado],
        outputs=[coluna_selecionada]
    )
    
    # Atualizar gráfico quando a coluna for selecionada
    coluna_selecionada.change(
        fn=atualizar_grafico,
        inputs=[coluna_selecionada, df_estado],
        outputs=[grafico]
    )
    
    # Processar pergunta
    botao_submeter.click(
        fn=processar_pergunta, 
        inputs=[input_pergunta, df_estado], 
        outputs=output_resposta
    )
    
    # Ações dos botões
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
    
    # Função intermediária para passar o resumo junto com o histórico
    def gerar_pdf_com_resumo(historico, resumo):
        return gerar_pdf(historico, resumo)
    
    # Salvar resumo quando carregar arquivo
    def salvar_resumo(resumo):
        return resumo
    
    input_arquivo.change(
        fn=salvar_resumo,
        inputs=[resumo_dados],
        outputs=[resumo_state]
    )
    
    botao_gerar_pdf.click(
        fn=gerar_pdf_com_resumo, 
        inputs=[historico_estado, resumo_state], 
        outputs=arquivo_pdf
    )
    
    botao_resetar.click(
        fn=resetar_aplicação, 
        inputs=[], 
        outputs=[
            input_arquivo, upload_status, tabela_dados, output_resposta, 
            arquivo_pdf, historico_estado, input_pergunta, resumo_dados,
            estatisticas_df, tipos_dados, grafico
        ]
    )

if __name__ == "__main__":
    app.launch()