from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import (QueryPipeline as QP, Link, InputComponent)
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import io

api_key = os.getenv("secret_key")

llm = Groq(model="llama3-70b-8192", api_key=api_key)

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
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str,
        df_str=df.head(5),
        colunas_detalhes=descrição_colunas(df)
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
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
            Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
        ]
    )
    qp.add_link("response_synthesis_prompt", "llm2")
    return qp

def detectar_tipo_arquivo(caminho_arquivo):
    """Detecta o tipo de arquivo baseado na extensão."""
    if caminho_arquivo is None:
        return None
    
    extensao = caminho_arquivo.lower().split('.')[-1]
    if extensao in ['csv']:
        return 'csv'
    elif extensao in ['xlsx', 'xls', 'xlsm', 'xlsb']:
        return 'excel'
    else:
        return 'desconhecido'

def carregar_dados(caminho_arquivo, df_estado):
    if caminho_arquivo is None or caminho_arquivo == "":
        return "Por favor, faça o upload de um arquivo CSV ou Excel para analisar.", pd.DataFrame(), df_estado, ""
    
    try:
        tipo_arquivo = detectar_tipo_arquivo(caminho_arquivo)
        
        if tipo_arquivo == 'csv':
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(caminho_arquivo, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                return "Erro: Não foi possível ler o arquivo CSV com nenhum encoding suportado.", pd.DataFrame(), df_estado, ""
                
        elif tipo_arquivo == 'excel':
            try:
                df = pd.read_excel(caminho_arquivo)
            except Exception as e:
                try:
                    df = pd.read_excel(caminho_arquivo, sheet_name=0)
                except Exception as e2:
                    return f"Erro ao carregar arquivo Excel: {str(e2)}", pd.DataFrame(), df_estado, ""
        else:
            return "Formato de arquivo não suportado. Por favor, faça upload de um arquivo CSV (.csv) ou Excel (.xlsx, .xls).", pd.DataFrame(), df_estado, ""
        
        if df.empty:
            return "O arquivo foi carregado, mas está vazio.", pd.DataFrame(), df_estado, ""
        
        colunas_str = '\n'.join(df.columns)
        tipo_arquivo_msg = "CSV" if tipo_arquivo == 'csv' else "Excel"
        return f"Arquivo {tipo_arquivo_msg} carregado com sucesso! ({len(df)} linhas, {len(df.columns)} colunas)", df.head(), df, colunas_str
        
    except Exception as e:
        return f"Erro ao carregar arquivo: {str(e)}", pd.DataFrame(), df_estado, ""

def processar_pergunta(pergunta, df_estado):
    if df_estado is not None and pergunta:
        qp = pipeline_consulta(df_estado)
        resposta = qp.run(query_str=pergunta)
        return resposta.message.content
    return ""

def get_descriptive_stats_and_info(df):
    """Gera o texto com estatísticas descritivas no formato de tabela Markdown."""
    if df is None or df.empty:
        return "Por favor, carregue um arquivo CSV ou Excel primeiro."

    output = io.StringIO()

    output.write("### Estatísticas Descritivas:\n\n")

    try:
        stats_desc = df.describe(include='all')
        header = "| Estatística | " + " | ".join([f"{col}" for col in stats_desc.columns]) + " |"
        output.write(header + "\n")
        separator = "|" + "---|" * (len(stats_desc.columns) + 1)
        output.write(separator + "\n")

        for stat in stats_desc.index:
            row = f"| {stat} |"
            for col in stats_desc.columns:
                val = stats_desc.loc[stat, col]
                if pd.isna(val):
                    row += " - |"
                else:
                    if stat in ['count', 'unique', 'freq']:
                        row += f" {int(val)} |"
                    elif stat in ['top']:
                        row += f" {val} |"
                    else:
                        if val == int(val):
                            row += f" {int(val)} |"
                        else:
                            row += f" {val:.2f} |".replace(".00 ", " ")
            output.write(row + "\n")

    except Exception as e:
        output.write(f"Erro ao gerar estatísticas descritivas: {e}")

    output.write("\n\n### Informações do DataFrame:\n")

    try:
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()
        formatted_info = info_text.replace("NaN", "-").replace("nan", "-")
        output.write(formatted_info)
    except Exception as e:
        output.write(f"Erro ao gerar informações do DataFrame: {e}")

    return output.getvalue()

def add_historico(pergunta, resposta, historico_estado):
    """Adiciona pergunta/resposta ao histórico."""
    if pergunta and resposta:
        historico_estado.append(("qa", (pergunta, resposta)))
        gr.Info("Pergunta e resposta adicionadas ao histórico do PDF!", duration=2)
        return historico_estado
    return historico_estado

def add_stats_to_historico(stats_text, historico_estado):
    """Adiciona texto de estatísticas ao histórico."""
    if stats_text and "Por favor, carregue" not in stats_text:
        historico_estado.append(("stats", stats_text))
        gr.Info("Estatísticas adicionadas ao histórico do PDF!", duration=2)
        return historico_estado
    gr.Warning("Nenhuma estatística gerada para adicionar ao histórico.", duration=2)
    return historico_estado

def gerar_pdf(historico_estado, titulo, nome_usuario):
    if not historico_estado:
        return "Nenhum dado para adicionar ao PDF.", None

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    caminho_pdf = f"relatorio_perguntas_respostas_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', '', 12)

    if titulo:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt=titulo, ln=True, align='C')
        pdf.ln(10)

    if nome_usuario:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, txt=f"Relatório gerado por: {nome_usuario}", ln=True, align='C')
        pdf.ln(10)

    for entry_type, content in historico_estado:
        if entry_type == "qa":
            pergunta, resposta = content
            pergunta_encoded = pergunta.encode('latin-1', 'replace').decode('latin-1')
            resposta_encoded = resposta.encode('latin-1', 'replace').decode('latin-1')

            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 8, txt=pergunta_encoded)
            pdf.ln(2)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, txt=resposta_encoded)
            pdf.ln(6)
        elif entry_type == "stats":
            stats_text = content
            stats_text = stats_text.replace("nan", "-")
            stats_encoded = stats_text.encode('latin-1', 'replace').decode('latin-1')

            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 8, txt="Estatísticas e Informações do DataFrame:")
            pdf.ln(2)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 6, txt=stats_encoded)
            pdf.ln(6)

    pdf.output(caminho_pdf)
    return caminho_pdf

def limpar_pergunta_resposta():
    return "", ""

def limpar_historico(historico_estado):
    """Limpa o estado do histórico para o relatório PDF."""
    gr.Info("Histórico do PDF limpo!")
    return []

def resetar_aplicação():
    return None, "A aplicação foi resetada. Por favor, faça upload de um novo arquivo CSV ou Excel.", pd.DataFrame(), "", "", None, [], "", "", ""

with gr.Blocks(theme='Soft') as app:

    gr.Markdown("# Analisando os dados🔎🎲")

    gr.Markdown('''
    Carregue um arquivo **CSV** ou **Excel** (.xlsx, .xls) e faça perguntas sobre os dados. A cada pergunta, você poderá
    visualizar a resposta e, se desejar, adicionar essa interação ao PDF final, basta clicar
    em "Adicionar ao histórico do PDF". Para fazer uma nova pergunta, clique em "Limpar pergunta e resultado".
    Você também pode visualizar as estatísticas descritivas do dataset e adicioná-las ao PDF.
    Após definir as entradas no histórico, clique em "Gerar PDF". Assim, será possível
    baixar um PDF com o registro completo das suas interações. Se você quiser analisar um novo dataset,
    basta clicar em "Quero analisar outro dataset" ao final da página.
    
    **Formatos suportados:** CSV (.csv), Excel Needs to be corrected (.xlsx, .xls, .xlsm, .xlsb)
    ''')

    input_arquivo = gr.File(
        file_count="single", 
        type="filepath", 
        label="Upload CSV ou Excel",
        file_types=[".csv", ".xlsx", ".xls", ".xlsm", ".xlsb"]
    )

    upload_status = gr.Textbox(label="Status do Upload:")

    tabela_dados = gr.DataFrame()

    output_colunas = gr.Textbox(label="Colunas Disponíveis:", lines=5)

    with gr.Accordion("Estatísticas e Informações do Dataset", open=False):
        botao_mostrar_stats = gr.Button("Mostrar Estatísticas e Info")
        output_stats = gr.Textbox(label="Estatísticas e Info do DataFrame:", lines=15, interactive=False)
        botao_add_stats_pdf = gr.Button("Adicionar Estatísticas ao Histórico do PDF")

    gr.Markdown("""
    Exemplos de perguntas:
    1. Qual é o número de registros no arquivo?
    2. Quais são os tipos de dados das colunas?
    3. Quais são as estatísticas descritivas das colunas numéricas?
    4. Quais são os valores únicos de uma coluna específica?
    5. Quantos valores nulos existem em cada coluna?
    6. Quais são as linhas onde a coluna X é maior que Y?
    7. Qual é a média da coluna Z agrupada por W?
    8. Quais são as 5 linhas com os maiores valores na coluna A?
    9. Qual é o percentil 75 da coluna B?
    10. Existe correlação entre as colunas C e D?
    11. Quantas linhas têm valores na coluna E iguais a 'valor_específico'?
    12. Qual é a soma total da coluna F para cada categoria em G?
    13. Quais são as linhas onde a coluna H está entre X e Y?
    14. Qual é a diferença entre os valores máximo e mínimo da coluna I?
    15. Quantas linhas têm valores não nulos em todas as colunas?
    """)

    input_pergunta = gr.Textbox(label="Digite sua pergunta sobre os dados")

    botao_submeter = gr.Button("Enviar")

    output_resposta = gr.Textbox(label="Resposta")

    titulo = gr.Textbox(label="Título do Relatório")
    nome_usuario = gr.Textbox(label="Seu Nome")

    with gr.Row():
        botao_limpeza = gr.Button("Limpar pergunta e resultado")
        botao_add_pdf = gr.Button("Adicionar Pergunta/Resposta ao Histórico do PDF")
        botao_limpar_historico = gr.Button("Limpar Histórico do PDF")
        botao_gerar_pdf = gr.Button("Gerar PDF")

    arquivo_pdf = gr.File(label="Download do PDF")

    botao_resetar = gr.Button("Quero analisar outro dataset!")

    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])

    input_arquivo.change(fn=carregar_dados, inputs=[input_arquivo, df_estado], outputs=[upload_status, tabela_dados, df_estado, output_colunas], show_progress=True)
    botao_mostrar_stats.click(fn=get_descriptive_stats_and_info, inputs=[df_estado], outputs=output_stats, show_progress=True)
    botao_add_stats_pdf.click(fn=add_stats_to_historico, inputs=[output_stats, historico_estado], outputs=historico_estado, show_progress=False)
    botao_submeter.click(fn=processar_pergunta, inputs=[input_pergunta, df_estado], outputs=output_resposta, show_progress=True)
    botao_limpeza.click(fn=limpar_pergunta_resposta, inputs=[], outputs=[input_pergunta, output_resposta])
    botao_add_pdf.click/ns=[input_pergunta, output_resposta, historico_estado], outputs=historico_estado)
    botao_limpar_historico.click(fn=limpar_historico, inputs=[historico_estado], outputs=historico_estado)
    botao_gerar_pdf.click(fn=gerar_pdf, inputs=[historico_estado, titulo, nome_usuario], outputs=arquivo_pdf, show_progress=True)
    botao_resetar.click(fn=resetar_aplicação, inputs=[], outputs=[input_arquivo, upload_status, tabela_dados, output_colunas, output_stats, arquivo_pdf, historico_estado, input_pergunta, output_resposta, titulo, nome_usuario])

if __name__ == "__main__":
    app.launch()