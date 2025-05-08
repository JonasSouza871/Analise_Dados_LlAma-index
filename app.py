import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import (QueryPipeline as QP, Link, InputComponent)

api_key = os.getenv("secret_key")

llm = Groq(model="llama3-70b-8192", api_key=api_key)

def get_column_details(df):
    details = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Aqui estão os detalhes das colunas do dataframe:\n" + details

def build_query_pipeline(df):
    instruction_str = (
        "1. Converta a consulta para código Python executável usando Pandas.\n"
        "2. A linha final do código deve ser uma expressão Python que possa ser chamada com a função `eval()`.\n"
        "3. O código deve representar uma solução para a consulta.\n"
        "4. IMPRIMA APENAS A EXPRESSÃO.\n"
        "5. Não coloque a expressão entre aspas.\n"
        "6. Se a consulta sugerir uma visualização, em uma nova linha APÓS a expressão, sugira um tipo de plotagem e coluna(s) no formato 'PLOT: tipo:coluna(s)', por exemplo 'PLOT: histogram:idade' ou 'PLOT: bar:categoria'. Se não houver plotagem relevante, apenas termine com a expressão."
    )

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
       "Saída do Pandas: {pandas_output}\n\n"
       "{plot_status}\n"
       "Resposta: \n\n"
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str,
        df_str=df.head(5),
        colunas_detalhes=get_column_details(df)
    )

    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )

    qp.add_chain(["input", "pandas_prompt", "llm1"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            # pandas_output will be added dynamically after parsing llm1 output and executing
        ]
    )
    qp.add_link("response_synthesis_prompt", "llm2")
    return qp

def parse_llm_output(llm_output):
    lines = llm_output.strip().split('\n')
    code_expression = lines[0] if lines else ""
    plot_suggestion = None
    for line in lines[1:]:
        if line.strip().upper().startswith("PLOT:"):
            plot_suggestion = line.strip()[5:].strip()
            break
    return code_expression, plot_suggestion

def generate_and_save_plot(df, plot_suggestion):
    plot_suggestion = plot_suggestion.lower()
    plot_type = None
    column = None

    match = re.match(r"(\w+):\s*(\w+)", plot_suggestion)
    if match:
        plot_type = match.group(1)
        column = match.group(2)

    if not plot_type or not column or column not in df.columns:
        return None, f"Sugestão de plotagem inválida ou coluna '{column}' não encontrada."

    plot_path = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"

    try:
        plt.figure(figsize=(8, 5))
        if plot_type == 'histogram':
            if pd.api.types.is_numeric_dtype(df[column]):
                sns.histplot(df[column], kde=True)
                plt.title(f"Distribuição de {column}")
                plt.xlabel(column)
                plt.ylabel("Frequência")
            else:
                 return None, f"Coluna '{column}' não é numérica para histograma."
        elif plot_type == 'bar':
            if pd.api.types.is_categorical_dtype(df[column]) or df[column].nunique() < 50: # Simple heuristic for bar
                 df[column].value_counts().sort_index().plot(kind='bar')
                 plt.title(f"Contagem por {column}")
                 plt.xlabel(column)
                 plt.ylabel("Contagem")
                 plt.xticks(rotation=45, ha='right')
            else:
                 return None, f"Coluna '{column}' tem muitos valores únicos para um gráfico de barras simples."

        elif plot_type == 'scatter':
             cols = column.split(':')
             if len(cols) == 2 and pd.api.types.is_numeric_dtype(df[cols[0]]) and pd.api.types.is_numeric_dtype(df[cols[1]]):
                 sns.scatterplot(data=df, x=cols[0], y=cols[1])
                 plt.title(f"Relação entre {cols[0]} e {cols[1]}")
                 plt.xlabel(cols[0])
                 plt.ylabel(cols[1])
             else:
                 return None, f"Sugestão de scatter plot inválida ou colunas não numéricas: {column}. Use formato 'scatter:colunaX:colunaY'."
        else:
            return None, f"Tipo de plotagem '{plot_type}' não suportado."

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        return plot_path, None

    except Exception as e:
        plt.close()
        return None, f"Erro ao gerar gráfico: {str(e)}"

def load_data(file_path, df_state):
    if file_path is None or file_path == "":
        return "Por favor, faça o upload de um arquivo CSV para analisar.", pd.DataFrame(), None, None, None
    try:
        df = pd.read_csv(file_path)
        status_message = f"Arquivo carregado com sucesso! {df.shape[0]} linhas, {df.shape[1]} colunas."
        return status_message, df.head(), df, None, None
    except Exception as e:
        return f"Erro ao carregar arquivo: {str(e)}", pd.DataFrame(), None, None, None

def process_query(query, df_state):
    if df_state is None:
        return "Por favor, carregue um arquivo CSV primeiro.", "", "", None
    if not query:
        return "Por favor, digite uma pergunta.", "", "", None

    qp = build_query_pipeline(df_state)

    try:
        llm1_output = qp.run_modules({"input": query, "llm1": {}})["llm1"]
        code_expression, plot_suggestion = parse_llm_output(llm1_output.message.content)

        pandas_output = "N/A (Erro na execução)"
        execution_error = None
        try:
            # Create a temporary parser instance just for execution
            temp_parser = PandasInstructionParser(df_state)
            pandas_output = temp_parser.parse_response(code_expression)
        except Exception as e:
            pandas_output = f"Erro ao executar código: {str(e)}"
            execution_error = str(e)

        plot_path = None
        plot_error = None
        if plot_suggestion and execution_error is None: # Only try plotting if pandas execution was successful
            plot_path, plot_error = generate_and_save_plot(df_state, plot_suggestion)
            if plot_error:
                 print(f"Erro na plotagem: {plot_error}") # Log plot error server-side
                 plot_path = None # Ensure plot_path is None if there was an error


        plot_status_message = ""
        if plot_path:
            plot_status_message = "Um gráfico foi gerado."
        elif plot_error:
             plot_status_message = f"Sugestão de gráfico recebida, mas falhou: {plot_error}"
        elif plot_suggestion:
            plot_status_message = f"Sugestão de gráfico recebida ('{plot_suggestion}'), mas não foi possível gerar."

        response_synthesis_inputs = {
            "query_str": query,
            "pandas_output": str(pandas_output)[:1000], # Limit output size for prompt
            "plot_status": plot_status_message # Pass plot status to synthesizer
        }

        response_synthesizer_output = qp.run_modules(
            {"response_synthesis_prompt": response_synthesis_inputs, "llm2": {}}
        )["llm2"]

        final_response = response_synthesizer_output.message.content

        # Add the generated code at the end as requested (separated from synthesis)
        final_response_with_code = f"{final_response}\n\nO código utilizado foi `{code_expression}`"


        return final_response_with_code, code_expression, plot_path, None

    except Exception as e:
        return f"Ocorreu um erro geral durante o processamento: {str(e)}", "", None, None

def add_history(query, response, code, plot_path, history_state):
    if query and (response or plot_path):
        history_state.append({"query": query, "response": response, "code": code, "plot_path": plot_path})
        gr.Info("Adicionado ao histórico do PDF!", duration=2)
    else:
        gr.Warning("Nenhuma pergunta ou resposta/gráfico válido para adicionar.", duration=2)
    return history_state

def generate_pdf(history_state):
    if not history_state:
        return "Nenhum dado para adicionar ao PDF.", None

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_path = f"relatorio_analise_{timestamp}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório de Análise de Dados", 0, 1, 'C')
    pdf.ln(10)

    for i, item in enumerate(history_state):
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 8, txt=f"Pergunta {i+1}: {item['query']}")
        pdf.ln(2)

        if item['response']:
            pdf.set_font("Arial", '', 10)
            # Basic formatting: replace triple backticks with code block simulation
            response_text = item['response'].replace('```python', '\n[Código Python Gerado]\n').replace('```', '\n[/Código]\n')
            pdf.multi_cell(0, 6, txt=response_text.strip())
            pdf.ln(2)

        if item['plot_path'] and os.path.exists(item['plot_path']):
            try:
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 5, "Gráfico Gerado:", 0, 1)
                pdf.ln(1)
                # Add image, scaling it down to fit within page width (approx 180 units)
                # Adjust width (w) as needed, height (h) will be auto-calculated to maintain aspect ratio
                pdf.image(item['plot_path'], x=pdf.get_x(), w=150)
                pdf.ln(5) # Add space after image
            except Exception as e:
                 print(f"Erro ao adicionar imagem {item['plot_path']} ao PDF: {e}")
                 pdf.set_font("Arial", 'I', 10)
                 pdf.cell(0, 5, f"[Erro ao incluir gráfico]", 0, 1)
                 pdf.ln(5)


        pdf.ln(5) # Add space after each entry

    try:
        pdf.output(pdf_path)
        return pdf_path, None
    except Exception as e:
        return None, f"Erro ao salvar PDF: {str(e)}"


def clear_query_results():
    return "", "", None, None

def reset_application():
    # Clean up old plot files if any exist
    for file_name in os.listdir('.'):
        if file_name.startswith('plot_') and file_name.endswith('.png'):
            try:
                os.remove(file_name)
            except OSError as e:
                print(f"Erro ao remover arquivo antigo {file_name}: {e}")

    return None, "A aplicação foi resetada. Por favor, faça upload de um novo arquivo CSV.", pd.DataFrame(), "", "", None, None, [], None

with gr.Blocks(theme='Soft') as app:

    gr.Markdown("# Analisando os dados🔎🎲")

    gr.Markdown('''
    Carregue um arquivo CSV e faça perguntas sobre os dados. A cada pergunta, você poderá
    visualizar a resposta, o código Pandas gerado e, se aplicável, um gráfico.
    Clique em "Adicionar ao histórico do PDF" para incluir a interação (pergunta, resposta, código e gráfico)
    no relatório final. Para fazer uma nova pergunta, clique em "Limpar pergunta e resultado".
    Após definir as interações no histórico, clique em "Gerar PDF" para baixar o relatório completo.
    Para analisar um novo dataset, clique em "Quero analisar outro dataset" ao final da página.
    ''')

    with gr.Row():
        input_arquivo = gr.File(file_count="single", type="filepath", label="Upload CSV")
        upload_status = gr.Textbox(label="Status do Upload:", interactive=False)

    gr.Markdown("### Primeiras linhas do DataFrame")
    tabela_dados = gr.DataFrame(interactive=False)

    gr.Markdown("""
    **Exemplos de perguntas:**
    1. Qual é o número de registros?
    2. Liste os nomes das colunas.
    3. Mostre as estatísticas descritivas para colunas numéricas.
    4. Qual a média da coluna 'sua_coluna_numerica'?
    5. Quantos valores únicos existem na coluna 'sua_coluna_categorica'?
    6. Mostre um histograma para a coluna 'sua_coluna_numerica'.
    7. Mostre um gráfico de barras para a coluna 'sua_coluna_categorica'.
    """)

    input_pergunta = gr.Textbox(label="Digite sua pergunta sobre os dados")

    botao_submeter = gr.Button("Enviar")

    gr.Markdown("### Resultado")
    output_resposta = gr.Textbox(label="Resposta do Analista:", interactive=False)
    output_code = gr.Textbox(label="Código Pandas Gerado:", interactive=False)
    output_plot = gr.Image(label="Gráfico Gerado:", visible=False)

    with gr.Row():
        botao_limpeza = gr.Button("Limpar Pergunta e Resultado")
        botao_add_pdf = gr.Button("Adicionar ao Histórico do PDF")

    gr.Markdown("### Relatório PDF")
    botao_gerar_pdf = gr.Button("Gerar PDF")
    arquivo_pdf = gr.File(label="Download do PDF")


    botao_resetar = gr.Button("Quero analisar outro dataset!")

    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])
    latest_code_state = gr.State(value="") # To store code for PDF
    latest_plot_path_state = gr.State(value=None) # To store plot path for PDF


    input_arquivo.change(
        fn=load_data,
        inputs=[input_arquivo, df_estado],
        outputs=[upload_status, tabela_dados, df_estado, output_resposta, output_plot] # Clear previous results on new upload
    )

    botao_submeter.click(
        fn=process_query,
        inputs=[input_pergunta, df_estado],
        outputs=[output_resposta, output_code, output_plot, latest_plot_path_state] # plot_path is stored in state
    ).success( # Chain to show the image component if a plot path is returned
        fn=lambda x: gr.Image(visible=x is not None),
        inputs=[latest_plot_path_state],
        outputs=[output_plot]
    )


    botao_limpeza.click(
        fn=clear_query_results,
        inputs=[],
        outputs=[input_pergunta, output_resposta, output_code, output_plot]
    )

    botao_add_pdf.click(
        fn=add_history,
        inputs=[input_pergunta, output_resposta, output_code, latest_plot_path_state, historico_estado],
        outputs=historico_estado
    )

    botao_gerar_pdf.click(
        fn=generate_pdf,
        inputs=[historico_estado],
        outputs=arquivo_pdf
    )

    botao_resetar.click(
        fn=reset_application,
        inputs=[],
        outputs=[input_arquivo, upload_status, tabela_dados, input_pergunta, output_resposta, output_code, output_plot, historico_estado, latest_plot_path_state]
    )

if __name__ == "__main__":
    app.launch()