from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import base64
import re
import traceback

api_key = os.getenv("secret_key")
llm = Groq(model="llama3-70b-8192", api_key=api_key)

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

def descricao_colunas(df):
    desc = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Detalhes das colunas do dataframe:\n" + desc

def plotar_grafico(df, tipo_grafico, x_col=None, y_col=None, hue_col=None, title="Gráfico"):
    print(f"Gerando gráfico: tipo={tipo_grafico}, x='{x_col}', y='{y_col}', hue='{hue_col}'")
    print(f"Colunas disponíveis no DataFrame: {df.columns.tolist()}")

    try:
        if x_col and x_col not in df.columns:
            plt.close()
            return None, f"Erro: Coluna X '{x_col}' não encontrada no DataFrame. Colunas disponíveis: {df.columns.tolist()}"
        if y_col and y_col not in df.columns:
            plt.close()
            return None, f"Erro: Coluna Y '{y_col}' não encontrada no DataFrame. Colunas disponíveis: {df.columns.tolist()}"
        if hue_col and hue_col is not None and hue_col not in df.columns:
            plt.close()
            return None, f"Erro: Coluna Hue '{hue_col}' não encontrada no DataFrame. Colunas disponíveis: {df.columns.tolist()}"

        fig = plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")

        if tipo_grafico == "bar":
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
        elif tipo_grafico == "scatter":
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, size=hue_col if hue_col else None)
        elif tipo_grafico == "line":
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
        elif tipo_grafico == "box":
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
        else:
            plt.close(fig)
            return None, "Tipo de gráfico não suportado. Use 'bar', 'scatter', 'line' ou 'box'."

        plt.title(title if title else f"{tipo_grafico.capitalize()} de {y_col if y_col else 'dados'} por {x_col if x_col else 'dados'}", fontsize=14, pad=15)
        if x_col: plt.xlabel(x_col, fontsize=12)
        if y_col: plt.ylabel(y_col, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}", None
    except Exception as e:
        if 'fig' in locals() and fig is not None:
             plt.close(fig)
        print(f"Exceção detalhada em plotar_grafico: {e}")
        traceback.print_exc()
        return None, f"Erro ao gerar gráfico: {str(e)}"

def pipeline_consulta(df):
    instruction_str = (
        "INSTRUÇÕES IMPORTANTES:\n"
        "1. Analise a consulta do usuário.\n"
        "2. Se a consulta NÃO pedir um gráfico:\n"
        "   a. Converta a consulta para uma ÚNICA linha de código Python executável usando Pandas (o dataframe é `df`).\n"
        "   b. A linha final DEVE ser uma expressão Python que possa ser chamada com `eval()`.\n"
        "   c. IMPRIMA APENAS ESTA EXPRESSÃO. Não adicione comentários Python (#) nem texto explicativo.\n"
        "   d. Não coloque a expressão entre aspas ou ```python ... ```.\n"
        "3. Se a consulta PEDIR um gráfico:\n"
        "   a. NÃO gere código Python para plotar o gráfico (ex: NADA de `df.plot()` ou `sns.plot()`).\n"
        "   b. Em vez disso, gere uma string no formato EXATO: 'Gráfico: tipo={tipo_grafico}, x={coluna_x}, y={coluna_y}, hue={coluna_hue}'.\n"
        "      - {tipo_grafico} pode ser 'bar', 'scatter', 'line', 'box'.\n"
        "      - {coluna_x}, {coluna_y} devem ser nomes de colunas existentes em `df`.\n"
        "      - {coluna_hue} deve ser um nome de coluna existente em `df` ou a palavra literal 'None' se não aplicável.\n"
        "   c. Se a consulta pedir um gráfico E também uma manipulação de dados (ex: 'gráfico da média de vendas por região'),\n"
        "      você DEVE fornecer AMBOS: a string 'Gráfico:...' E, em uma NOVA LINHA, a expressão Python para calcular os dados textuais (ex: `df.groupby('regiao')['vendas'].mean()`).\n"
        "      A string 'Gráfico:...' deve vir PRIMEIRO.\n"
        "      Exemplo de saída para 'gráfico de barras da soma de vendas por produto':\n"
        "      Gráfico: tipo=bar, x=Produto, y=Vendas, hue=None\n"
        "      df.groupby('Produto')['Vendas'].sum()\n"
        "   d. Se a consulta pedir APENAS um gráfico de colunas existentes (ex: 'gráfico de dispersão de Preço vs Quantidade'):\n"
        "      Forneça APENAS a string 'Gráfico: tipo=scatter, x=Preço, y=Quantidade, hue=None'. Nenhuma expressão Python adicional é necessária.\n"
        "5. Certifique-se que os nomes das colunas na string 'Gráfico:' correspondem EXATAMENTE aos nomes das colunas no dataframe `df`."
    )

    pandas_prompt_str = (
        "Você está trabalhando com um dataframe do pandas chamado `df`.\n"
        "Detalhes das colunas (nome: tipo):\n{colunas_detalhes}\n\n"
        "Resultado de `print(df.head())`:\n"
        "{df_str}\n\n"
        "Siga estas instruções DETALHADAMENTE:\n{instruction_str}\n\n"
        "Consulta do Usuário: {query_str}\n\n"
        "Sua Resposta (string 'Gráfico:...' e/ou expressão Python):"
    )

    response_synthesis_prompt_str = (
        "Dada uma pergunta de entrada, atue como analista de dados e elabore uma resposta a partir dos resultados da consulta.\n"
        "Responda de forma natural e concisa, sem introduções como 'A resposta é:' ou algo semelhante.\n"
        "Consulta Original: {query_str}\n\n"
        "Instruções/Código Gerado para Pandas (pode conter 'Gráfico:...' e/ou uma expressão Python):\n{pandas_instructions}\n\n"
        "Resultado da Execução da Expressão Python (se houver, caso contrário será 'None' ou um erro):\n{pandas_output}\n\n"
        "Sua Resposta Final para o Usuário:\n"
        "Se `pandas_instructions` continha 'Gráfico: ...', REPITA EXATAMENTE essa string 'Gráfico: ...' na sua resposta final para o usuário. NÃO a modifique.\n"
        "Se `pandas_output` for um objeto (como um Axes de Matplotlib), NÃO o inclua diretamente. Descreva o resultado textual com base em `pandas_output`.\n"
        "Ao final, se uma expressão Python foi usada (vista em `pandas_instructions` e não começando com 'Gráfico:'), exiba o código usado para gerar a parte textual da resposta, no formato: 'O código utilizado para a análise textual foi: `{codigo_python}`'. Se `pandas_instructions` continha apenas 'Gráfico:...' ou não continha código Python para `eval`, não inclua esta frase sobre o código."
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

def carregar_dados(caminho_arquivo, df_estado):
    if not caminho_arquivo:
        return "Por favor, faça o upload de um arquivo CSV ou Excel.", pd.DataFrame(), df_estado

    try:
        ext = os.path.splitext(caminho_arquivo.name)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(caminho_arquivo.name)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(caminho_arquivo.name)
        elif ext == '.xlsb':
            df = pd.read_excel(caminho_arquivo.name, engine='pyxlsb')
        else:
            return "Formato de arquivo não suportado. Use CSV, Excel (.xlsx, .xls) ou Excel Binário (.xlsb).", pd.DataFrame(), df_estado
        return "Arquivo carregado com sucesso!", df.head(), df
    except Exception as e:
        return f"Erro ao carregar arquivo: {str(e)}", pd.DataFrame(), df_estado

def processar_pergunta(pergunta, df_estado):
    if df_estado is not None and not df_estado.empty and pergunta:
        try:
            qp = pipeline_consulta(df_estado)
            raw_response_obj = qp.run(query_str=pergunta)

            print(f"Tipo de objeto de resposta do pipeline: {type(raw_response_obj)}")
            print(f"Objeto de resposta completo: {raw_response_obj}")

            if hasattr(raw_response_obj, 'response'):
                resposta_texto = str(raw_response_obj.response)
            elif hasattr(raw_response_obj, 'message') and hasattr(raw_response_obj.message, 'content'):
                resposta_texto = str(raw_response_obj.message.content)
            elif isinstance(raw_response_obj, str):
                resposta_texto = raw_response_obj
            else:
                resposta_texto = str(raw_response_obj)

            print(f"Resposta textual extraída para processamento: {resposta_texto}")

            img_data = None

            grafico_info_match = re.search(
                r"Gráfico:\s*tipo=([\w-]+),\s*x=([^,]+),\s*y=([^,]+),\s*hue=([^,\n]+)",
                resposta_texto,
                re.IGNORECASE
            )

            if grafico_info_match:
                print("Padrão 'Gráfico:' encontrado na resposta_texto.")
                tipo, x_col, y_col, hue_col = [g.strip() for g in grafico_info_match.groups()]

                print(f"Informações do gráfico extraídas: tipo={tipo}, x='{x_col}', y='{y_col}', hue='{hue_col}'")

                if hue_col.lower() == "none":
                    hue_col = None

                img_data, erro_grafico = plotar_grafico(df_estado, tipo, x_col, y_col, hue_col, title=f"Gráfico: {pergunta}")
                if erro_grafico:
                    resposta_texto += f"\n\nAVISO DE GRÁFICO: {erro_grafico}"
                    img_data = None
                    print(f"Erro ao gerar gráfico: {erro_grafico}")
                else:
                    print("Gráfico gerado com sucesso (dados base64).")
            else:
                print("Padrão 'Gráfico:' NÃO encontrado na resposta_texto.")

            return resposta_texto, img_data
        except Exception as e:
            print(f"Erro detalhado no processar_pergunta: {e}")
            traceback.print_exc()
            return f"Erro no pipeline: {str(e)}", None
    elif df_estado is None or df_estado.empty:
         return "Por favor, carregue um arquivo CSV ou Excel primeiro.", None
    return "Por favor, faça uma pergunta.", None

def add_historico(pergunta, resposta_texto, historico_estado):
    if pergunta and resposta_texto:
        if not historico_estado or historico_estado[-1] != (pergunta, resposta_texto):
            historico_estado.append((pergunta, resposta_texto))
            return historico_estado, gr.Info("Adicionado ao histórico do PDF!")
        else:
            return historico_estado, gr.Info("Já está no histórico.")
    return historico_estado, gr.Warning("Nenhuma pergunta/resposta para adicionar.")

def gerar_pdf(historico_estado):
    if not historico_estado:
        return None, gr.Warning("Nenhum dado para gerar o PDF.")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    caminho_pdf = f"relatorio_analise_{timestamp}.pdf"

    pdf = FPDF()
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSansCondensed.ttf")
    font_bold_path = os.path.join(os.path.dirname(__file__), "DejaVuSansCondensed-Bold.ttf")

    try:
        if not os.path.exists(font_path):
             raise RuntimeError(f"Arquivo de fonte não encontrado: {font_path}")
        if not os.path.exists(font_bold_path):
             raise RuntimeError(f"Arquivo de fonte não encontrado: {font_bold_path}")
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.add_font("DejaVu", "B", font_bold_path, uni=True)
    except RuntimeError as e:
        print(f"Erro ao adicionar fontes: {e}. Verifique se os arquivos .ttf (DejaVuSansCondensed.ttf, DejaVuSansCondensed-Bold.ttf) estão no diretório do script.")
        return None, gr.Error(f"Erro de fonte PDF: {e}. Certifique-se de que os arquivos .ttf estão no mesmo diretório do script.")

    pdf.add_page()
    pdf.set_font("DejaVu", 'B', 16)

    try:
        pdf.cell(0, 10, "Relatório de Análise de Dados", ln=True, align='C')
        pdf.ln(10)

        for i, (pergunta, resposta) in enumerate(historico_estado, 1):
            pdf.set_font("DejaVu", 'B', 12)
            pdf.multi_cell(0, 8, f"Pergunta {i}: {pergunta}", ln=True)
            pdf.set_font("DejaVu", '', 11)
            pdf.multi_cell(0, 6, resposta)
            pdf.ln(5)

        pdf.output(caminho_pdf)
        return caminho_pdf, gr.Info("PDF gerado com sucesso!")
    except Exception as e:
        print(f"Erro detalhado ao gerar PDF: {str(e)}")
        traceback.print_exc()
        return None, gr.Error(f"Erro ao gerar PDF: {str(e)}")

def limpar_pergunta_resposta():
    return "", "", None

def resetar_aplicacao():
    return None, "Aplicação resetada. Faça upload de um novo arquivo.", pd.DataFrame(), "", None, [], "", None

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

    df_estado = gr.State(value=pd.DataFrame())
    historico_estado = gr.State(value=[])

    with gr.Tabs():
        with gr.Tab("Carregar Dados"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_arquivo = gr.File(
                        label="Upload de Arquivo (CSV, Excel ou Excel Binário)"
                    )
                    upload_status = gr.Textbox(label="Status do Upload", interactive=False)
                with gr.Column(scale=2):
                    tabela_dados = gr.DataFrame(label="Prévia dos Dados", interactive=False)

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
                    output_grafico = gr.Image(label="Gráfico")

            with gr.Row():
                botao_limpeza = gr.Button("Limpar", variant="secondary")
                botao_add_pdf = gr.Button("Adicionar ao Histórico", variant="secondary")
                botao_gerar_pdf = gr.Button("Gerar PDF", variant="primary")

            arquivo_pdf = gr.File(label="Download do PDF", interactive=False)

    botao_resetar = gr.Button("Novo Conjunto de Dados", variant="secondary")

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
        outputs=[historico_estado, upload_status]
    )
    botao_gerar_pdf.click(
        fn=gerar_pdf,
        inputs=[historico_estado],
        outputs=[arquivo_pdf, upload_status]
    )
    botao_resetar.click(
        fn=resetar_aplicacao,
        inputs=[],
        outputs=[input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta, output_grafico]
    )

if __name__ == "__main__":
    # Certifique-se de ter os arquivos DejaVuSansCondensed.ttf e DejaVuSansCondensed-Bold.ttf
    # no mesmo diretório que este script, ou ajuste os caminhos em gerar_pdf.
    app.launch()