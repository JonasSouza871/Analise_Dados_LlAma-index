from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent, CustomQueryComponent
from llama_index.core.llms import ChatMessage
from llama_index.core.component_specs import ComponentSpec
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg') # ESSENCIAL: Usar backend não interativo para Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
import traceback
from typing import Any, Dict

# --- Custom Component para extrair código Python do LLM1 ---
class PandasCodeExtractor(CustomQueryComponent):
    """
    Extrai código Python de uma string que pode também conter uma linha 'Gráfico:'.
    A linha 'Gráfico:' é preservada em 'original_instructions'.
    O código Python extraído (ou 'None') é fornecido em 'python_code_str'.
    """

    def _validate_component_inputs(self, input_dict: dict) -> dict:
        return input_dict

    @property
    def _input_keys(self) -> set:
        return {"llm_output_str"}

    @property
    def _output_keys(self) -> set:
        return {"python_code_str", "original_instructions"}

    def _run_component(self, **kwargs) -> dict:
        input_data = kwargs["llm_output_str"]
        
        if isinstance(input_data, ChatMessage): # Saída comum de LLMs em LlamaIndex
            input_str = input_data.content
        elif isinstance(input_data, str):
            input_str = input_data
        else:
            # Tentar converter para string se for outro tipo (ex: dict de resposta)
            if hasattr(input_data, 'text'): # Como algumas respostas de LLM
                input_str = input_data.text
            else:
                input_str = str(input_data)
        
        input_str = input_str.strip() # Limpar espaços extras

        lines = input_str.split('\n')
        python_code_to_eval = ""
        
        if lines:
            first_line_cleaned = lines[0].strip().lower()
            if first_line_cleaned.startswith("gráfico:") or first_line_cleaned.startswith("`gráfico:"): # Lidar com markdown ocasional
                if len(lines) > 1:
                    python_code_to_eval = "\n".join(lines[1:]).strip()
                else:
                    python_code_to_eval = "None" # Conforme instruído ao LLM
            else:
                python_code_to_eval = input_str # Nenhuma linha "Gráfico:", assume que tudo é código Python
        
        if not python_code_to_eval.strip():
            python_code_to_eval = "None"
            
        return {"python_code_str": python_code_to_eval, "original_instructions": input_str}

# --- Parser Pandas com especificações de componente mais explícitas ---
class PatchedPandasInstructionParser(PandasInstructionParser):
    def _validate_component_inputs(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "query_str" not in input_dict:
            raise ValueError("Input 'query_str' não encontrado para PandasInstructionParser.")
        return input_dict
    
    @property
    def _input_keys(self) -> set:
        return {"query_str"}

    @property
    def _output_keys(self) -> set:
        return {"result"}

    def _run_component(self, *, query_str: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            val = eval(query_str, {"pd": pd, "df": self.df}, {})
        except Exception as e_eval:
            try:
                compile(query_str, "<string>", "eval")
            except SyntaxError as e_syntax:
                error_message = f"Erro de sintaxe no código Python gerado: {e_syntax}. Código problemático: '{query_str}'"
                print(error_message)
                raise ValueError(error_message) from e_syntax
            error_message = f"Erro ao executar o código Python: {e_eval}. Código problemático: '{query_str}'"
            print(error_message)
            raise ValueError(error_message) from e_eval
        return {"result": val}

# --- Configuração Inicial ---
api_key = os.getenv("secret_key")
if not api_key:
    raise ValueError("Por favor, defina a variável de ambiente 'secret_key' com sua API Key do Groq.")
llm = Groq(model="llama3-70b-8192", api_key=api_key)

THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="gray", neutral_hue="slate"
).set(
    body_background_fill="white", body_background_fill_dark="#1a1a1a",
    block_background_fill="#f9fafb", block_background_fill_dark="#2d2d2d",
    button_primary_background_fill="*primary_600", button_primary_text_color="white",
    button_secondary_background_fill="*neutral_200", button_secondary_text_color="*neutral_800",
)

def descricao_colunas(df):
    desc = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Detalhes das colunas do dataframe:\n" + desc

def plotar_grafico(df, tipo_grafico, x_col=None, y_col=None, hue_col=None, title="Gráfico"):
    print(f"PLOTAR_GRAFICO: tipo={tipo_grafico}, x='{x_col}', y='{y_col}', hue='{hue_col}'")
    print(f"PLOTAR_GRAFICO: Colunas disponíveis: {df.columns.tolist()}")
    fig = None
    try:
        if x_col and x_col not in df.columns:
            return None, f"Erro: Coluna X '{x_col}' não encontrada. Disponíveis: {df.columns.tolist()}"
        if y_col and y_col not in df.columns:
            return None, f"Erro: Coluna Y '{y_col}' não encontrada. Disponíveis: {df.columns.tolist()}"
        if hue_col and hue_col is not None and hue_col not in df.columns:
            return None, f"Erro: Coluna Hue '{hue_col}' não encontrada. Disponíveis: {df.columns.tolist()}"

        fig = plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")

        if tipo_grafico == "bar": sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
        elif tipo_grafico == "scatter": sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, size=hue_col if hue_col else None)
        elif tipo_grafico == "line": sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
        elif tipo_grafico == "box": sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
        else:
            if fig: plt.close(fig)
            return None, "Tipo de gráfico não suportado. Use 'bar', 'scatter', 'line' ou 'box'."

        plt.title(title if title else f"{tipo_grafico.capitalize()} de {y_col or 'dados'} por {x_col or 'dados'}", fontsize=14, pad=15)
        if x_col: plt.xlabel(x_col, fontsize=12)
        if y_col: plt.ylabel(y_col, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        if fig: plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}", None
    except Exception as e:
        if fig: plt.close(fig)
        print(f"PLOTAR_GRAFICO: Exceção detalhada: {e}")
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
        "   b. Gere uma string na PRIMEIRA LINHA no formato EXATO: 'Gráfico: tipo={tipo_grafico}, x={coluna_x}, y={coluna_y}, hue={coluna_hue}'.\n"
        "      - {tipo_grafico} pode ser 'bar', 'scatter', 'line', 'box'.\n"
        "      - {coluna_x}, {coluna_y} devem ser nomes de colunas existentes em `df`.\n"
        "      - {coluna_hue} deve ser um nome de coluna existente em `df` ou a palavra literal 'None' se não aplicável.\n"
        "   c. Se a consulta pedir um gráfico E também uma manipulação de dados (ex: 'gráfico da média de vendas por região'):\n"
        "      Forneça a string 'Gráfico:...' na PRIMEIRA LINHA.\n"
        "      Na SEGUNDA LINHA, forneça a expressão Python para calcular os dados textuais (ex: `df.groupby('regiao')['vendas'].mean()`).\n"
        "      Exemplo de saída para 'gráfico de barras da soma de vendas por produto':\n"
        "      Gráfico: tipo=bar, x=Produto, y=Vendas, hue=None\n"
        "      df.groupby('Produto')['Vendas'].sum()\n"
        "   d. Se a consulta pedir APENAS um gráfico de colunas existentes (ex: 'gráfico de dispersão de Preço vs Quantidade'):\n"
        "      Forneça a string 'Gráfico: tipo=scatter, x=Preço, y=Quantidade, hue=None' na PRIMEIRA LINHA.\n"
        "      Na SEGUNDA LINHA, forneça a expressão Python `None`. (Literalmente a palavra None, sem aspas)\n"
        "      Exemplo de saída:\n"
        "      Gráfico: tipo=scatter, x=Preço, y=Quantidade, hue=None\n"
        "      None\n"
        "4. A SEGUNDA LINHA da sua saída (ou a única linha se não houver gráfico e nem a string 'Gráfico:') é o que será avaliado como código Python. Certifique-se de que seja uma expressão Python válida.\n"
        "5. Certifique-se que os nomes das colunas na string 'Gráfico:' correspondem EXATAMENTE aos nomes das colunas no dataframe `df`."
    )
    pandas_prompt_str = (
        "Você está trabalhando com um dataframe do pandas chamado `df`.\n"
        "Detalhes das colunas (nome: tipo):\n{colunas_detalhes}\n\n"
        "Resultado de `print(df.head())`:\n"
        "{df_str}\n\n"
        "Siga estas instruções DETALHADAMENTE:\n{instruction_str}\n\n"
        "Consulta do Usuário: {query_str}\n\n"
        "Sua Resposta (string 'Gráfico:...' na primeira linha se aplicável, e expressão Python na segunda linha):"
    )
    response_synthesis_prompt_str = (
        "Dada uma pergunta de entrada, atue como analista de dados e elabore uma resposta a partir dos resultados da consulta.\n"
        "Responda de forma natural e concisa, sem introduções como 'A resposta é:'.\n"
        "Consulta Original: {query_str}\n\n"
        "Instruções/Código Gerado para Pandas (saída completa do passo anterior, pode conter 'Gráfico:...' e/ou uma expressão Python):\n{pandas_instructions}\n\n"
        "Resultado da Execução da Expressão Python (se houver, caso contrário será o objeto Python None ou um erro):\n{pandas_output}\n\n"
        "Sua Resposta Final para o Usuário:\n"
        "Se `pandas_instructions` continha 'Gráfico: ...', REPITA EXATAMENTE essa string 'Gráfico: ...' na sua resposta final para o usuário. NÃO a modifique.\n"
        "Se `pandas_output` for um objeto (como um Axes de Matplotlib), NÃO o inclua diretamente. Descreva o resultado textual com base em `pandas_output` (se não for None e não for apenas o resultado da expressão 'None').\n"
        "Ao final, se uma expressão Python foi efetivamente executada e produziu um resultado significativo (verifique se `pandas_instructions` continha código Python além de 'Gráfico:...' e 'None', e se `pandas_output` não é apenas o resultado de `None`), extraia essa expressão de `pandas_instructions` (a parte que não é 'Gráfico:...' e não é 'None') e exiba-a no formato: 'O código utilizado para a análise textual foi: `{codigo_python}`'."
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5), colunas_detalhes=descricao_colunas(df)
    )
    pandas_output_parser_component = PatchedPandasInstructionParser(df=df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    qp = QP(
        modules={
            "input": InputComponent(), "pandas_prompt": pandas_prompt, "llm1": llm,
            "code_extractor": PandasCodeExtractor(),
            "pandas_output_parser": pandas_output_parser_component,
            "response_synthesis_prompt": response_synthesis_prompt, "llm2": llm,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1"])
    qp.add_link("llm1", "code_extractor", dest_key="llm_output_str")
    qp.add_link("code_extractor", "pandas_output_parser", src_key="python_code_str", dest_key="query_str")
    qp.add_links([
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link("code_extractor", "response_synthesis_prompt", src_key="original_instructions", dest_key="pandas_instructions"),
        Link("pandas_output_parser", "response_synthesis_prompt", src_key="result", dest_key="pandas_output"),
    ])
    qp.add_link("response_synthesis_prompt", "llm2")
    return qp

def carregar_dados(caminho_arquivo_obj, df_estado_atual): # Renomeado para clareza
    if not caminho_arquivo_obj:
        return "Por favor, faça o upload de um arquivo.", pd.DataFrame(), df_estado_atual
    caminho_arquivo = caminho_arquivo_obj.name # Obter o caminho do arquivo do objeto Gradio
    try:
        ext = os.path.splitext(caminho_arquivo)[1].lower()
        if ext == '.csv': df = pd.read_csv(caminho_arquivo)
        elif ext in ['.xlsx', '.xls']: df = pd.read_excel(caminho_arquivo)
        elif ext == '.xlsb': df = pd.read_excel(caminho_arquivo, engine='pyxlsb')
        else:
            return "Formato de arquivo não suportado. Use CSV, Excel (.xlsx, .xls) ou Excel Binário (.xlsb).", pd.DataFrame(), df_estado_atual
        return "Arquivo carregado com sucesso!", df.head(), df
    except Exception as e:
        print(f"CARREGAR_DADOS: Erro: {e}")
        traceback.print_exc()
        return f"Erro ao carregar arquivo: {str(e)}", pd.DataFrame(), df_estado_atual

def processar_pergunta(pergunta, df_estado):
    if df_estado is None or df_estado.empty:
         return "Por favor, carregue um arquivo CSV ou Excel primeiro.", None
    if not pergunta:
        return "Por favor, faça uma pergunta.", None
    try:
        qp = pipeline_consulta(df_estado)
        raw_response_obj = qp.run(query_str=pergunta)
        print(f"PROCESSAR_PERGUNTA: Tipo de objeto de resposta do pipeline: {type(raw_response_obj)}")
        print(f"PROCESSAR_PERGUNTA: Objeto de resposta completo: {raw_response_obj}")

        if hasattr(raw_response_obj, 'response'): resposta_texto = str(raw_response_obj.response)
        elif hasattr(raw_response_obj, 'message') and hasattr(raw_response_obj.message, 'content'):
            resposta_texto = str(raw_response_obj.message.content)
        elif isinstance(raw_response_obj, str): resposta_texto = raw_response_obj
        else: resposta_texto = str(raw_response_obj)
        print(f"PROCESSAR_PERGUNTA: Resposta textual extraída para processamento: {resposta_texto}")

        img_data = None
        grafico_info_match = re.search(
            r"Gráfico:\s*tipo=([\w-]+),\s*x=([^,]+),\s*y=([^,]+),\s*hue=([^,\n]+)",
            resposta_texto, re.IGNORECASE
        )
        if grafico_info_match:
            print("PROCESSAR_PERGUNTA: Padrão 'Gráfico:' encontrado.")
            tipo, x_col, y_col, hue_col = [g.strip() for g in grafico_info_match.groups()]
            print(f"PROCESSAR_PERGUNTA: Info gráfico: tipo={tipo}, x='{x_col}', y='{y_col}', hue='{hue_col}'")
            if hue_col.lower() == "none": hue_col = None
            img_data, erro_grafico = plotar_grafico(df_estado, tipo, x_col, y_col, hue_col, title=f"Gráfico: {pergunta}")
            if erro_grafico:
                resposta_texto += f"\n\nAVISO DE GRÁFICO: {erro_grafico}"
                img_data = None
                print(f"PROCESSAR_PERGUNTA: Erro ao gerar gráfico: {erro_grafico}")
            else:
                print("PROCESSAR_PERGUNTA: Gráfico gerado (base64).")
        else:
            print("PROCESSAR_PERGUNTA: Padrão 'Gráfico:' NÃO encontrado.")
        return resposta_texto, img_data
    except Exception as e:
        print(f"PROCESSAR_PERGUNTA: Erro detalhado: {e}")
        traceback.print_exc()
        return f"Erro no pipeline: {str(e)}", None

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
    # Assume que os arquivos de fonte estão no mesmo diretório do script
    # O __file__ pode não funcionar corretamente se o script for empacotado (ex: PyInstaller)
    # mas para execução direta é o padrão.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "DejaVuSansCondensed.ttf")
    font_bold_path = os.path.join(script_dir, "DejaVuSansCondensed-Bold.ttf")

    try:
        if not os.path.exists(font_path):
            raise RuntimeError(f"Arquivo de fonte NÃO ENCONTRADO: {font_path}. Certifique-se de que 'DejaVuSansCondensed.ttf' está no mesmo diretório do script.")
        if not os.path.exists(font_bold_path):
            raise RuntimeError(f"Arquivo de fonte NÃO ENCONTRADO: {font_bold_path}. Certifique-se de que 'DejaVuSansCondensed-Bold.ttf' está no mesmo diretório do script.")
        
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.add_font("DejaVu", "B", font_bold_path, uni=True)
    except RuntimeError as e:
        print(f"GERAR_PDF: Erro ao adicionar fontes: {e}")
        traceback.print_exc()
        return None, gr.Error(f"Erro de fonte PDF: {e}. Verifique os arquivos .ttf.")

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
        print(f"GERAR_PDF: PDF gerado com sucesso em {caminho_pdf}")
        return caminho_pdf, gr.Info("PDF gerado com sucesso!")
    except Exception as e:
        print(f"GERAR_PDF: Erro ao escrever PDF: {str(e)}")
        traceback.print_exc()
        return None, gr.Error(f"Erro ao gerar PDF: {str(e)}")

def limpar_pergunta_resposta():
    return "", "", None

def resetar_aplicacao():
    return None, "Aplicação resetada.", pd.DataFrame(), "", None, [], "", None

# --- Interface Gradio ---
with gr.Blocks(theme=THEME, css="""
    .gr-button {margin: 5px;} .gr-textbox {border-radius: 5px;}
    #title {text-align: center; padding: 20px; color: #1e3a8a;}
    #subtitle {text-align: center; color: #4b5563; margin-bottom: 20px;}
    .gr-box {border-radius: 10px; padding: 15px;}
""") as app:
    gr.Markdown("# DataInsight Pro 📈🔍", elem_id="title")
    gr.Markdown("Explore seus dados...", elem_id="subtitle")

    df_estado = gr.State(value=pd.DataFrame())
    historico_estado = gr.State(value=[])

    with gr.Tabs():
        with gr.Tab("Carregar Dados"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_arquivo = gr.File(label="Upload de Arquivo (CSV, Excel)")
                    upload_status = gr.Textbox(label="Status do Upload", interactive=False)
                with gr.Column(scale=2):
                    tabela_dados = gr.DataFrame(label="Prévia dos Dados", interactive=False)
        with gr.Tab("Análise de Dados"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Faça sua Pergunta")
                    input_pergunta = gr.Textbox(label="Pergunta", placeholder="Ex: 'Qual a média ...'", lines=2)
                    botao_submeter = gr.Button("Analisar", variant="primary")
                    gr.Markdown("**Dicas...**\n- ...\n- ...")
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

    input_arquivo.change(carregar_dados, [input_arquivo, df_estado], [upload_status, tabela_dados, df_estado])
    botao_submeter.click(processar_pergunta, [input_pergunta, df_estado], [output_resposta, output_grafico])
    botao_limpeza.click(limpar_pergunta_resposta, [], [input_pergunta, output_resposta, output_grafico])
    botao_add_pdf.click(add_historico, [input_pergunta, output_resposta, historico_estado], [historico_estado, upload_status])
    botao_gerar_pdf.click(gerar_pdf, [historico_estado], [arquivo_pdf, upload_status])
    botao_resetar.click(resetar_aplicacao, [], [input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta, output_grafico])

if __name__ == "__main__":
    print("Iniciando aplicação DataInsight Pro...")
    print(f"Verificando fontes PDF no diretório: {os.path.dirname(os.path.abspath(__file__))}")
    app.launch()