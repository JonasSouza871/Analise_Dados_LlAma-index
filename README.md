# 📊 Analisador de Dados CSV/Excel com IA

Uma aplicação web que permite carregar arquivos CSV/Excel, fazer perguntas em linguagem natural sobre os dados e gerar relatórios PDF.

## 🚀 Funcionalidades

- Suporte a CSV e Excel (.xlsx, .xls, .xlsm, .xlsb)
- Consultas em linguagem natural usando IA
- Estatísticas descritivas automáticas
- Geração de relatórios PDF
- Interface web intuitiva

## 🔧 Tecnologias

- Python, Gradio, Pandas, LlamaIndex, Groq (Llama3-70B)

## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/analisador-dados-ia.git
cd analisador-dados-ia

# Instale as dependências
pip install gradio pandas llama-index llama-index-llms-groq llama-index-experimental fpdf2 openpyxl xlrd python-dotenv

# Configure a API Key do Groq
echo "secret_key=sua_chave_groq_aqui" > .env

# Execute a aplicação
python app.py
```





