# AI-Powered Data Analysis Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.x-purple.svg)

## Overview

This application provides an interactive web interface for analyzing CSV and Excel files using natural language queries powered by AI. Built with Gradio and LlamaIndex, it leverages the Groq API (Llama3-70B model) to process user questions about datasets and generate comprehensive PDF reports.

## Features

- **Multi-format Support**: Load CSV and Excel files (.xlsx, .xls, .xlsm, .xlsb)
- **Natural Language Queries**: Ask questions about your data in plain language
- **Automated Statistics**: Generate descriptive statistics and DataFrame information
- **PDF Report Generation**: Export your analysis and queries to professional PDF reports
- **Interactive Web Interface**: User-friendly Gradio-based UI
- **Flexible Encoding**: Automatic handling of multiple CSV encodings (UTF-8, Latin-1, ISO-8859-1, CP1252)

## Technologies Used

- **Python**: Core programming language
- **Gradio**: Web interface framework
- **Pandas**: Data manipulation and analysis
- **LlamaIndex**: LLM orchestration framework
- **Groq API**: LLM inference (Llama3-70B-8192 model)
- **FPDF**: PDF generation
- **OpenPyXL**: Excel file support
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

## Repository Structure

```
.
├── src/
│   └── app.py              # Main application file
├── docs/                   # Documentation files
├── assets/                 # Static assets and resources
├── examples/               # Example datasets and use cases
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com))

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-username/Analise_Dados_LlAma-index.git
cd Analise_Dados_LlAma-index
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your Groq API key:
   - Create a `.env` file in the root directory
   - Add your API key:
```
secret_key=your_groq_api_key_here
```

## Usage

1. Start the application:
```bash
python src/app.py
```

2. Open your browser and navigate to the local URL displayed (typically `http://127.0.0.1:7860`)

3. Upload a CSV or Excel file

4. Ask questions about your data or view descriptive statistics

5. Add questions, answers, and statistics to the PDF history

6. Generate and download a comprehensive PDF report

### Example Questions

- What is the number of records in the file?
- What are the data types of the columns?
- What are the descriptive statistics for numeric columns?
- What are the unique values in a specific column?
- How many null values exist in each column?
- Which rows have column X greater than Y?
- What is the average of column Z grouped by W?
- What are the top 5 rows with the highest values in column A?
- Is there correlation between columns C and D?
- What is the sum of column F for each category in G?

## How It Works

1. **Data Loading**: The application reads CSV/Excel files using Pandas with automatic encoding detection
2. **Query Processing**: User questions are converted to executable Pandas code using LlamaIndex and Groq's LLM
3. **Response Generation**: The LLM analyzes the query results and generates natural language responses
4. **PDF Export**: User interactions and statistics are compiled into a professional PDF report

## Key Components

### Data Analysis Pipeline

The application uses a sophisticated query pipeline that:
- Converts natural language to Pandas code
- Executes queries safely on the DataFrame
- Synthesizes human-readable responses from query results

### Supported File Formats

- **CSV**: Comma-separated values with automatic encoding detection
- **Excel**: .xlsx, .xls, .xlsm, .xlsb formats with multi-sheet support

## Limitations

- Excel files with multiple sheets will only load the first sheet by default
- Maximum file size depends on available system memory
- Query complexity is limited by the LLM's code generation capabilities
- PDF generation uses basic formatting (DejaVu or Arial fonts)
- Requires active internet connection for Groq API calls

## Data Sources

This application works with any CSV or Excel file containing tabular data. Upload your own datasets for analysis.

## Configuration

The application can be configured through environment variables:
- `secret_key`: Your Groq API key (required)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Data Analysis Portfolio Project

## Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/)
- Powered by [Groq](https://groq.com/)
- Interface by [Gradio](https://gradio.app/)
