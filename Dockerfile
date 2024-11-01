# Use a imagem oficial do Python como base
FROM python:3.10.12

# Define o diretório de trabalho no container
WORKDIR /app

# Copia o arquivo de requisitos para o diretório de trabalho
COPY requirements.txt .

# Instala as dependências necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação para o diretório de trabalho
COPY . .

# Define a porta para o Render (ou para seu ambiente de teste)
ENV PORT=10000

# Exponha a porta definida
EXPOSE 10000

# Comando para rodar o Streamlit quando o container iniciar
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0