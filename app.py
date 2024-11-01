import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import asyncio
import gdown
import os


# ID do arquivo no Google Drive e caminho de saída temporário
file_id = "1C34x7CUYGt-OhjHJH4ldbq-DqZ_pyB6g"
output_path = "/tmp/best_model.pth"

# Função para baixar com barra de progresso
def download_with_progress(file_id, output_path):
    # Link direto para o arquivo
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Exibe a barra de progresso e mensagem de carregamento
    with st.spinner('Baixando modelo, aguarde...'):
        # Baixa o arquivo sem callback
        gdown.download(download_url, output_path, quiet=False)

# Verifica se o modelo já foi baixado
if not os.path.exists(output_path):
    download_with_progress(file_id, output_path)

# Configura o dispositivo como CPU
device = torch.device('cpu')

# Função para carregar a imagem do arquivo
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Interface do Streamlit
st.title("Classificação de Imagens com EfficientNet")

# Mensagem sobre o propósito do projeto
st.warning(
    "### Atenção: Prova de Conceito\n"
    "Este aplicativo é uma prova de conceito para estudos de tratamento de imagens com Inteligência Artificial (IA).\n  "
    "Os resultados apresentados não devem ser usados como diagnóstico médico, decisão jurídica, ou qualquer outro tipo de avaliação definitiva. "
    "Esta ferramenta foi criada para fins educacionais e de pesquisa.\n\n"
    "**Recomendações:**\n"
    "- O uso de IA em áreas como saúde, segurança e tomadas de decisão importantes deve sempre seguir normas éticas e legais rigorosas.\n"
    "- Os resultados gerados por IA devem ser analisados e validados por especialistas humanos e não podem ser utilizados isoladamente para decisões críticas.\n"
    "- Por favor, siga as regulamentações locais de privacidade e proteção de dados ao utilizar tecnologias de IA.\n\n"
    "** Utilize este aplicativo somente como parte de seus estudos ou experimentos. Para diagnósticos ou análises reais, consulte sempre especialistas qualificados.**"
)

# Opção para selecionar o método de entrada de imagem
input_option = st.radio("Escolha uma opção para carregar a imagem:", ("Upload de imagem", "Captura imagem pela Câmera"))

# Variável para armazenar a imagem
image = None

# Carregar a imagem de acordo com a opção selecionada
if input_option == "Upload de imagem":
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="Imagem Carregada", use_column_width=True)
else:
    camera_image = st.camera_input("Tire uma foto")
    if camera_image is not None:
        image = load_image(camera_image)

# Função assíncrona para carregar o modelo EfficientNet pré-treinado e ajustado
async def load_model_async():
    model = models.efficientnet_v2_l(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    state_dict = torch.load(output_path, map_location=device)
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

# Inicia o carregamento do modelo de forma assíncrona
model = st.session_state.get("model")
if model is None:
    st.write("Carregando o modelo, por favor aguarde...")
    model = asyncio.run(load_model_async())
    st.session_state["model"] = model  # Armazena no estado da sessão para evitar recargas

# Transformações para pré-processamento
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Executar a predição se a imagem foi carregada e o modelo está pronto
if image is not None and model is not None:
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)
        prediction = torch.sigmoid(output).item()

    st.write("Resultado da predição:", prediction)
