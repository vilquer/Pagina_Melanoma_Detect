# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image

# # Configura o dispositivo como CPU, já que será usado em outro hardware
# device = torch.device('cpu')

# # Função para carregar a imagem
# def load_image(image_file):
#     img = Image.open(image_file)
#     return img

# # Interface no Streamlit
# st.title("Classificação de Imagens com EfficientNet")

# # Carregar uma imagem
# uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

# # Carregar o modelo EfficientNet pré-treinado e ajustado
# @st.cache_resource  # Cache do modelo para evitar recarga constante
# def load_model():
#     model = models.efficientnet_v2_l(weights=None)  # Carregar sem pesos padrão
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # Ajustar saída para 1 classe
    
#     # Carregar o state_dict do modelo treinado
#     state_dict = torch.load('best_model.pth', map_location=device)
    
#     # Remover o prefixo "module." se estiver presente
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key.replace("module.", "")  # Remove o prefixo "module."
#         new_state_dict[new_key] = value

#     # Carregar o state_dict modificado no modelo  
#     model.load_state_dict(new_state_dict)
#     model.to(device)
#     model.eval()  # Colocar em modo de avaliação
#     return model

# # Aplicar transformações à imagem (deve coincidir com as do treino)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# if uploaded_file is not None:
#     # Carregar e exibir a imagem
#     image = load_image(uploaded_file)
#     st.image(image, caption="Imagem Carregada", use_column_width=True)
    
#     # Pré-processar a imagem
#     input_tensor = preprocess(image)
#     input_batch = input_tensor.unsqueeze(0)  # Criar batch com a imagem
    
#     # Carregar o modelo
#     model = load_model()

#     # Fazer a predição
#     with torch.no_grad():
#         input_batch = input_batch.to(device)
#         output = model(input_batch)
#         prediction = torch.sigmoid(output).item()

#     # Exibir a predição
#     st.write("Resultado da predição:", prediction)
# load_state_dict


# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image

# # Configura o dispositivo como CPU
# device = torch.device('cpu')

# # Função para carregar a imagem do arquivo
# def load_image(image_file):
#     img = Image.open(image_file)
#     return img

# # Interface do Streamlit
# st.title("Classificação de Imagens com EfficientNet")

# # Mensagem sobre o propósito do projeto
# st.warning(
#     "### Atenção: Prova de Conceito\n"
#     "Este aplicativo é uma prova de conceito para estudos de tratamento de imagens com Inteligência Artificial (IA).\n  "
#     "Os resultados apresentados não devem ser usados como diagnóstico médico, decisão jurídica, ou qualquer outro tipo de avaliação definitiva. "
#     "Esta ferramenta foi criada para fins educacionais e de pesquisa.\n\n"
#     "**Recomendações:**\n"
#     "- O uso de IA em áreas como saúde, segurança e tomadas de decisão importantes deve sempre seguir normas éticas e legais rigorosas.\n"
#     "- Os resultados gerados por IA devem ser analisados e validados por especialistas humanos e não podem ser utilizados isoladamente para decisões críticas.\n"
#     "- Por favor, siga as regulamentações locais de privacidade e proteção de dados ao utilizar tecnologias de IA.\n\n"
#     "** Utilize este aplicativo somente como parte de seus estudos ou experimentos. Para diagnósticos ou análises reais, consulte sempre especialistas qualificados.**"
# )
# # Opção para selecionar o método de entrada de imagem
# input_option = st.radio("Escolha uma opção para carregar a imagem:", ("Upload de imagem", "Captura pela webcam"))

# # Variável para armazenar a imagem
# image = None

# # Carregar a imagem de acordo com a opção selecionada
# if input_option == "Upload de imagem":
#     uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image = load_image(uploaded_file)
#         st.image(image, caption="Imagem Carregada", use_column_width=True)  # Exibe a imagem apenas no caso de upload
# else:
#     # Capturar imagem da webcam usando st.camera_input
#     camera_image = st.camera_input("Tire uma foto")
#     if camera_image is not None:
#         image = load_image(camera_image)  # Carregar a imagem capturada pela câmera sem exibir novamente

# # Carregar o modelo EfficientNet pré-treinado e ajustado
# @st.cache_resource
# def load_model():
#     model = models.efficientnet_v2_l(weights=None)
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
#     state_dict = torch.load('best_model.pth', map_location=device)
#     new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     model.to(device)
#     model.eval()
#     return model

# # Transformações para pré-processamento
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Executar a predição se a imagem foi carregada
# if image is not None:
#     input_tensor = preprocess(image)
#     input_batch = input_tensor.unsqueeze(0).to(device)

#     model = load_model()
#     with torch.no_grad():
#         output = model(input_batch)
#         prediction = torch.sigmoid(output).item()

#     st.write("Resultado da predição:", prediction)



import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import asyncio
import gdown
import os

# ID do arquivo no Google Drive
file_id = "1C34x7CUYGt-OhjHJH4ldbq-DqZ_pyB6g"
output_path = "best_model.pth"

# Verifica se o modelo já foi baixado
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)


# Configura o dispositivo como CPU
device = torch.device('cpu')

# Função para carregar a imagem do arquivo
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Interface do Streamlit
st.title("Classificação de Imagens com EfficientNet teste")

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
