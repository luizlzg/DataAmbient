# Explicação geral

O seguinte repositório é uma ferramenta para gestão de ambientes, baseado nas pessoas que o frequentam, e, para isso, é necessário uma câmera.

A ferramenta funciona da seguinte forma:

1- ocorre a detecção da face de pessoas que passam em frente à câmera; <br> <br>
2- é associado um código único à face da pessoa e, desse modo, a pessoa é registrada no sistema de forma anônima; <br> <br> 
3- através da face da pessoa e seu respectivo código único, é possível classificar o gênero e idade das pessoas que passam na câmera, registrar quantas pessoas estão no ambiente e calcular o tempo que as pessoas permanecem no ambiente e quanto tempo elas demoram para voltar para o ambiente;<br> <br>
4- por fim, tudo isso é exibido em um dashboard.

# Arquivos

Cada arquivo do repositório é responsável por uma certa coisa.

dashboard.py: é o arquivo que irá executar a interface de exibição, ou seja, o dashboard em que é possível observar as métricas calculadas.

gender_age_detection.py: é o arquivo em que se encontram as redes neurais responsáveis por classificar gênero e idade da pessoa.

utils.py: contém algumas funções utilitárias que são aproveitadas nos outros arquivos.

data_ambient.py: responsável pela classe que realiza todo o processamento da imagem e cálculo das métricas. Esse código é o que vai calcular as métricas, classificar gênero e idade.

main.py: é o código principal a ser executado. É o código que vai receber a imagem, direcionar para o processamento e enviar o retorno para o dashboard.

webcam.py: é o código responsável por captar a imagem da webcam e enviar para o código principal.

# Passo a passo de execução

No local de execução do código, crie duas pastas:

"images": para armazenar a face das pessoas que passam na câmera <br>
"weights": para armazenar as redes neurais presentes no seguinte link: https://drive.google.com/drive/folders/12GgdN99XYer34eK7XG8XjeDyXpyqAjuQ?usp=sharing 

Então, instale as dependências e bibliotecas necessárias com o comando:

	pip install -r requirements.txt

Em seguida, execute o dashboard para começar a acompanhar as métricas, com o comando:

	python3 -m streamlit run dashboard.py

Depois, comece a capturar a imagem da webcam, com o comando:

	python3 webcam.py

E, por fim, execute o código principal:

	python3 main.py
