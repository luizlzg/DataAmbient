# Explicação geral

O seguinte repositório é uma ferramenta para gestão de ambientes, baseado nas pessoas que o frequentam, e, para isso, é necessário uma câmera.

A ferramenta funciona da seguinte forma:

1- ocorre a detecção da face de pessoas que passam em frente à câmera; <br> <br>
2- é associado um código único à face da pessoa e, desse modo, a pessoa é registrada no sistema de forma anônima; <br> <br> 
3- o código único do usuário, juntamente com o seu horário de entrada ou de saída é registrado no banco de dados.

# Arquivos

Cada arquivo do repositório é responsável por uma certa coisa.

database.py: código responsável por se conectar ao banco de dados e oferecer todos os métodos responsáveis por registrar as informações no banco.

main.py: é o código principal a ser executado. É o código que vai receber a imagem, direcionar para o processamento e enviar o retorno para o banco de dados.

image_process.py: é o código que irá realizar todos os processamentos necessários na imagem: detecção de face, obtenção do código único da face (embedding) e comparação de embeddings (compara as faces para ver se a pessoa já foi registrada no banco de dados).

# Passo a passo de execução

Execute o código principal:

	python3 main.py

Em um terminal separado, execute o código que irá obter as imagens da câmera e enviar para o processamento local:

	python3 run_camera.py

OBS: o código acima é um exemplo de como rodar a câmera, mas ainda não foi testado.


# Obtendo a quantidade de pessoas

No repositório há um exemplo de como se conectar ao banco de dados e executar a query que irá retornar a quantidade de pessoas. Execute:

	python3 printar_qtd_pessoas.py

Este é apenas um código de exemplo e para testar o desempenho da ferramenta, visto que a exibição da quantidade de pessoas será feita no dashboard.
