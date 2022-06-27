# Deep Cam

Aplicação que tem como objetivo analisar images e obter informações, bem como inferir características como idade, gênero, expressões faciais, detecção de pessoas, contagem de passantes e tempo de permanência em uma área delimitada no vídeo.

![](https://github.com/leonardogandrade/deep_cam/blob/master/etc/header.png?style=centerme)

### Consiguração do Ambiente

1 - Crie um ambiente virtual

```
python3 -m venv --system-site-packages ./venv
```

2 - Ative o ambiente virtual

```
source ./venv/bin/activate
```

Note que o terminal fica fixado como (venv)

3 - Atualização do PIP

```
pip install --upgrade pip
```

4 - instale os requisitos

```
pip install -r requirements.txt
```

## GERAL

- Você deve baixar os modelos pré treinados no link abaixo e descompactar na raiz do projeto
  https://drive.google.com/drive/folders/1adP_R590UKdA6jnRLkmld0fkEjfHaGVX?usp=sharing

- A pasta assets contêm exemplos de imagens e vídeos

- Os resultados das análises estarão sempre na pasta results

# Análises

## Predição de expressões faciais:

![](https://github.com/leonardogandrade/deep_cam/blob/master/etc/gif_expressions.gif?style=centerme)

- Execute o comando:

```
python3 expressionDetect.py assets/expressions1_480P.mov
```

## Área de interesse:

![](https://github.com/leonardogandrade/deep_cam/blob/master/etc/gif_fence.gif?style=centerme)

Exemplo - Quando rodando, marcar a área de interesse com o mouse (clicar em 4 pontos), o quinto clique limpa a tela:

- Execute o comando:

```
python3 fenceDetect.py assets/video4_480P.mov
```

## Predição de gênero e idade:

![](https://github.com/leonardogandrade/deep_cam/blob/master/etc/age_gender.jpg?style=centerme)

- Execute o comando:

```
python3 ageGenderPrediction.py assets/person.png
```

## Relatórios:

![](https://github.com/leonardogandrade/deep_cam/blob/master/etc/report.jpg?style=centerme)

- Execute o comando:

```
python3 reports.py
```
