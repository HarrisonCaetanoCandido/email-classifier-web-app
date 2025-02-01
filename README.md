# Projeto de Classificação e Resposta Automática de Emails Financeiros

Este projeto tem como objetivo automatizar a classificação de emails em **Financeiros** e **Não Financeiros**, além de gerar respostas automáticas contextualizadas. A solução é composta por:

1. **Interface Web (HTML + JavaScript):** Permite ao usuário inserir o conteúdo do email e visualizar a classificação e a resposta sugerida.
2. **Backend (Flask API):** Recebe o conteúdo do email, classifica-o usando um modelo de machine learning e gera uma resposta automática.
3. **Modelo de Machine Learning:** Um modelo BERT ajustado para classificar emails financeiros.

## Estrutura do Projeto

```
projeto-emails/
│
├── frontend/
│   ├── index.html              # Interface web para inserir emails
│   └── styles.css              # Estilos CSS para a interface
│
├── backend/
│   ├── app.py                  # API Flask para classificação e resposta
│   └── modelo_financeiro/      # Modelo ajustado (salvo após o treinamento)
│
├── scripts/
│   └── treinar_modelo.py       # Script para treinar o modelo
│
├── README.md                   # Este arquivo
└── requirements.txt            # Dependências do projeto
```

---

## Como Executar o Projeto

### 1. Pré-requisitos

- Python 3.8 ou superior.
- Bibliotecas Python listadas em `requirements.txt`.
- Navegador moderno (Chrome, Firefox, Edge, etc.).

### 2. Instalação

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Treine o modelo (opcional, se ainda não tiver um modelo ajustado):
   ```bash
   python scripts/treinar_modelo.py
   ```

### 3. Executando o Backend

1. Navegue até a pasta `backend`:
   ```bash
   cd backend
   ```

2. Inicie o servidor Flask:
   ```bash
   python app.py
   ```

   O servidor estará disponível em `http://127.0.0.1:5000`.

### 4. Executando o Frontend

1. Abra o arquivo `frontend/index.html` no navegador.
2. Insira o conteúdo do email e clique em "Classificar Email".
3. A classificação e a resposta sugerida serão exibidas na tela.

---

## Detalhes do Projeto

### Interface Web (Frontend)

A interface web foi desenvolvida em **HTML** e **JavaScript** para permitir que o usuário insira o conteúdo do email e visualize os resultados. O frontend se comunica com o backend via requisições HTTP (POST).

#### Arquivo: `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classificador de Emails Financeiros</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1>Classificador de Emails Financeiros</h1>
    <textarea id="emailContent" placeholder="Cole o conteúdo do email aqui..."></textarea>
    <br>
    <button onclick="classifyEmail()">Classificar Email</button>

    <div id="result">
        <h2>Resultado:</h2>
        <p><strong>Classificação:</strong> <span id="classification"></span></p>
        <p><strong>Resposta Sugerida:</strong> <span id="suggestedResponse"></span></p>
    </div>

    <script>
        function classifyEmail() {
            const emailContent = document.getElementById('emailContent').value;

            fetch('http://127.0.0.1:5000/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email_content: emailContent }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('classification').innerText = data.classification;
                document.getElementById('suggestedResponse').innerText = data.suggested_response;
            })
            .catch(error => {
                console.error('Erro ao classificar o email:', error);
            });
        }
    </script>
</body>
</html>
```

#### Justificativa:
- **Simplicidade:** A interface é simples e intuitiva, permitindo que o usuário insira o email e visualize os resultados rapidamente.
- **Comunicação com o Backend:** O uso de `fetch` para enviar requisições POST ao backend permite uma integração eficiente.

---

### Backend (Flask API)

O backend foi desenvolvido em **Flask** e é responsável por:
1. Receber o conteúdo do email.
2. Classificar o email usando o modelo ajustado.
3. Gerar uma resposta automática com base na classificação.

#### Arquivo: `backend/app.py`

```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Carregue o modelo ajustado
model_path = "./modelo_financeiro_ajustado"
classifier = pipeline("text-classification", model=model_path)

# Função para gerar respostas automáticas
def generate_response(classification):
    if classification == "Financeiro":
        return "Olá, agradecemos por entrar em contato. Como podemos ajudar com sua questão financeira?"
    else:
        return "Olá, agradecemos sua mensagem! Para questões não financeiras, entre em contato com o suporte geral."

# Rota para classificar emails
@app.route('/classify', methods=['POST'])
def classify_email():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    email_content = data.get('email_content', '')

    # Classificação do email
    result = classifier(email_content)[0]
    classification = "Financeiro" if result['label'] == 'LABEL_1' else "Não Financeiro"

    # Geração de resposta automática
    suggested_response = generate_response(classification)

    return jsonify({
        "classification": classification,
        "suggested_response": suggested_response
    })

if __name__ == '__main__':
    app.run(debug=True)
```

#### Justificativa:
- **Rota `/classify`:** Foi escolhida para receber o conteúdo do email e retornar a classificação e a resposta sugerida.
- **Modelo Ajustado:** O uso de um modelo BERT ajustado permite uma classificação precisa de emails financeiros.
- **Respostas Contextualizadas:** As respostas são geradas com base na classificação, garantindo que sejam relevantes ao contexto.

---

### Modelo de Machine Learning

O modelo utilizado é um **BERT em português** ajustado para classificar emails como **Financeiros** ou **Não Financeiros**. O ajuste fino foi realizado usando um conjunto de dados específico para o domínio financeiro.

#### Arquivo: `scripts/treinar_modelo.py`

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Carregue o modelo e o tokenizador
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Conjunto de dados de exemplo
data = {
    "text": [
        "Olá, gostaria de saber o saldo da minha conta corrente.",
        "Preciso de ajuda para realizar uma transferência bancária.",
        "Feliz Natal e um próspero Ano Novo!",
        "Gostaria de agendar uma consulta médica."
    ],
    "label": [1, 1, 0, 0]  # 1 = Financeiro, 0 = Não Financeiro
}

# Converta para um Dataset do Hugging Face
dataset = Dataset.from_dict(data)

# Tokenize os dados
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Configurações de treinamento
training_args = TrainingArguments(
    output_dir="./modelo_financeiro_ajustado",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Ajuste fino
trainer.train()

# Salve o modelo ajustado
model.save_pretrained("./modelo_financeiro_ajustado")
tokenizer.save_pretrained("./modelo_financeiro_ajustado")
```

#### Justificativa:
- **BERT em Português:** Escolhido por ser um modelo pré-treinado em português, adequado para tarefas de classificação de texto.# Projeto de Classificação e Resposta Automática de Emails Financeiros


# Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Ajuste fino
trainer.train()

# Salve o modelo ajustado
model.save_pretrained("./modelo_financeiro_ajustado")
tokenizer.save_pretrained("./modelo_financeiro_ajustado")
