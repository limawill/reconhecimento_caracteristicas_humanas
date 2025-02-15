import cv2
import numpy as np
from tensorflow.keras.models import load_model


def load_model_from_path(model_path: str, model_name: str):
    """
    Carrega um modelo Keras a partir de um arquivo .h5.

    Args:
        model_path (str): Caminho para o arquivo .h5 contendo o modelo treinado.
        model_name (str): Nome do modelo (para mensagens de log).

    Returns:
        model: Modelo Keras carregado.
    """
    try:
        model = load_model(model_path)
        print(f"Modelo {model_name} carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo {model_name}: {e}")
        return None

def map_age_to_range(age_pred: float) -> str:
    """
    Mapeia a previsão de idade para uma faixa etária (ex: 20-30).

    Args:
        age_pred (float): Valor predito pelo modelo de idade.

    Returns:
        str: Faixa etária correspondente.
    """
    age = int(age_pred)  # Converte a previsão para um valor inteiro
    lower_bound = (age // 10) * 10  # Calcula o limite inferior da faixa
    upper_bound = lower_bound + 10  # Calcula o limite superior da faixa
    return f"{lower_bound}-{upper_bound}"

def detect_faces(image: cv2.Mat, gender_model, age_model) -> cv2.Mat:
    """
    Detecta rostos em uma imagem, classifica o gênero e a idade, e desenha um quadrado verde ao redor de cada rosto detectado.

    Args:
        image (cv2.Mat): A imagem na qual os rostos serão detectados.
        gender_model: Modelo de classificação de gênero.
        age_model: Modelo de classificação de idade.

    Returns:
        cv2.Mat: A imagem com os rostos detectados, quadrados verdes, gênero e faixa etária classificados.
    """
    # Carrega o classificador Haar Cascade para detecção de rostos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detecta os rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Para cada rosto detectado
    for (x, y, w, h) in faces:
        # Extrai a região do rosto
        face_roi = image[y:y+h, x:x+w]
        
        # Pré-processa o rosto para o modelo de gênero e idade
        face_resized = cv2.resize(face_roi, (224, 224))  # Redimensiona para 224x224 (tamanho esperado pelos modelos)
        face_normalized = face_resized / 255.0  # Normaliza os pixels
        face_input = np.expand_dims(face_normalized, axis=0)  # Adiciona dimensão do batch
        
        # Faz a previsão do gênero
        gender_pred = gender_model.predict(face_input)
        gender = "Feminino" if gender_pred[0][0] > 0.5 else "Masculino"

        # Faz a previsão da idade e mapeia para uma faixa etária
        age_pred = age_model.predict(face_input)
        age_range = map_age_to_range(age_pred[0][0])

        # Desenha um quadrado verde ao redor do rosto
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Exibe o gênero e a faixa etária acima do quadrado
        label = f"{gender}, {age_range}"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

def detect_faces_in_image(image_path: str, gender_model, age_model) -> None:
    """
    Detecta rostos em uma imagem, classifica o gênero e a idade, e exibe a imagem com os resultados.

    Args:
        image_path (str): O caminho para a imagem na qual os rostos serão detectados.
        gender_model: Modelo de classificação de gênero.
        age_model: Modelo de classificação de idade.
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Detecta os rostos na imagem e classifica o gênero e a idade
    image_with_faces = detect_faces(image, gender_model, age_model)
    
    # Exibe a imagem com os rostos detectados e gênero e faixa etária classificados
    cv2.imshow("Rostos Detectados", image_with_faces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_webcam(gender_model, age_model) -> None:
    """
    Detecta rostos em tempo real a partir da webcam, classifica o gênero e a idade, e exibe o vídeo com os resultados.

    Args:
        gender_model: Modelo de classificação de gênero.
        age_model: Modelo de classificação de idade.
    """
    # Abre a webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return
    
    while True:
        # Captura frame por frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detecta os rostos no frame atual e classifica o gênero e a idade
        frame_with_faces = detect_faces(frame, gender_model, age_model)
        
        # Exibe o frame com os rostos detectados e gênero e faixa etária classificados
        cv2.imshow("Rostos Detectados - Webcam", frame_with_faces)
        
        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libera a webcam e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

def main_menu(gender_model, age_model) -> None:
    """
    Exibe um menu para o usuário escolher entre detectar rostos em uma imagem ou na webcam.

    Args:
        gender_model: Modelo de classificação de gênero.
        age_model: Modelo de classificação de idade.
    """
    print("Escolha uma opção:")
    print("1. Detectar rostos em uma imagem")
    print("2. Detectar rostos na webcam")
    print("3. Sair")
    
    choice = input("Digite o número da opção desejada: ")
    
    if choice == '1':
        image_path = input("Digite o caminho da imagem: ")
        detect_faces_in_image(image_path, gender_model, age_model)
    elif choice == '2':
        detect_faces_in_webcam(gender_model, age_model)
    elif choice == '3':
        print("Saindo...")
        exit()
    else:
        print("Opção inválida. Tente novamente.")


def select_model_cnn() -> str:

    print("Escolha uma opção:")
    print("1. InceptionV3")
    print("2. MobileNetV2")
    print("3. ResNet50")
    print("4. EfficientNetB0")
    print("5. cnn_tcc_will")
    print("6. Sair")
    
    choice = input("Digite o número da opção desejada: ")
    
    if choice == '1':
        input_model = "InceptionV3"
    elif choice == '2':
        input_model = "MobileNetV2"
    elif choice == '3':
        input_model = "ResNet50"
    elif choice == '4':
        input_model = "EfficientNetB0"
    elif choice == '5':
        input_model = "cnn_tcc_will"
    else:
        input_model = None

    return input_model
    


if __name__ == "__main__":
    # Carrega os modelos de gênero e idade
    
    select_model = select_model_cnn()
    
    print(f"Retorno da select_model_cnn: {select_model}")
    
    if select_model:
        print("dentro do if")
        gender_model_path = f"./modelos/sexo/{select_model}_gender_classifier.h5"
        age_model_path = f"./modelos/idade/ager_classifier_{select_model}.h5"
        
        print(f"gender_model_path do if: {gender_model_path}")
        print(f"age_model_path: {age_model_path}")

        gender_model = load_model_from_path(gender_model_path, "gênero")
        print(f"gender_model: {gender_model}")
        age_model = load_model_from_path(age_model_path, "idade")
        print(f"age_model: {age_model}")
    
        if gender_model and age_model:
            main_menu(gender_model, age_model)
    else:
        print("Modelo não selecionado.")
        exit()
