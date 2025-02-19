import os
import sys
import cv2
import shutil
import numpy as np
from tensorflow.keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL

# Redireciona stderr para /dev/null
sys.stderr = open(os.devnull, 'w')


# Mapeamento das emoções
emotion_mapping = {
    0: 'anger',
    1: 'contempt',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'neutral',
    6: 'sadness',
    7: 'surprise'
}

def print_centered_block(lines):
    """Imprime um bloco de texto centralizado no terminal."""
    terminal_width = shutil.get_terminal_size().columns
    max_length = max(len(line) for line in lines)  # Descobre o maior comprimento

    for line in lines:
        padded_line = line.center(max_length)  # Ajusta o comprimento de todas as linhas
        print(padded_line.center(terminal_width))  # Centraliza no terminal


def load_model_from_path(model_path: str, model_name: str):
    try:
        model = load_model(model_path)
        print(f"Modelo {model_name} carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo {model_name}: {e}")
        return None

def map_age_to_range(age_pred: float) -> str:
    age = int(age_pred)
    lower_bound = (age // 10) * 10
    upper_bound = lower_bound + 10
    return f"{lower_bound}-{upper_bound}"


def detect_faces(image: cv2.Mat, gender_model, age_model, emotion_model) -> cv2.Mat:
    """
    Detecta rostos em uma imagem e classifica gênero, idade e humor.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converte a imagem inteira para escala de cinza (para detecção de faces)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]  # Região do rosto em cores (para gênero e idade)

        # Processamento para os modelos de gênero e idade (224x224)
        face_resized_224 = cv2.resize(face_roi, (224, 224)) / 255.0
        face_input_224 = np.expand_dims(face_resized_224, axis=0)

        # Processamento para o modelo de humor (48x48) - Correção aqui!
        face_resized_48 = cv2.resize(face_roi, (48, 48))  # Redimensiona a face_roi (que está em cores)
        face_gray_48 = cv2.cvtColor(face_resized_48, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
        face_input_48_gray = np.expand_dims(face_gray_48, axis=0) / 255.0 # Normaliza
        face_input_48_gray = np.expand_dims(face_input_48_gray, axis=-1)  # Adiciona a dimensão do canal (1)

        # Predições
        gender_pred = gender_model.predict(face_input_224)
        gender = "Feminino" if gender_pred[0][0] > 0.5 else "Masculino"
        label_gender = f"Genero: {gender}" 

        age_pred = age_model.predict(face_input_224)
        age_range = map_age_to_range(age_pred[0][0])
        label_age = f"Idade: {age_range}" 

        emotion_pred = emotion_model.predict(face_input_48_gray)  # Usa a imagem em escala de cinza com a dimensão do canal
        emotion_index = np.argmax(emotion_pred)
        emotion_label = emotion_mapping[emotion_index]
        label_emotion = f"Humor: {emotion_label}"
        
        y_offset = -20  # Ajuste a posição vertical do texto
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(image, label_gender, (x, y + y_offset - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe gênero
        cv2.putText(image, label_age, (x, y + y_offset - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe idade (deslocado 20 pixels para baixo)
        cv2.putText(image, label_emotion, (x, y + y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe emoção (deslocado 40 pixels para baixo)

    
        
        # Exibir informações na imagem
        #label = f"{gender}, \n {age_range},\n {emotion_label}"
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image


def detect_faces_in_image(image_path: str, gender_model, age_model, emotion_model):
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return
    image_with_faces = detect_faces(image, gender_model, age_model, emotion_model)
    cv2.imshow("Rostos Detectados", image_with_faces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_webcam(gender_model, age_model, emotion_model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_faces = detect_faces(frame, gender_model, age_model, emotion_model)
        cv2.imshow("Rostos Detectados - Webcam", frame_with_faces)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main_menu(gender_model, age_model, emotion_model):
    os.system('clear')
    
    menu_text = [
        "=========== MENU PRINCIPAL ===========",
        "                                      ",
        "Escolha uma opção:",
        "1. Detectar rostos em uma imagem",
        "2. Detectar rostos na webcam",
        "3. Sair",
        "======================================="
    ]
    
    print_centered_block(menu_text)  # Exibe o menu centralizado
    
    choice = input("Digite o número da opção desejada: ")
    if choice == '1':
        image_path = input("Digite o caminho da imagem: ")
        detect_faces_in_image(image_path, gender_model, age_model, emotion_model)
    elif choice == '2':
        detect_faces_in_webcam(gender_model, age_model, emotion_model)
    elif choice == '3':
        print("Saindo...")
        exit()
    else:
        print("Opção inválida. Tente novamente.")


def select_model_cnn() -> str:
    os.system('clear')  # Limpa a tela

    model_options = {
        "1": "InceptionV3",
        "2": "MobileNetV2",
        "3": "ResNet50",
        "4": "EfficientNetB0",
        "5": "cnn_tcc_will",
        "6": "Sair"
    }

    menu_text = [
        "=========== MODELO CNN ===========",
        "                                  ",
        "Escolha uma opção:",
    ] + [f"{key}. {value}" for key, value in model_options.items()] + [
        "==================================="
    ]


    print_centered_block(menu_text)

    while True:
        choice = input("Digite o número da opção desejada: ").strip()

        if choice in model_options:
            if choice == "6":  # Se for a opção "Sair"
                print("Saindo...")
                exit()  # Encerra o programa

            return model_options[choice]  # Retorna o modelo escolhido

        print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    select_model = select_model_cnn()
    if select_model:
        gender_model_path = f"./modelos/sexo/{select_model}_gender_classifier.h5"
        age_model_path = f"./modelos/idade/ager_classifier_{select_model}.h5"
        emotion_model_path = f"./modelos/humor/{select_model}_humor_classifier.h5"
        
        gender_model = load_model_from_path(gender_model_path, "gênero")
        age_model = load_model_from_path(age_model_path, "idade")
        emotion_model = load_model_from_path(emotion_model_path, "humor")
        
        if gender_model and age_model and emotion_model:
            main_menu(gender_model, age_model, emotion_model)
    else:
        print("Modelo não selecionado.")
        exit()


