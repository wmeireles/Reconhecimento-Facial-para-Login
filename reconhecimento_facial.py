import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Diretório para armazenar imagens de rostos
DATASET_DIR = "faces_dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Inicializando o classificador Haar Cascade para detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Função para capturar imagens do usuário
def capturar_face(usuario):
    cap = cv2.VideoCapture(0)
    count = 0
    user_dir = os.path.join(DATASET_DIR, usuario)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    while count < 50:  # Captura 50 imagens do rosto
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            cv2.imwrite(f"{user_dir}/{count}.jpg", face)
            count += 1
        cv2.imshow("Captura de Rosto", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Função para treinar o modelo KNN
def treinar_modelo():
    X, y = [], []
    labels = {}
    label_id = 0
    
    for user in os.listdir(DATASET_DIR):
        user_path = os.path.join(DATASET_DIR, user)
        if os.path.isdir(user_path):
            labels[label_id] = user
            for img in os.listdir(user_path):
                img_path = os.path.join(user_path, img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                X.append(image.flatten())
                y.append(label_id)
            label_id += 1
    
    X, y = np.array(X), np.array(y)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    
    with open("face_recognizer.pkl", "wb") as f:
        pickle.dump((knn, labels), f)
    print("Modelo treinado com sucesso!")

# Função para reconhecer o rosto e fazer login
def reconhecer_face():
    with open("face_recognizer.pkl", "rb") as f:
        knn, labels = pickle.load(f)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100)).flatten()
            label = knn.predict([face])[0]
            usuario = labels[label]
            cv2.putText(frame, f"Usuario: {usuario}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    escolha = input("1. Capturar Rosto\n2. Treinar Modelo\n3. Reconhecer e Logar\nEscolha uma opção: ")
    if escolha == "1":
        usuario = input("Digite o nome do usuário: ")
        capturar_face(usuario)
    elif escolha == "2":
        treinar_modelo()
    elif escolha == "3":
        reconhecer_face()
    else:
        print("Opção inválida!")
