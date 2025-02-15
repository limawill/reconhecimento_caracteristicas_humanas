import cv2
from deepface import DeepFace

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Analyze each detected face
        for (x, y, w, h) in faces:
            # Draw a blue rectangle around each face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract the face region
            face = frame[y:y+h, x:x+w]

            # Analyze the face using DeepFace
            try:
                analysis = DeepFace.analyze(face, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                print(f"Analysis result: {analysis}")  # Debug print to inspect the analysis result

                if isinstance(analysis, list):
                    analysis = analysis[0]  # If analysis is a list, take the first element

                age = analysis['age']
                gender_confidence = analysis['gender']
                emotion = analysis['dominant_emotion']

                # Determine the gender with the higher confidence value
                gender = 'Man' if gender_confidence['Man'] > gender_confidence['Woman'] else 'Woman'

                # Display the analysis results on the frame
                cv2.putText(frame, f'Age: {age}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Emotion: {emotion}', (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print(f"Erro ao analisar a face: {e}")

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()