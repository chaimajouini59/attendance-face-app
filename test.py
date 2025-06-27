import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage  # Import pour les images

# --- Configuration mail ---
EMAIL_ADDRESS = "email"
EMAIL_PASSWORD = "drdndrumsrzxdvtg"  # Mot de passe application Gmail recommandé
EMAIL_RECEIVER = "email"

# --- Chemin pour les images inconnues temporaires ---
UNKNOWN_FACES_TEMP_DIR = "unknown_faces_temp"
if not os.path.exists(UNKNOWN_FACES_TEMP_DIR):
    os.makedirs(UNKNOWN_FACES_TEMP_DIR)


# --- Fonction d'envoi d'email ---
def send_email_unknown_face(date_time_str, image_path=None):
    subject = "Alerte: visage inconnu detecte"
    body = f"Un visage inconnu a ete detecte le {date_time_str}."

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as fp:
                img = MIMEImage(fp.read(), _subtype="jpeg")  # ou png selon le format
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                msg.attach(img)
            print(f"Image {os.path.basename(image_path)} attachee a l'email.")
        except Exception as e:
            print(f"Erreur lors de l'attachement de l'image: {e}")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, EMAIL_RECEIVER, text)
        server.quit()
        print("Email envoye avec succes pour visage inconnu.")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email: {e}")
    finally:
        # Supprimer l'image temporaire apres l'envoi de l'email
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            print(f"Image temporaire {os.path.basename(image_path)} supprimee.")


# --- Dlib ---
detector = dlib.get_frontal_face_detector()
# Utilisation de chemins relatifs ou dynamiques pour une meilleure portabilite
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dlib_path = os.path.join(base_dir, "data", "data_dlib")

predictor = dlib.shape_predictor(os.path.join(data_dlib_path, "C:/Users/Admin/Downloads/Face-Recognition-Based-Attendance-System-main/data/data_dlib/shape_predictor_68_face_landmarks.dat"))
face_reco_model = dlib.face_recognition_model_v1(
    os.path.join(data_dlib_path, "C:/Users/Admin/Downloads/Face-Recognition-Based-Attendance-System-main/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"))

# --- Configuration SQLite ---
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
table_name = "attendance"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)
conn.commit()
conn.close()


# --- Classe Face_Recognizer ---
class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        self.frame_cnt = 0

        self.face_features_known_list = []  # Les descripteurs faciaux des personnes connues
        self.face_name_known_list = []  # Les noms des personnes connues

        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        self.current_frame_face_position_list = []  # Positions [x, y] du coin haut-gauche des visages detectes
        self.current_frame_face_feature_list = []  # Descripteurs faciaux des visages detectes dans le cadre actuel

        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10  # Re-classifier toutes les 10 frames si le nombre de visages est stable

        self.last_email_time = 0  # Pour gerer l'envoi d'email limite dans le temps (pour les inconnus)

    def get_face_database(self):
        # Chemin du fichier CSV de features
        features_csv_path = os.path.join(base_dir, "data", "features_all.csv")

        if os.path.exists(features_csv_path):
            csv_rd = pd.read_csv(features_csv_path, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    # Convertir en float, gerer les valeurs vides (NaN)
                    features_someone_arr.append(float(csv_rd.iloc[i][j]) if pd.notna(csv_rd.iloc[i][j]) else 0.0)
                self.face_features_known_list.append(features_someone_arr)
            logging.info(f"Faces in Database: {len(self.face_features_known_list)}")
            return 1
        else:
            logging.warning(f"'{features_csv_path}' not found! Please run 'features_extraction_to_csv.py' first.")
            return 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        # Cette fonction est appelée quand le nombre de visages est stable.
        # Elle essaie d'associer les visages actuels a ceux du cadre precedent
        # en se basant sur la distance entre leurs centroides.
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                e_distance_current_frame_person_x_list.append(distance)

            if e_distance_current_frame_person_x_list:  # S'assurer que la liste n'est pas vide
                last_frame_num = e_distance_current_frame_person_x_list.index(
                    min(e_distance_current_frame_person_x_list))
                # Verifier si l'indice est valide avant d'y acceder
                if last_frame_num < len(self.last_frame_face_name_list):
                    self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer for Attendance", (20, 40), self.font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, f"Frame: {self.frame_cnt}", (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"FPS: {self.fps.__round__(2)}", (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, f"Faces: {self.current_frame_face_cnt}", (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Press 'Q' to Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            # Afficher le numero de Face au-dessus du visage si le centroide est disponible
            if i < len(self.current_frame_face_centroid_list):
                # Ce texte peut etre retire si on prefere juste le nom
                cv2.putText(img_rd, "Face_" + str(i + 1),
                            (int(self.current_frame_face_centroid_list[i][0]),
                             int(self.current_frame_face_centroid_list[i][1])),
                            self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # print(f"{name} is already marked as present for {current_date}") # Commente si tu ne veux pas de message repetitif
            pass
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)",
                           (name, current_time, current_date))
            conn.commit()
            print(f"{name} marked as present for {current_date} at {current_time}")

        conn.close()

    def process(self, stream):
        if self.get_face_database():  # S'assurer que la base de donnees de visages est chargee
            while stream.isOpened():
                self.frame_cnt += 1
                flag, img_rd = stream.read()
                if not flag:
                    print("Erreur: Impossible de lire le flux video. Verifiez la camera ou le chemin du fichier.")
                    break

                # Redimensionner l'image pour des performances et une taille d'affichage coherentes
                img_rd = cv2.resize(img_rd, (640, 480))

                self.update_fps()
                faces = detector(img_rd, 0)  # Detecter les visages dans l'image actuelle

                # Mettre a jour les compteurs et listes de visages
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []  # Reinitialiser pour le cadre actuel
                self.current_frame_face_position_list = []  # Reinitialiser pour le cadre actuel

                # Scene 1: Nombre de visages inchange et pas encore besoin de re-classifier
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and \
                        (self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("Scene 1: No face cnt changes in this frame!!!")

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1
                    else:
                        self.reclassify_interval_cnt = 0  # Reinitialiser si tous les visages sont connus

                    # Reconstruire current_frame_face_centroid_list pour le suivi
                    for i in range(self.current_frame_face_cnt):
                        self.current_frame_face_centroid_list.append(
                            np.array([(faces[i].left() + faces[i].right()) / 2,
                                      (faces[i].top() + faces[i].bottom()) / 2])
                        )

                    self.centroid_tracker()  # Appliquer le suivi de centroide

                    for i in range(self.current_frame_face_cnt):
                        # Dessiner le rectangle autour du visage
                        cv2.rectangle(img_rd, (faces[i].left(), faces[i].top()), (faces[i].right(), faces[i].bottom()),
                                      (255, 255, 255), 2)

                        # Position pour le texte du nom (au-dessus du visage)
                        self.current_frame_face_position_list.append((faces[i].left(), faces[i].top() - 10))

                    for i in range(self.current_frame_face_cnt):
                        name_to_display = self.current_frame_face_name_list[i]
                        position = self.current_frame_face_position_list[i]
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if name_to_display == "unknown":
                            cv2.putText(img_rd, f"Unknown {timestamp}", position, self.font, 0.6, (0, 0, 255), 2,
                                        cv2.LINE_AA)

                            now_time = time.time()
                            # Envoi d'email limite a une fois toutes les 30 secondes pour les inconnus
                            if (now_time - self.last_email_time > 30):  # Utilisez self.last_email_time
                                # Capture du visage inconnu
                                face_img = img_rd[faces[i].top():faces[i].bottom(), faces[i].left():faces[i].right()]
                                if face_img.size != 0:  # S'assurer que l'image n'est pas vide
                                    face_img_resized = cv2.resize(face_img, (150, 150))  # Redimensionner pour l'email
                                    image_filename = f"unknown_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                                    image_path = os.path.join(UNKNOWN_FACES_TEMP_DIR, image_filename)
                                    cv2.imwrite(image_path, face_img_resized)
                                    send_email_unknown_face(timestamp, image_path)
                                    self.last_email_time = now_time  # Mettre a jour le temps du dernier email envoye
                        else:
                            cv2.putText(img_rd, name_to_display, position, self.font, 0.8, (0, 255, 255), 1,
                                        cv2.LINE_AA)
                            self.attendance(name_to_display)  # Enregistre la presence de l'employe

                # Scene 2: Nombre de visages change ou intervalle de re-classification atteint
                else:
                    logging.debug("Scene 2: Faces cnt changes in this frame or reclassify interval reached")
                    self.reclassify_interval_cnt = 0  # Reinitialiser le compteur de re-classification
                    self.current_frame_face_feature_list = []  # Reinitialiser la liste des features
                    self.current_frame_face_name_list = ["unknown"] * self.current_frame_face_cnt  # Initialiser avec "unknown"

                    for i in range(self.current_frame_face_cnt):
                        shape = predictor(img_rd, faces[i])
                        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
                        self.current_frame_face_feature_list.append(face_descriptor)
                        self.current_frame_face_centroid_list.append(
                            np.array([(faces[i].left() + faces[i].right()) / 2,
                                      (faces[i].top() + faces[i].bottom()) / 2])
                        )
                        self.current_frame_face_position_list.append((faces[i].left(), faces[i].top() - 10))

                    # Effectuer la reconnaissance pour chaque visage detecte
                    for i in range(self.current_frame_face_cnt):
                        distances = []
                        # Parcourir la base de donnees de visages connus
                        for known_feature in self.face_features_known_list:
                            # S'assurer que known_feature est bien un tableau de 128 elements float
                            if isinstance(known_feature, list) and len(known_feature) == 128:
                                distance = self.return_euclidean_distance(self.current_frame_face_feature_list[i],
                                                                          known_feature)
                                distances.append(distance)
                            else:
                                distances.append(999)  # Distance elevee pour les descripteurs invalides

                        min_distance = min(distances) if distances else 999

                        if min_distance < 0.4:  # Si la distance est inferieure au seuil, c'est un visage connu
                            min_distance_index = distances.index(min_distance)
                            self.current_frame_face_name_list[i] = self.face_name_known_list[min_distance_index]
                            self.attendance(self.current_frame_face_name_list[i])  # Enregistre la presence
                        else:
                            self.current_frame_face_name_list[i] = "unknown"
                            # Gestion de l'email pour "unknown" ici aussi pour la Scene 2
                            now_time = time.time()
                            if (now_time - self.last_email_time > 30):
                                face_img = img_rd[faces[i].top():faces[i].bottom(), faces[i].left():faces[i].right()]
                                if face_img.size != 0:
                                    face_img_resized = cv2.resize(face_img, (150, 150))
                                    image_filename = f"unknown_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                                    image_path = os.path.join(UNKNOWN_FACES_TEMP_DIR, image_filename)
                                    cv2.imwrite(image_path, face_img_resized)
                                    send_email_unknown_face(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                            image_path)
                                    self.last_email_time = now_time

                    # Afficher les noms et rectangles pour la scene 2
                    for i in range(self.current_frame_face_cnt):
                        name_to_display = self.current_frame_face_name_list[i]
                        position = self.current_frame_face_position_list[i]
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if name_to_display == "unknown":
                            cv2.putText(img_rd, f"Unknown {timestamp}", position, self.font, 0.6, (0, 0, 255), 2,
                                        cv2.LINE_AA)
                        else:
                            cv2.putText(img_rd, name_to_display, position, self.font, 0.8, (0, 255, 255), 1,
                                        cv2.LINE_AA)

                        # Dessiner les rectangles autour des visages detectes pour la scene 2
                        cv2.rectangle(img_rd, (faces[i].left(), faces[i].top()), (faces[i].right(), faces[i].bottom()),
                                      (255, 255, 255), 2)

                self.draw_note(img_rd)  # Dessiner les informations generales

                cv2.imshow("Face Recognition", img_rd)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # 'q' ou 'Esc' pour quitter
                    break

            stream.release()
            cv2.destroyAllWindows()
        else:
            print(
                "Erreur : Impossible de charger la base de donnees de visages. Assurez-vous que 'features_all.csv' existe et contient des donnees.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Configurez le niveau de logging pour voir les messages INFO
    face_recognizer = Face_Recognizer()
    cap = cv2.VideoCapture(0)  # Utilisez la webcam par defaut (0)
    face_recognizer.process(cap)
