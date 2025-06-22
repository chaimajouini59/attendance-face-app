import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Configuration des chemins
base_path = "C:/Users/Admin/Downloads/Face-Recognition-Based-Attendance-System-main/Face-Recognition-Based-Attendance-System-main"
path_images_from_camera = os.path.join(base_path, "data/data_faces_from_camera/")
shape_predictor_path = os.path.join(base_path, "C:/Users/Admin/Downloads/Face-Recognition-Based-Attendance-System-main/data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_model_path = os.path.join(base_path, "C:/Users/Admin/Downloads/Face-Recognition-Based-Attendance-System-main/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
features_csv_path = os.path.join(base_path, "data/features_all.csv")

# Initialisation des modèles
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_reco_model = dlib.face_recognition_model_v1(face_model_path)

def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)
    logging.info("%-40s %-20s", "Image with faces detected:", path_img)

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("Aucun visage détecté dans %s", path_img)
    return face_descriptor

def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)

    if photos_list:
        for img_name in photos_list:
            img_path = os.path.join(path_face_personX, img_name)
            logging.info("Lecture de l'image : %s", img_path)
            features_128d = return_128d_features(img_path)
            if features_128d != 0:
                features_list_personX.append(features_128d)
    else:
        logging.warning("⚠️ Aucun fichier dans le dossier : %s", path_face_personX)

    if features_list_personX:
        return np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        return np.zeros(128, dtype=object)

def main():
    logging.basicConfig(level=logging.INFO)

    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    with open(features_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            logging.info("Traitement de : %s", person)
            path_person = os.path.join(path_images_from_camera, person)
            features_mean_personX = return_features_mean_personX(path_person)

            if len(person.split('_')) == 2:
                person_name = person
            else:
                person_name = person.split('_', 2)[-1]

            features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
            writer.writerow(features_mean_personX)
            logging.info("✅ Enregistré : %s", person_name)

    logging.info("✅ Toutes les features ont été sauvegardées dans : %s", features_csv_path)

if __name__ == '__main__':
    main()
