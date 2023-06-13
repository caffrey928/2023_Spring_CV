import os
import numpy as np
import cv2
from tqdm import tqdm
from yolo_algorithm import yolo_algorithm
from ultralytics import YOLO

def gen_solution(dataset_path: str, subjects: list, solution_path: str):
    if not os.path.exists(solution_path):
        os.makedirs(solution_path)

    model = YOLO('../model/0.856.pt')  # load a custom model
    sequence_idx = 0
    for subject in subjects:
        solution_subject_path = os.path.join(solution_path, subject)
        if not os.path.exists(solution_subject_path):
            os.makedirs(solution_subject_path)

        count = 0
        for path in os.listdir(os.path.join(dataset_path, subject)):
            if path.isnumeric():
                count += 1

        for action_number in range(count):
            print(f'Generating solution for {subject} {action_number + 1:02d}')

            solution_label_folder = os.path.join(solution_path, subject, f'{action_number + 1:02d}')
            if not os.path.exists(solution_label_folder):
                os.makedirs(solution_label_folder)

            conf_file = open(os.path.join(solution_label_folder, 'conf.txt'), "w")

            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                label_name = os.path.join(solution_label_folder, f'{idx}.png')
                # image = cv2.imread(image_name)

                output, conf = yolo_algorithm(model , image_name)
                
                cv2.imwrite(label_name, output*255)
                conf_file.write(str(conf) + '\n')
            
            conf_file.close()
    return

if __name__ == '__main__':
    # dataset_path = r'D:\CV23_Ganzin_final_project\dataset\public'
    dataset_path = '../data/'
    subjects = ['S5', 'S6', 'S7', 'S8']
    solution_path = '../solution/'
    gen_solution(dataset_path, subjects, solution_path)