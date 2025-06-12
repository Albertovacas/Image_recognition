import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim



class FaceDetector:
    def __init__(self, model):
        self.model = model
        self._set_paths()

    def _set_paths(self):
        base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.model_dir = os.path.join(base_dir, 'model')
        self.src_dir = os.path.join(base_dir, 'src')
        self.data_dir = os.path.join(base_dir, 'data')
        self.results_dir = os.path.join(base_dir, 'results')

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'faces'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'matched'), exist_ok=True)


    def get_faces(self, images=None, save_image=True):
        people_dir = os.path.join(self.data_dir, 'people')
        
        if images is None:
            images_to_read = os.listdir(people_dir)
        else:
            missing = [img for img in images if img not in os.listdir(people_dir)]
            if missing:
                raise Exception(
                    f"Images not found in directory {people_dir}: {missing}"
                )
            images_to_read = images

        for image_name in images_to_read:
            input_path = os.path.join(people_dir, image_name)
            file_stem = os.path.splitext(image_name)[0]
            output_prefix = os.path.join(self.results_dir, 'faces', file_stem)
            self._detect_faces(input_path, output_prefix, save_image)


    def find_images_with_all_faces(self, threshold=0.75):
        faces_dir = os.path.join(self.results_dir, 'faces')
        photos_dir = os.path.join(self.data_dir, 'photos')
        reference_files = [f for f in os.listdir(faces_dir) if f.endswith('.jpg')]

        grouped_faces = {}
        for f in reference_files:
            base_name = f.split('.')[0]
            grouped_faces.setdefault(base_name, []).append(f)

        for base_name, face_files in grouped_faces.items():
            save_dir = os.path.join(self.results_dir, 'matched', base_name)
            os.makedirs(save_dir, exist_ok=True)

            for face_file in face_files:
                ref_path = os.path.join(faces_dir, face_file)
                reference_face = cv2.imread(ref_path)
                if reference_face is None:
                    print(f"Could not read reference face: {ref_path}")
                    continue

                images_to_scan = os.listdir(photos_dir)
                for img_name in images_to_scan:
                    img_path = os.path.join(photos_dir, img_name)
                    detected_faces = self._detect_faces(img_path,'',False)

                    for face in detected_faces:
                        similarity = self._calculate_ssim(reference_face, face)
                        print(f'The similarity normal with {img_name} is {similarity}')
                        if similarity > threshold:
                            save_path = os.path.join(save_dir, img_name)
                            if not os.path.exists(save_path):
                                img = cv2.imread(img_path)
                                cv2.imwrite(save_path, img)
                            break  # Save once per image
    
    # Function to calculate SSIM
    def _calculate_ssim(self, face1, face2):
        # Convert images to grayscale
        grayA = cv2.cvtColor(cv2.resize(face1, (75, 75)), cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(cv2.resize(face2, (75, 75)), cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM between the two images
        score, _ = ssim(grayA, grayB, full=True)
        return score


    def _detect_faces(self, image_path, output_path, save_image):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        (h, w) = img.shape[:2]

        # Preprocess
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.model.setInput(blob)
        detections = self.model.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = img[y1:y2, x1:x2]
            faces.append(face)
            if save_image:
                face_path = f"{output_path}_{i}.jpg"
                cv2.imwrite(face_path, face)
                
        return faces
                    
                    
