import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QLabel, QButtonGroup
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class ImageClassifier(QDialog):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        uic.loadUi('E:/Artificial Intelligence/Data Science & ML Diploma AMIT/05-Computer Vision/Final project of module/Final Project - George/GUI Detection.ui', self)

        # Default image path
        self.image_path = 'E:/Artificial Intelligence/Data Science & ML Diploma AMIT/05-Computer Vision/Final project of module/Final Project - George/Home.jpg'

        # Upload image button
        self.upload_button = self.findChild(QPushButton, 'BtnUploadImage')
        self.upload_button.clicked.connect(self.load_image)

        # Show image again button
        self.show_img_button = self.findChild(QPushButton, 'BtnOriginal')
        self.show_img_button.clicked.connect(lambda: self.load_image(again=True))

        # Image viewer
        self.image_viewer_1 = self.findChild(QGraphicsView, 'ImageWindow')
        self.scene = QGraphicsScene(self)
        # self.image_viewer_1.setScene(self.scene)
        img = cv2.imread(self.image_path)
        img_resized = cv2.resize(img, (620, 420))  # Resize for display
        height, width, _ = img_resized.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(q_img))
        self.image_viewer_1.setScene(self.scene)

        # Image Shape label
        self.image_shape_label = self.findChild(QLabel, 'ImageShapeLabel')


        # Canny edges button
        self.canny_button = self.findChild(QPushButton, 'BtnCanny')
        self.canny_button.clicked.connect(self.show_canny_edges)

        # Models names mapping
        self.models_names = {'My-Net': 'best_keras_model.h5', 'VGG-16': 'bestVGG16_model.h5','VGG-19': 'bestVGG16_model.h5',
                             'ResNet': 'bestResNet152V2_model.h5', 'Inception': 'bestInceptionV3_model.h5', 'DenseNet': 'bestDenseNet_model.h5','Models Ensemble': 'empty'}
       
        # Initial Model Name
        self.model_name = None

        # Radio buttons
        self.radio_group = self.findChild(QButtonGroup, 'ModelsGroup')
        self.radio_group.buttonClicked.connect(self.radio_button_clicked)
        
        
        # Predict button
        self.predict_button = self.findChild(QPushButton, 'BtnPredict')
        self.predict_button.clicked.connect(self.predict_with_model)

        # Result labels
        self.result_label_class = self.findChild(QLabel, 'result_label_class')
        self.result_label_confidence = self.findChild(QLabel, 'result_label_confidence')

        # Clear result button
        self.clear_button = self.findChild(QPushButton, 'BtnClear')
        self.clear_button.clicked.connect(self.clear_result)


        # Class names mapping
        self.classes = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

        self.image_path = None

        self.note_label = self.findChild(QLabel, 'Note')
        self.note_label.setText('----> This Takes a while!')


    def clear_result(self):
        self.result_label_class.setText('------------------')
        self.result_label_confidence.setText('---- %')
        self.result_label_class.setStyleSheet("color: black; font-size: 15px; font-weight: bold;")
        self.result_label_confidence.setStyleSheet("color: black; font-size: 15px; font-weight: bold;")

    def radio_button_clicked(self):
        for button in self.radio_group.buttons():
            if button.isChecked():
                self.model_name = button.text()
                self.model_path = self.models_names[self.model_name]


    def load_model(self):
        # Load model
        try:
            self.model = load_model(f'E:/Artificial Intelligence/Data Science & ML Diploma AMIT/05-Computer Vision/Final project of module/Final Project - George/Model/Weights/{self.model_path}')
        except:
            self.radio_button_clicked()
            self.model = load_model(f'E:/Artificial Intelligence/Data Science & ML Diploma AMIT/05-Computer Vision/Final project of module/Final Project - George/Model/Weights/{self.model_path}')


    def show_canny_edges(self):
        if self.image_path:
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (620, 420))  # Resize for display
            edges = cv2.Canny(img_resized, 50, 130)


            height, width = edges.shape
            bytes_per_line = width
            q_img = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            self.scene.clear()
            self.scene.addPixmap(QPixmap.fromImage(q_img))
            self.image_viewer_1.setScene(self.scene)


    def load_image(self, again=False):
        if not again:
            options = QFileDialog.Options()
            self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if self.image_path:
            try:
                # Load and resize image for viewing
                self.current_image = cv2.imread(self.image_path)
                self.image_shape_label.setText(f'Image shape: {self.current_image.shape[0]} x {self.current_image.shape[1]}')
                img_resized = cv2.resize(self.current_image, (620, 420))  # Resize for display
                height, width, _ = img_resized.shape
                bytes_per_line = 3 * width
                q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                self.scene.clear()
                self.scene.addPixmap(QPixmap.fromImage(q_img))
                self.image_viewer_1.setScene(self.scene)
            except Exception as e:
                self.result_label.setText(f"Error loading image: {e}")

    def preprocess_image(self, image_path, model_name):
        size = (150,150)
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, size)  # Resize to model input size
            if model_name != 'My-Net':
                img = img.astype('float32') / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        except Exception as e:
            self.result_label.setText(f"Error processing image: {e}")
            return None

    def get_ensemble_prediction(self):
        self.results = np.zeros(6)

        for model in ['My-Net', 'ResNet', 'DenseNet']:
            self.crnt_ensemble = load_model(f'E:/Artificial Intelligence/Data Science & ML Diploma AMIT/05-Computer Vision/Final project of module/Final Project - George/Model/Weights/{self.models_names[model]}')
            if self.image_path:
                img = self.preprocess_image(self.image_path, model)
                if img is not None:
                    try:
                        predictions = self.crnt_ensemble.predict(img)
                        self.results += (predictions[0]/3)
                    except Exception as e:
                        print('error')
                        self.result_label_class.setText(f"Prediction error: {e}")
        class_idx =   np.argmax(self.results) 
        confidence =  np.max(self.results) * 100

        # Update labels with class names and confidence
        self.result_label_class.setText(f'{self.classes[class_idx].title()}')
        self.result_label_class.setStyleSheet("color: green; font-size: 24px; font-weight: bold;")
        self.result_label_confidence.setText(f'{confidence:.2f}%')
        self.result_label_confidence.setStyleSheet("color: green; font-size: 24px; font-weight: bold;") 



    def predict_with_model(self):
        if self.model_name != "Models Ensemble":
            self.load_model()
            if self.image_path:
                img = self.preprocess_image(self.image_path, self.model_name)
                if img is not None:
                    try:
                        predictions = self.model.predict(img)
                        class_idx = np.argmax(predictions[0])
                        confidence = np.max(predictions[0]) * 100

                        # Update labels with class names and confidence
                        self.result_label_class.setText(f'{self.classes[class_idx].title()}')
                        self.result_label_class.setStyleSheet("color: green; font-size: 24px; font-weight: bold;")
                        self.result_label_confidence.setText(f'{confidence:.2f}%')
                        self.result_label_confidence.setStyleSheet("color: green; font-size: 24px; font-weight: bold;")
                    except Exception as e:
                        print('error')
                        self.result_label_class.setText(f"Prediction error: {e}")
            else:
                pass
                self.result_label_class.setText("Upload img first")

        else:
            self.get_ensemble_prediction()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.setWindowTitle('Image Classifier')
    window.setWindowIcon(QIcon('E:/Artificial Intelligence/Data Science & ML Diploma AMIT/05-Computer Vision/Final project of module/Final Project - George/icon.png'))

    window.show()
    sys.exit(app.exec_())
