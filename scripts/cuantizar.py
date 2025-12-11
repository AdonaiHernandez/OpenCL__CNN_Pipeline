import os
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat

FP32_MODEL_PATH = "modelo_nombres.onnx"
QUANT_MODEL_PATH = "modelo_int8.onnx"

# DATOS DE VALIDACIÓN
CALIBRATION_IMAGE_DIR = "./WIDER_val/images" 
CALIBRATION_GT_FILE = "./wider_face_val_bbx_gt.txt"

NUM_CALIBRATION_IMAGES = 200

# Parámetros del modelo
INPUT_WIDTH = 320
INPUT_HEIGHT = 240

def preprocess_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    img_resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = (img_rgb.astype(np.float32) - 127.0) / 128.0
    input_tensor = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)
    return input_tensor

class FaceDetectionDataReader(CalibrationDataReader):
    # lee la lista de imágenes desde el archivo .txt de anotaciones.
    def __init__(self, image_dir, gt_file_path, input_name):
        self.input_name = input_name
        self.image_paths = []
        
        print(f"Leyendo la lista de imágenes de calibración desde: {gt_file_path}")
        
        with open(gt_file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            relative_path = lines[i].strip()
            full_path = os.path.join(image_dir, relative_path)
            self.image_paths.append(full_path)
            i += 1
            
            # Línea 2: Número de cajas (que debemos saltar)
            num_boxes = int(lines[i].strip())
            i += 1
            
            # Saltamos las líneas de las cajas
            if num_boxes > 0:
                i += num_boxes

        # Limitar al número de imágenes de calibración
        if len(self.image_paths) < NUM_CALIBRATION_IMAGES:
            print(f"Aviso: Se encontraron {len(self.image_paths)} imágenes, menos de las {NUM_CALIBRATION_IMAGES} solicitadas.")
        self.image_paths = self.image_paths[:NUM_CALIBRATION_IMAGES]
        
        print(f"Se usarán {len(self.image_paths)} imágenes para la calibración.")
        self.data_iter = iter(self.image_paths)
        
    def get_next(self):
        image_path = next(self.data_iter, None)
        if image_path:
            input_tensor = preprocess_image(image_path)
            if input_tensor is not None:
                return {self.input_name: input_tensor}
        return None

def main():
    print("Iniciando cuantización estática usando el CONJUNTO DE VALIDACIÓN...")

    model = onnx.load(FP32_MODEL_PATH)
    input_name = model.graph.input[0].name
    print(f"Nombre de la entrada del modelo: {input_name}")

    calibration_data_reader = FaceDetectionDataReader(
        CALIBRATION_IMAGE_DIR, 
        CALIBRATION_GT_FILE, 
        input_name
    )
    
    nodos_to_ignore = ["Concat_374", "Softmax_scores", "Concat_boxes"]

    print("Cuantizando el modelo... (Esto puede tardar unos minutos)")
    quantize_static(
        model_input=FP32_MODEL_PATH,
        model_output=QUANT_MODEL_PATH,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodos_to_ignore
    )
    
    print("\n¡Cuantización completada!")
    print(f"Modelo original (FP32): {FP32_MODEL_PATH} ({os.path.getsize(FP32_MODEL_PATH) / 1024**2:.2f} MB)")
    print(f"Modelo cuantizado (INT8): {QUANT_MODEL_PATH} ({os.path.getsize(QUANT_MODEL_PATH) / 1024**2:.2f} MB)")

if __name__ == '__main__':
    main()