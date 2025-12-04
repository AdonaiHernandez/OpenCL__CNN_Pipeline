import cv2
import numpy as np
import onnxruntime as ort
import math

MODEL_PATH = "modelo_int8_cut.onnx"
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 320
INPUT_HEIGHT = 240
VARIANCES = [0.1, 0.2]

def generate_prior_boxes(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT):
    feature_maps = [[30, 40], [15, 20], [8, 10], [4, 5]]
    min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    steps = [8, 16, 32, 64]
    prior_boxes = []
    for i, fm in enumerate(feature_maps):
        fm_h, fm_w = fm[0], fm[1]
        step = steps[i]
        for h in range(fm_h):
            for w in range(fm_w):
                for min_size in min_sizes[i]:
                    cx = (w + 0.5) * step / input_width
                    cy = (h + 0.5) * step / input_height
                    box_w = float(min_size) / input_width
                    box_h = float(min_size) / input_height
                    prior_boxes.append([cx, cy, box_w, box_h])
    priors = np.array(prior_boxes, dtype=np.float32)
    return np.clip(priors, 0.0, 1.0)

def decode_boxes(raw_boxes, priors):
    decoded_cx_cy = priors[:, :2] + raw_boxes[:, :2] * VARIANCES[0] * priors[:, 2:]
    decoded_w_h = priors[:, 2:] * np.exp(raw_boxes[:, 2:] * VARIANCES[1])
    decoded_boxes = np.concatenate([decoded_cx_cy, decoded_w_h], axis=1)
    decoded_boxes[:, :2] -= decoded_boxes[:, 2:] / 2
    decoded_boxes[:, 2:] += decoded_boxes[:, :2]
    return decoded_boxes

def postprocess_with_priors(frame, scores, raw_boxes, priors):
    orig_h, orig_w, _ = frame.shape
    face_scores = scores[0, :, 1]
    valid_indices = np.where(face_scores > CONFIDENCE_THRESHOLD)[0]
    
    if len(valid_indices) == 0:
        return frame

    filtered_boxes = raw_boxes[0, valid_indices, :]
    filtered_priors = priors[valid_indices, :]
    filtered_scores = face_scores[valid_indices]

    final_boxes_normalized = decode_boxes(filtered_boxes, filtered_priors)
    final_boxes = (final_boxes_normalized * np.array([orig_w, orig_h, orig_w, orig_h])).astype(np.int32)

    boxes_for_nms = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        boxes_for_nms.append([x1, y1, x2 - x1, y2 - y1])
    
    indices_nms = cv2.dnn.NMSBoxes(boxes_for_nms, filtered_scores.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    if len(indices_nms) > 0:
        indices_nms = indices_nms.flatten()
        for i in indices_nms:
            box = boxes_for_nms[i]
            score = filtered_scores[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Cara: {score:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    return frame

# --- Carga del modelo e Inicialización ---
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print("Modelo ONNX "+MODEL_PATH+" cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo ONNX: {e}")
    exit()

print("Generando prior boxes...")
prior_boxes = generate_prior_boxes()
print(f"Se generaron {len(prior_boxes)} prior boxes.")

# --- Bucle principal de la Webcam ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("\nPresiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    img_resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    input_tensor = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), axis=0)

    # Inferencia
    scores, raw_boxes = session.run(output_names, {input_name: input_tensor})

    result_frame = postprocess_with_priors(frame, scores, raw_boxes, prior_boxes)
    
    cv2.imshow('Deteccion de Caras Corregida', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()