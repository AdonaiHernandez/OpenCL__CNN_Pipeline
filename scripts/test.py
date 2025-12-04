import cv2
import numpy as np
import onnxruntime as ort

def preprocess_image(image_path, input_shape, input_dtype):
    """
    Carga una imagen y la pre-procesa para que coincida con los requisitos de entrada del modelo.
    """

    _, _, height, width = input_shape
    
    # 1. Cargar la imagen usando OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {image_path}")
        
    print(f"Imagen original cargada con forma: {img.shape}") # (H, W, C)

    # 2. Cambiar el tamaño de la imagen al que espera el modelo
    img_resized = cv2.resize(img, (width, height))
    
    # 3. Transponer las dimensiones de (H, W, C) a (C, H, W)
    img_transposed = img_resized.transpose(2, 0, 1)

    # 4. Añadir una dimensión de batch al principio -> (1, C, H, W)
    input_tensor = np.expand_dims(img_transposed, axis=0)

    # 5. Asegurarse de que el tipo de dato es el correcto (ej. uint8 para modelos cuantizados)
    input_tensor = input_tensor.astype(input_dtype)
    
    print(f"Tensor de entrada preparado con forma: {input_tensor.shape} y tipo: {input_tensor.dtype}")
    
    return input_tensor

def run_inference_and_visualize(model_path, image_path):
    """
    Ejecuta la inferencia en una imagen, guarda los resultados y visualiza el primer canal.
    """
    # 1. Cargar el modelo y crear una sesión de inferencia
    try:
        session = ort.InferenceSession(model_path)
        print("✅ Modelo ONNX cargado correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    # 2. Obtener información de la entrada del modelo
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    
    # Mapear el tipo de dato ONNX a NumPy
    onnx_to_numpy_type = {
        'tensor(uint8)': np.uint8,
        'tensor(int8)': np.int8,
        'tensor(float)': np.float32,
    }
    input_dtype = onnx_to_numpy_type.get(input_meta.type, np.float32)

    print(f"El modelo espera una entrada llamada '{input_name}' con forma {input_shape} y tipo {input_dtype}.")

    # 3. Pre-procesar la imagen
    input_tensor = preprocess_image(image_path, input_shape, input_dtype)

    # 4. Ejecutar la inferencia
    print("\nEjecutando inferencia...")
    outputs = session.run(None, {input_name: input_tensor})
    output_tensor = outputs[0] # El resultado es una lista, tomamos el primer elemento
    print(f"Inferencia completada. Tensor de salida obtenido con forma: {output_tensor.shape}")

    # 5. Guardar el tensor de salida en archivos
    np.save('output_tensor.npy', output_tensor)
    print("✅ Tensor de salida guardado en 'output_tensor.npy'")
    
    # Para el archivo de texto, "aplanamos" las dimensiones para que sea más legible
    output_reshaped = output_tensor.reshape(output_tensor.shape[0], -1)
    np.savetxt('output_tensor.txt', output_reshaped, fmt='%d')
    print("✅ Tensor de salida guardado en 'output_tensor.txt'")

    # 6. Visualizar el primer canal del resultado
    # Extraer el primer (y único) batch
    first_batch_output = output_tensor[0, :, :, :]
    # Extraer el primer canal
    first_channel = first_batch_output[2, :, :]
    
    print(f"\nVisualizando el primer canal de salida (forma: {first_channel.shape}).")
    

    channel_normalized = cv2.normalize(first_channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Primer Canal de Salida', channel_normalized)
    print("Presiona cualquier tecla en la ventana de la imagen para cerrar.")
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    MODEL_PATH = "tres_primeros_nodos.onnx"  
    IMAGE_PATH = "img.jpg"            

    run_inference_and_visualize(MODEL_PATH, IMAGE_PATH)