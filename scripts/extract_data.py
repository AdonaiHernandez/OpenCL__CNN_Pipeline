import onnx
import numpy as np
from onnx import numpy_helper

MODELO_ONNX = "modelo_sin_1.onnx"
FICHERO_H_SALIDA = "model_params_2.h"

TENSORES_A_EXTRAER = {
    "conv2_weights": "410_quantized",
    "conv2_bias": "412_quantized"
}

def get_c_type(numpy_dtype):
    if numpy_dtype == np.uint8: return "const uint8_t"
    elif numpy_dtype == np.int8: return "const int8_t"
    elif numpy_dtype == np.int32: return "const int32_t"
    elif numpy_dtype == np.float32: return "const float"
    else: raise TypeError(f"Tipo de dato no soportado: {numpy_dtype}")


def export_tensors_to_header():
    print(f"Cargando modelo: {MODELO_ONNX}")
    model = onnx.load(MODELO_ONNX)
    
    initializers = {init.name: numpy_helper.to_array(init) 
                    for init in model.graph.initializer}
    
    print(f"Escribiendo fichero de cabecera: {FICHERO_H_SALIDA}")
    
    with open(FICHERO_H_SALIDA, "w") as f:
        f.write("// Fichero generado automáticamente. No editar manualmente.\n")
        f.write(f"// Modelo de origen: {MODELO_ONNX}\n")
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")

        for var_name, tensor_name in TENSORES_A_EXTRAER.items():
            if tensor_name not in initializers:
                print(f"Aviso: El tensor '{tensor_name}' no se encontró en el modelo.")
                continue
            
            tensor_data = initializers[tensor_name]
            
            if tensor_data.ndim == 4:
                print(f"  > Reordenando tensor de pesos '{tensor_name}' a [cout, kh, kw, cin]")
                tensor_data = tensor_data.transpose((0, 2, 3, 1))

            c_type = get_c_type(tensor_data.dtype)
            flat_data = tensor_data.flatten()
            
            print(f"  > Exportando '{tensor_name}' como '{var_name}' (Tamaño: {len(flat_data)})")
            
            f.write(f"// Tensor: {tensor_name}, Forma Original ONNX: {list(initializers[tensor_name].shape)}\n")
            f.write(f"{c_type} {var_name}[{len(flat_data)}] = {{\n")
            
            for i, val in enumerate(flat_data):
                f.write(str(val))
                if i < len(flat_data) - 1:
                    f.write(", ")
                if (i + 1) % 16 == 0:
                    f.write("\n")
            
            f.write("\n};\n\n")
            
    print(f"\n¡Éxito! El fichero '{FICHERO_H_SALIDA}' ha sido creado. ✅")

if __name__ == "__main__":
    export_tensors_to_header()