import onnx
from onnxsim import simplify

onnx_model = onnx.load("modelo.onnx")

model_simplified, check = simplify(onnx_model)

assert check, "La simplificación del modelo ha fallado."

onnx.save(model_simplified, "tu_modelo_optimizado.onnx")

print("El modelo ha sido optimizado y las capas de BatchNormalization han sido fusionadas.")