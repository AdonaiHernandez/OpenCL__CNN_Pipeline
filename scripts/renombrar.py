import onnx

OUTPUT_TENSORS_TO_EXCLUDE = [
    "374", "boxes", "scores"
]


def run_quantization_with_auto_naming(model_in_path, model_out_path, tensors_to_exclude):
    """
    Función principal que nombra nodos, genera la lista de exclusión y cuantiza.
    """
    
    print("--- Analizando y nombrando nodos ---")
    
    temp_named_model_path = model_out_path
    model = onnx.load(model_in_path)
    
    node_exclusion_list = []
    
    for node in model.graph.node:
        if not node.name:
            node.name = f"{node.op_type}_{node.output[0]}"
            print(f"Nodo sin nombre encontrado. Asignando nuevo nombre: '{node.name}'")
    
        if any(output_name in tensors_to_exclude for output_name in node.output):
            if node.name not in node_exclusion_list:
                node_exclusion_list.append(node.name)

    onnx.save(model, temp_named_model_path)
    print(f"\nModelo temporal con nombres guardado en: {temp_named_model_path}")
    
run_quantization_with_auto_naming("tu_modelo_optimizado.onnx", "modelo_nombres.onnx", OUTPUT_TENSORS_TO_EXCLUDE)