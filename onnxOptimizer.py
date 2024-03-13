import onnxsim
import onnx
def optimize_onnx_model(model_path, output_path):
    optimized_model, check = onnxsim.simplify(model_path)
    if check:
        onnx.save_model(optimized_model, output_path)
    else:
        print("Failed to optimize the ONNX model.")

optimize_onnx_model("Liza/rmvpe.onnx", "Liza/rmvpe_opt.onnx")