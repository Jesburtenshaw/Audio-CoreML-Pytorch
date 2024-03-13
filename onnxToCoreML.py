import coremltools as ct

inputPath = "./Liza/LISA_simple.onnx"
outputPath = "./Liza/LISA.mlpackage"

ct.converters.onnx.convert(inputPath, outputPath)

