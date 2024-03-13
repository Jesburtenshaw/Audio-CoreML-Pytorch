# Soundlabs README

### Adding models
Add "LISA.pth" to this root folder. If you want to place it elsewhere, modify "coreml_convert.py" to point to a different location (or model).

### Running this repo

The easiest way to run this repo is to open this folder in Pycharm and create a local Python environment for it.

PyCharm should detect all missing packages from "requirements.txt" and offer to install them.

"requirements.txt" has been patched from RVC to remove runtime warnings and errors.

Open "coreml_convert.py" as the current file in PyCharm. A "Current File" indicator should show up next to the Run button. Run this file.

The file will output "rvc.mlpackage". After this is created, running "coreml_inference" will verify that the model runs.
