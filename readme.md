py -m venv venv  (venv creation)

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned  (for first time activating vnv)

pip install -r req.txt  (inside venv)

pip list

py train.py

(https://aka.ms/vs/17/release/vc_redist.x64.exe) needed to install this as for runtime of TensorFlow, PyTorch, OpenCV, NumPy

from tensorflow.keras.models import load_model  (in line 5 test.py)
py test.py

py app.py (run flask project created)