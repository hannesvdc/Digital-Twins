import torch as pt

if pt.cuda.is_available():
    print("CUDA is available")
    print("Number of devices: ", pt.cuda.device_count())
    device = pt.cuda.current_device()
    print('Device name:', pt.cuda.get_device_name(device))
else:
    print("CUDA is not available")