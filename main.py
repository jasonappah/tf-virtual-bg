import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(
    model_path="./models/deeplabv3_257_mv_gpu.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(output_details)
print("\n\n")
print(input_details)

# Open the device at the ID 0
cap = cv2.VideoCapture(0)

# Check whether user selected camera is opened successfully.

if not (cap.isOpened()):
    print("Could not open video device")

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('preview', frame)
    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  
  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def segment(frame):
    tmp=classify_image(interpreter, frame)
    return tmp
    
resized=cv2.resize(frame, (257,257))    
tmp2=segment(resized)
print(tmp2)

while True:
    cv2.imshow("Segment!", tmp2)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
