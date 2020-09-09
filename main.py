import tflite_runtime.interpreter as tflite
import cv2

# # Load the TFLite model and allocate tensors.
# interpreter = tflite.Interpreter(
#     model_path="./models/deeplabv3_257_mv_gpu.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print(output_details)
# print("\n\n")
# print(input_details)

# Open the device at the ID 0
cap = cv2.VideoCapture(1)

# Check whether user selected camera is opened successfully.

if not (cap.isOpened()):
    print("Could not open video device")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('preview', frame)
    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
