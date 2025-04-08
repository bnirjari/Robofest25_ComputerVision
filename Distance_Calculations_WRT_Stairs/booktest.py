import cv2

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Calculate the center of the frame
    center_x = width // 2
    center_y = height // 2

    # Draw a small circle at the center of the frame
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Display the center coordinates on the frame
    cv2.putText(frame, f"Center: ({center_x}, {center_y})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the center marker
    cv2.imshow("Frame with Center", frame)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Default Resolution: {int(width)}x{int(height)}")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
