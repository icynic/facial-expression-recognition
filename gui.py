import tkinter as tk
from tkinter import ttk
from tkinter import messagebox # Import messagebox for error popups
from PIL import Image, ImageTk
import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier

class ExpressionRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Recognition")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close

        # --- Configuration  ---
        # self.cap_device = 0 # Default will be set after detection
        self.cap_width = 640 # Reduced for smoother GUI display
        self.cap_height = 480
        self.use_brect = True

        # --- State Variables ---
        self.cap = None
        self.is_running = False
        self.update_id = None # To store the id for root.after
        self.available_cameras = [] # To store available camera indices
        self.cap_device = None # Will hold the selected camera index
        self.selected_camera_var = tk.StringVar() # For Combobox
        self.max_faces_var = tk.IntVar(value=5) # Variable for max faces Spinbox
        self.process_freq_var = tk.IntVar(value=1) # Variable for processing frequency
        self.frame_count = 0 # Counter for frames
        self.last_processed_debug_image = None # Store last drawn image
        self.last_detected_face_count = 0 # Store last face count for status

        # --- Load Models and Labels ---
        self.mp_face_mesh_solution = mp.solutions.face_mesh # Store the solution object
        self.face_mesh = None # Initialize face_mesh model to None initially
        self.keypoint_classifier = None
        self.keypoint_classifier_labels = None
        self.load_classifier_and_labels() # Load only classifier and labels initially

        # --- GUI Setup ---
        self.setup_gui()
        self.detect_and_populate_cameras() # Detect cameras after GUI is set up

    def load_classifier_and_labels(self):
        """Loads the custom classification model and labels."""
        try:
            # Custom KeyPoint Classifier
            self.keypoint_classifier = KeyPointClassifier()

            # Read labels
            with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                      encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                self.keypoint_classifier_labels = [row[0] for row in reader]
            print("Classifier and labels loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading classifier/labels: {e}")
            messagebox.showerror("Model Error", f"Failed to load classifier/labels: {e}")
            # Optionally disable start button or show error in GUI
            if hasattr(self, 'start_button'):
                self.start_button.config(state=tk.DISABLED)
            if hasattr(self, 'result_label'):
                self.result_label.config(text=f"Error loading models: {e}")
            return False

    def initialize_face_mesh(self, max_faces):
        """Initializes or re-initializes the MediaPipe Face Mesh model."""
        try:
            # Close existing model if it exists
            if self.face_mesh:
                self.face_mesh.close()
                print("Closed existing Face Mesh model.")

            print(f"Initializing Face Mesh with max_faces={max_faces}...")
            # Initialize MediaPipe Face Mesh
            self.face_mesh = self.mp_face_mesh_solution.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            print("Face Mesh model initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing Face Mesh model: {e}")
            messagebox.showerror("Model Error", f"Failed to initialize Face Mesh: {e}")
            self.face_mesh = None # Ensure it's None on failure
            return False

    def setup_gui(self):
        """Creates and arranges the Tkinter widgets."""
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Create Placeholder Image ---
        # Create a blank black image with the desired dimensions
        placeholder_img = Image.new('RGB', (self.cap_width, self.cap_height), color='black')
        # Convert the PIL image to a Tkinter PhotoImage
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_img)
        # --- End Placeholder ---

        # Video Display Label - Use the placeholder image
        self.video_label = ttk.Label(main_frame, image=self.placeholder_photo, anchor='center')
        self.video_label.image = self.placeholder_photo # Keep a reference!
        # self.video_label.configure(background='black', anchor='center') # No longer needed
        self.video_label.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S)) # Span 3 cols now
        main_frame.rowconfigure(0, weight=1) # Make video area expand
        main_frame.columnconfigure(0, weight=1)

        # Control Frame
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, columnspan=5, pady=5, sticky=(tk.W, tk.E)) # Span 5 cols now

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Camera Selection
        camera_label = ttk.Label(control_frame, text="Camera:")
        camera_label.pack(side=tk.LEFT, padx=(10, 2))

        self.camera_selector = ttk.Combobox(
            control_frame,
            textvariable=self.selected_camera_var,
            state='readonly',
            width=15
        )
        self.camera_selector.bind('<<ComboboxSelected>>', self.on_camera_select)
        self.camera_selector.pack(side=tk.LEFT, padx=2)

        # Max Faces Spinbox
        max_faces_label = ttk.Label(control_frame, text="Max Faces:")
        max_faces_label.pack(side=tk.LEFT, padx=(10, 2))

        self.max_faces_spinbox = ttk.Spinbox(
            control_frame,
            from_=1, # Minimum 1 face
            to=100,   # Maximum 10 faces (adjust as needed)
            textvariable=self.max_faces_var,
            width=5,
            state='readonly' # Or 'normal' if you want typing
        )
        self.max_faces_spinbox.pack(side=tk.LEFT, padx=2)

        # Processing Frequency Spinbox
        process_freq_label = ttk.Label(control_frame, text="Process Freqency (Frames):")
        process_freq_label.pack(side=tk.LEFT, padx=(10, 2))

        self.process_freq_spinbox = ttk.Spinbox(
            control_frame,
            from_=1, # Minimum process every frame
            to=100,   # Maximum process every 10 frames (adjust as needed)
            textvariable=self.process_freq_var,
            width=4,
            state='readonly' # Or 'normal' if you want typing
        )
        self.process_freq_spinbox.pack(side=tk.LEFT, padx=2)

        # Result Label
        self.result_label = ttk.Label(main_frame, text="Status: Initializing...", font=("Helvetica", 14))
        self.result_label.grid(row=2, column=0, columnspan=5, pady=5, sticky=(tk.W, tk.E)) # Span 5 cols

        # Initial state checks (only classifier/labels needed now)
        if not self.keypoint_classifier or not self.keypoint_classifier_labels:
            self.start_button.config(state=tk.DISABLED)
            self.result_label.config(text="Status: Error - Classifier/Labels not loaded.")
        # Camera check happens in detect_and_populate_cameras

    def detect_and_populate_cameras(self):
        """Detects available camera devices and populates the Combobox."""
        print("Detecting cameras...")
        self.available_cameras = []
        index = 0
        while True:
            cap = cv.VideoCapture(index, cv.CAP_DSHOW) # Use CAP_DSHOW for better compatibility on Windows
            if cap.isOpened():
                self.available_cameras.append(index)
                cap.release()
                print(f"Found camera: {index}")
                index += 1
                if index > 5: # Check only first few indices to avoid long waits
                    break
            else:
                cap.release()
                print(f"No camera found at index: {index}")
                # Stop if we fail twice in a row after finding at least one
                if index > 0 and not self.available_cameras:
                     break # No cameras found
                elif index > self.available_cameras[-1] + 1 and len(self.available_cameras) > 0:
                     break # Stop searching if we miss one after finding some
                elif index > 5: # Safety break
                    break
                index += 1 # Check next index even if this one failed, unless criteria above met


        camera_options = [f"Camera {i}" for i in self.available_cameras]

        if camera_options:
            self.camera_selector['values'] = camera_options
            self.cap_device = self.available_cameras[0]
            self.selected_camera_var.set(camera_options[0])
            print(f"Default camera set to: {self.cap_device}")
            # Enable start button only if classifier/labels are also loaded
            if self.keypoint_classifier and self.keypoint_classifier_labels:
                self.start_button.config(state=tk.NORMAL)
            else:
                self.start_button.config(state=tk.DISABLED)
        else:
            print("No cameras detected.")
            self.selected_camera_var.set("No cameras found")
            self.camera_selector['values'] = []
            self.start_button.config(state=tk.DISABLED) # Disable start if no cameras
            self.result_label.config(text="Status: Error - No cameras detected.")

    def on_camera_select(self, event=None):
        """Handles the event when a camera is selected from the Combobox."""
        selected_option = self.selected_camera_var.get()
        try:
            self.cap_device = int(selected_option.split()[-1])
            print(f"Selected camera device: {self.cap_device}")
            # If capture was stopped, ensure start is enabled (if classifier/labels ok)
            if not self.is_running and self.keypoint_classifier and self.keypoint_classifier_labels:
                 self.start_button.config(state=tk.NORMAL)
        except (IndexError, ValueError):
            print(f"Error parsing camera selection: {selected_option}")
            self.cap_device = None
            self.start_button.config(state=tk.DISABLED)


    def start_capture(self):
        """Initializes FaceMesh, the selected camera, and starts the loop."""
        if self.is_running:
            print("Capture already running.")
            return

        if self.cap_device is None:
            messagebox.showerror("Camera Error", "No valid camera selected.")
            self.result_label.config(text="Status: Error - Select a camera.")
            return

        if not self.keypoint_classifier or not self.keypoint_classifier_labels:
             messagebox.showerror("Model Error", "Classifier or labels not loaded.")
             self.result_label.config(text="Status: Error - Models not loaded.")
             return

        # --- Initialize Face Mesh Model ---
        max_faces = self.max_faces_var.get()
        if not self.initialize_face_mesh(max_faces):
            self.result_label.config(text="Status: Error - Failed to init Face Mesh.")
            self.start_button.config(state=tk.NORMAL)
            return # Stop if Face Mesh failed to initialize
        # --- End Face Mesh Init ---

        print(f"Attempting to open camera device: {self.cap_device}")
        self.cap = cv.VideoCapture(self.cap_device, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera device {self.cap_device}")
            messagebox.showerror("Camera Error", f"Could not open camera {self.cap_device}.")
            self.result_label.config(text=f"Status: Error - Cannot open camera {self.cap_device}")
            self.cap = None
            # Release FaceMesh model if camera failed
            if self.face_mesh:
                self.face_mesh.close()
                self.face_mesh = None
            self.start_button.config(state=tk.NORMAL) # Allow trying again
            self.stop_button.config(state=tk.DISABLED)
            self.camera_selector.config(state='readonly')
            self.max_faces_spinbox.config(state='readonly') # Re-enable spinbox on failure
            self.process_freq_spinbox.config(state='readonly') # Re-enable freq spinbox
            return

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        print("Camera started.")

        self.is_running = True
        self.frame_count = 0 # Reset frame counter
        self.last_processed_debug_image = None # Reset last image
        self.last_detected_face_count = 0 # Reset last count
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.camera_selector.config(state='disabled') # Disable selector while running
        self.max_faces_spinbox.config(state='disabled') # Disable spinbox while running
        self.process_freq_spinbox.config(state='disabled') # Disable freq spinbox while running
        self.result_label.config(text="Status: Running...")

        self.update_frame() # Start the loop

    def stop_capture(self):
        """Stops the frame update loop, releases camera and FaceMesh model."""
        if not self.is_running:
            print("Capture is not running.")
            return

        print("Stopping capture...")
        self.is_running = False
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None

        if self.cap:
            self.cap.release()
            self.cap = None
            print("Camera released.")

        # Close the FaceMesh model when stopping
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
            print("Face Mesh model closed.")


        # Re-enable start button only if a valid camera is selected and classifier/labels loaded
        if self.cap_device is not None and self.keypoint_classifier and self.keypoint_classifier_labels:
            self.start_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.DISABLED)

        self.stop_button.config(state=tk.DISABLED)
        self.camera_selector.config(state='readonly') # Re-enable selector
        self.max_faces_spinbox.config(state='readonly') # Re-enable spinbox
        self.process_freq_spinbox.config(state='readonly') # Re-enable freq spinbox

        # --- Reset video label to placeholder ---
        if hasattr(self, 'placeholder_photo'):
            self.video_label.config(image=self.placeholder_photo, text="") # Show placeholder again
            self.video_label.image = self.placeholder_photo # Update reference
        else:
             # Fallback if placeholder wasn't created (shouldn't happen with the setup_gui change)
             self.video_label.config(image='', text="Video Feed Stopped", background='grey')
        # --- End Reset ---
        self.result_label.config(text="Status: Stopped")
        self.last_processed_debug_image = None # Clear last image
        self.last_detected_face_count = 0


    def update_frame(self):
        """Reads a frame, processes it based on frequency, and updates the GUI."""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            print("Update loop stopping: Not running or camera issue.")
            # Simplified error handling on loop stop if running
            if self.is_running:
                 self.stop_capture()
                 self.result_label.config(text="Status: Error - Camera disconnected?")
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            self.stop_capture()
            self.result_label.config(text="Status: Error - Failed to read from camera.")
            return

        # Increment frame counter
        self.frame_count += 1
        process_this_frame = (self.frame_count % self.process_freq_var.get() == 0)

        debug_image = None # Initialize debug_image

        if process_this_frame:
            # --- Core Processing Logic ---
            # Ensure FaceMesh is initialized before processing
            if not self.face_mesh:
                print("Update loop stopping: FaceMesh not initialized.")
                self.stop_capture()
                self.result_label.config(text="Status: Error - FaceMesh model error.")
                return

            frame_flipped = cv.flip(frame, 1)
            current_debug_image = copy.deepcopy(frame_flipped)
            num_faces_detected = 0

            image_rgb = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.face_mesh.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.multi_face_landmarks is not None:
                num_faces_detected = len(results.multi_face_landmarks)
                for face_landmarks in results.multi_face_landmarks:
                    brect = self.calc_bounding_rect(current_debug_image, face_landmarks)
                    if brect is None: continue
                    landmark_list = self.calc_landmark_list(current_debug_image, face_landmarks)
                    pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                    facial_emotion_id = self.keypoint_classifier(pre_processed_landmark_list)
                    current_expression = self.keypoint_classifier_labels[facial_emotion_id] if facial_emotion_id < len(self.keypoint_classifier_labels) else "Unknown ID"
                    current_debug_image = self.draw_bounding_rect(self.use_brect, current_debug_image, brect)
                    current_debug_image = self.draw_info_text(current_debug_image, brect, current_expression)
            else:
                # Keep num_faces_detected as 0 if no landmarks found
                num_faces_detected = 0


            # Store the processed image and face count
            self.last_processed_debug_image = current_debug_image
            self.last_detected_face_count = num_faces_detected
            debug_image = current_debug_image # Use the newly processed image
            # --- End Core Processing Logic ---
        else:
            # Use the last processed image if available, otherwise use the raw flipped frame
            if self.last_processed_debug_image is not None:
                debug_image = self.last_processed_debug_image
            else:
                # If no frame has been processed yet, just show the flipped raw frame
                debug_image = cv.flip(frame, 1)


        # --- Update GUI ---
        if debug_image is not None:
            img_rgb_for_pil = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb_for_pil)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk, text="")
        else:
             # Fallback: If somehow debug_image is still None, show raw flipped frame
             frame_flipped = cv.flip(frame, 1)
             img_rgb_for_pil = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)
             pil_image = Image.fromarray(img_rgb_for_pil)
             imgtk = ImageTk.PhotoImage(image=pil_image)
             self.video_label.imgtk = imgtk
             self.video_label.config(image=imgtk, text="")


        # Update status label based on the last *detection* result
        status_text = "Status: "
        if self.last_detected_face_count > 0:
            status_text += f"{self.last_detected_face_count} face(s) detected"
        else:
             status_text += "No face detected"

        self.result_label.config(text=status_text)

        if self.is_running:
            # Schedule next update - adjust delay slightly if needed based on processing time?
            # Keeping 15ms might be fine, actual frame rate will depend on processing cost.
            self.update_id = self.root.after(15, self.update_frame)

    def on_closing(self):
        """Called when the window is closed."""
        print("Closing application...")
        self.stop_capture() # Ensure camera and model are released
        # self.face_mesh.close() is now handled within stop_capture
        self.root.destroy()

    # --- Helper Functions (from original script, slightly adapted to be methods) ---
    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        # Convert to NumPy array
        landmark_array = np.array(landmark_list, dtype=np.float32)

        # Check if the array is empty or has only one point (edge case)
        if landmark_array.shape[0] < 1:
            return np.array([], dtype=np.float32) # Return empty if no landmarks
        if landmark_array.shape[0] == 1:
            # If only one landmark, translation/normalization doesn't make sense
            # Return a zero vector of the expected flattened size (2)
            return np.zeros(2, dtype=np.float32)


        # Translation: Subtract the first landmark's coordinates
        base_point = landmark_array[0]
        translated_landmarks = landmark_array - base_point

        # Flatten the translated array
        flattened_landmarks = translated_landmarks.flatten()

        # Normalization: Divide by the maximum absolute value
        max_abs_value = np.max(np.abs(flattened_landmarks))

        # Avoid division by zero
        if max_abs_value == 0:
            # If max abs value is 0, all relative points are (0,0)
            normalized_landmarks = np.zeros_like(flattened_landmarks)
        else:
            normalized_landmarks = flattened_landmarks / max_abs_value

        return normalized_landmarks

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect and brect: # Check if brect is valid
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2) # Changed color/thickness
        return image

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        # Check if landmarks are valid before proceeding
        if not landmarks or not landmarks.landmark:
             return None # Return None if no landmarks
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        # Check if landmark_array is empty before calling boundingRect
        if landmark_array.shape[0] == 0:
             return None # Return None if array is empty
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def draw_info_text(self, image, brect, facial_text):
         if not brect: # Check if brect is valid
             return image
         # Draw background rectangle above the bounding box
         text_bg_y1 = brect[1] - 22 if brect[1] > 22 else 0 # Ensure it doesn't go offscreen top
         text_bg_y2 = brect[1]
         cv.rectangle(image, (brect[0], text_bg_y1), (brect[2], text_bg_y2), (0, 255, 0), -1) # Match rect color

         # Prepare text
         info_text = f"Emotion: {facial_text}" if facial_text else "Emotion: N/A"
         # Calculate text position
         text_x = brect[0] + 5
         text_y = text_bg_y2 - 4 # Position inside the background rect

         # Draw text
         cv.putText(image, info_text, (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA) # Black text on green bg
         return image


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ExpressionRecognizerApp(root)
    root.mainloop()