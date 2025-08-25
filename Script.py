import tkinter as tk
from tkinter import scrolledtext, ttk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading, queue

# -------------------- Model config --------------------
MODEL_PATH = "Mobilenetv2_ISL_model.h5"
INPUT_W = INPUT_H = 224  # your model input
# Fill this with your actual label order:
CLASS_NAMES = [
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I',
    'J','K','L','M','N','O','P','Q','R',
    'S','T','U','V','W','X','Y','Z'
]

model = tf.keras.models.load_model(MODEL_PATH)

# -------------------- MediaPipe Hands (hands only) --------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- Video capture --------------------
cap = cv2.VideoCapture(0)

# -------------------- Speech (pyttsx3) in a background worker --------------------
# Using a single engine in a dedicated thread avoids run-loop conflicts.
import pyttsx3

class TTSManager:
    def __init__(self):
        self.is_speaking = False
        print("Simple TTS Manager initialized")
    
    def speak(self, text):
        """Speak text using a simple, reliable approach"""
        if not text or not text.strip():
            print("No text to speak")
            return
        
        def _speak_in_thread():
            try:
                self.is_speaking = True
                print(f"Starting to speak: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Create a fresh engine for each speech request
                engine = pyttsx3.init()
                
                # Configure the engine
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                # Get and set voice
                voices = engine.getProperty('voices')
                if voices and len(voices) > 0:
                    engine.setProperty('voice', voices[0].id)
                
                # Speak the text
                engine.say(text)
                engine.runAndWait()
                
                # Clean up
                engine.stop()
                del engine
                
                print("Speech completed successfully")
                self.is_speaking = False
                
            except Exception as e:
                print(f"Speech error: {e}")
                self.is_speaking = False
                
                # Try alternative approach if pyttsx3 fails
                try:
                    import os
                    # Use Windows built-in speech on Windows
                    if os.name == 'nt':
                        import subprocess
                        # Use PowerShell's speech synthesis
                        ps_cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
                        subprocess.run(["powershell", "-Command", ps_cmd], check=True, capture_output=True)
                        print("Speech completed using PowerShell fallback")
                except Exception as fallback_error:
                    print(f"Fallback speech also failed: {fallback_error}")
        
        # Run speech in a separate thread
        speech_thread = threading.Thread(target=_speak_in_thread, daemon=True)
        speech_thread.start()
    
    def stop_speech(self):
        """Stop current speech (simplified)"""
        print("Speech stop requested (will complete current phrase)")
        self.is_speaking = False
    
    def is_engine_speaking(self):
        """Check if engine is currently speaking"""
        return self.is_speaking
    
    def shutdown(self):
        """Shutdown the TTS manager"""
        print("TTS Manager shutdown")
        self.is_speaking = False

tts_manager = TTSManager()

# -------------------- Modern Tkinter App --------------------
class SignApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_styles()
        self.create_widgets()
        self.setup_layout()
        
        # State management
        self.last_label = None
        self.frames_since_last_insert = 0
        self.is_speaking = False
        self.detection_enabled = True
        
        # Start the main loop
        self.update_frame()
    
    def setup_window(self):
        """Configure the main window with modern styling"""
        self.root.title("🤟 AI Sign Language Translator")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.configure(bg='#f0f0f0')
        
        # Set icon if available
        try:
            self.root.iconbitmap(default='icon.ico')
        except:
            pass
    
    def create_styles(self):
        """Create modern ttk styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure modern button styles
        self.style.configure('Speak.TButton', 
                           background='#4CAF50', 
                           foreground='white',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 10))
        
        self.style.configure('Clear.TButton', 
                           background='#FF9800', 
                           foreground='white',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 10))
        
        self.style.configure('Stop.TButton', 
                           background='#f44336', 
                           foreground='white',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 10))
        
        self.style.configure('Toggle.TButton', 
                           background='#2196F3', 
                           foreground='white',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 10))
    
    def create_widgets(self):
        """Create all UI widgets with modern styling"""
        # Main container with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # Title label
        self.title_label = tk.Label(
            self.main_frame, 
            text="AI Sign Language to Text Translator", 
            font=("Segoe UI", 16, "bold"), 
            bg='#f0f0f0', 
            fg='#333333'
        )
        
        # Camera frames with modern styling
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="📹 Camera Views", padding="10")
        
        self.frame_cam_left = tk.Frame(self.camera_frame, bg='black', relief='raised', bd=2)
        self.frame_cam_right = tk.Frame(self.camera_frame, bg='black', relief='raised', bd=2)
        
        # Camera labels with placeholder
        self.cam_left_lbl = tk.Label(
            self.frame_cam_left, 
            text="Main Camera View\n(Waiting for camera...)", 
            bg='black', 
            fg='white',
            font=('Segoe UI', 12)
        )
        self.cam_right_lbl = tk.Label(
            self.frame_cam_right, 
            text="Hand Detection View\n(ROI will appear here)", 
            bg='black', 
            fg='white',
            font=('Segoe UI', 12)
        )
        
        # Text area with modern styling
        self.text_frame = ttk.LabelFrame(self.main_frame, text="📝 Detected Text", padding="10")
        
        self.text_box = scrolledtext.ScrolledText(
            self.text_frame, 
            width=70, 
            height=8, 
            font=("Consolas", 14),
            bg='#ffffff',
            fg='#333333',
            selectbackground='#0078d4',
            selectforeground='white',
            wrap=tk.WORD
        )
        
        # Status bar
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_label = tk.Label(
            self.status_frame,
            text="✅ Ready - Show your hand signs to the camera",
            font=('Segoe UI', 10),
            bg='#f0f0f0',
            fg='#28a745'
        )
        
        # Control buttons with modern styling
        self.button_frame = ttk.LabelFrame(self.main_frame, text="🎛️ Controls", padding="10")
        
        self.speak_btn = ttk.Button(
            self.button_frame, 
            text="🔊 Speak Text", 
            style='Speak.TButton',
            command=self.on_speech
        )
        
        self.stop_btn = ttk.Button(
            self.button_frame, 
            text="⏹️ Stop Speaking", 
            style='Stop.TButton',
            command=self.on_stop_speech
        )
        
        self.clear_btn = ttk.Button(
            self.button_frame, 
            text="🗑️ Clear Text", 
            style='Clear.TButton',
            command=self.on_clear
        )
        
        self.toggle_btn = ttk.Button(
            self.button_frame, 
            text="⏸️ Pause Detection", 
            style='Toggle.TButton',
            command=self.toggle_detection
        )
        
        self.quit_btn = ttk.Button(
            self.button_frame, 
            text="❌ Quit", 
            style='Stop.TButton',
            command=self.on_quit
        )
        
        # Add keyboard shortcuts
        self.root.bind('<Control-s>', lambda e: self.on_speech())
        self.root.bind('<Control-c>', lambda e: self.on_clear())
        self.root.bind('<Control-p>', lambda e: self.toggle_detection())
        self.root.bind('<Escape>', lambda e: self.on_stop_speech())
    
    def setup_layout(self):
        """Setup the modern responsive layout"""
        # Main frame
        self.main_frame.grid(row=0, column=0, sticky='nsew')
        
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Configure main frame grid
        self.main_frame.grid_rowconfigure(1, weight=2)  # Camera area
        self.main_frame.grid_rowconfigure(2, weight=1)  # Text area
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Layout widgets
        self.title_label.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        # Camera section
        self.camera_frame.grid(row=1, column=0, sticky='nsew', pady=(0, 10))
        self.camera_frame.grid_columnconfigure(0, weight=1)
        self.camera_frame.grid_columnconfigure(1, weight=1)
        self.camera_frame.grid_rowconfigure(0, weight=1)
        
        self.frame_cam_left.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        self.frame_cam_right.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        self.cam_left_lbl.pack(fill='both', expand=True)
        self.cam_right_lbl.pack(fill='both', expand=True)
        
        # Text section
        self.text_frame.grid(row=2, column=0, sticky='nsew', pady=(0, 10))
        self.text_frame.grid_columnconfigure(0, weight=1)
        self.text_frame.grid_rowconfigure(0, weight=1)
        
        self.text_box.grid(row=0, column=0, sticky='nsew')
        
        # Status section
        self.status_frame.grid(row=3, column=0, sticky='ew', pady=(0, 5))
        self.status_label.pack(side='left')
        
        # Button section
        self.button_frame.grid(row=4, column=0, sticky='ew')
        
        self.speak_btn.grid(row=0, column=0, padx=(0, 10), pady=5)
        self.stop_btn.grid(row=0, column=1, padx=(0, 10), pady=5)
        self.clear_btn.grid(row=0, column=2, padx=(0, 10), pady=5)
        self.toggle_btn.grid(row=0, column=3, padx=(0, 10), pady=5)
        self.quit_btn.grid(row=0, column=4, padx=(0, 10), pady=5)
    
    def update_status(self, message, color='#28a745'):
        """Update the status bar with a message"""
        self.status_label.configure(text=message, fg=color)
    
    def on_speech(self):
        """Speech function that reads from the detected text box"""
        # Get text from the text box ("1.0" means line 1, character 0 = start)
        text = self.text_box.get("1.0", tk.END)
        
        # Remove whitespace and newlines
        text = text.strip()
        
        # Debug: print what we're trying to speak
        print(f"DEBUG: Text box content: '{text}'")
        print(f"DEBUG: Text length: {len(text)}")
        
        # Check if there's any text
        if not text or len(text) == 0:
            self.update_status("⚠️ No text in the text box to speak!", '#ffc107')
            print("DEBUG: No text found in text box")
            return
        
        # Try to speak the text
        try:
            print(f"DEBUG: Calling TTS with: '{text}'")
            tts_manager.speak(text)
            self.update_status(f"🔊 Speaking: {text[:30]}{'...' if len(text) > 30 else ''}")
            self.is_speaking = True
            print("DEBUG: Speech request sent successfully")
        except Exception as e:
            error_msg = f"❌ Speech error: {str(e)}"
            self.update_status(error_msg, '#dc3545')
            print(f"DEBUG: Speech error: {e}")
    
    def on_stop_speech(self):
        """Stop current speech"""
        try:
            tts_manager.stop_speech()
            self.update_status("⏹️ Speech stopped", '#6c757d')
            self.is_speaking = False
        except Exception as e:
            self.update_status(f"❌ Error stopping speech: {str(e)}", '#dc3545')
    
    def on_clear(self):
        """Enhanced clear function with confirmation for large text"""
        current_text = self.text_box.get("1.0", tk.END).strip()
        
        if not current_text:
            self.update_status("ℹ️ Text area is already empty", '#6c757d')
            return
            
        # Stop any current speech before clearing
        if self.is_speaking:
            self.on_stop_speech()
        
        # Clear the text
        self.text_box.delete("1.0", tk.END)
        
        # Reset detection state
        self.last_label = None
        self.frames_since_last_insert = 0
        
        self.update_status("🗑️ Text cleared successfully", '#28a745')
    
    def toggle_detection(self):
        """Toggle hand detection on/off"""
        self.detection_enabled = not self.detection_enabled
        
        if self.detection_enabled:
            self.toggle_btn.configure(text="⏸️ Pause Detection", style='Toggle.TButton')
            self.update_status("▶️ Detection resumed", '#28a745')
        else:
            self.toggle_btn.configure(text="▶️ Resume Detection", style='Speak.TButton')
            self.update_status("⏸️ Detection paused", '#ffc107')
    
    def on_quit(self):
        """Clean shutdown with proper resource cleanup"""
        self.update_status("🔄 Shutting down...")
        
        try:
            # Stop speech
            tts_manager.shutdown()
            
            # Release camera
            if cap.isOpened():
                cap.release()
            
            # Close windows
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.root.quit()
            self.root.destroy()

    def update_frame(self):
        """Enhanced frame processing with better error handling and detection toggle"""
        try:
            ok, frame = cap.read()
            if not ok:
                # Camera error - show error message
                self.update_status("❌ Camera error - Check camera connection", '#dc3545')
                self.root.after(100, self.update_frame)  # Retry less frequently
                return
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Add status overlay to frame
            status_text = "PAUSED" if not self.detection_enabled else "DETECTING"
            status_color = (0, 165, 255) if not self.detection_enabled else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            roi_disp = None  # image to show on the right pane
            
            # Only process detection if enabled
            if self.detection_enabled:
                try:
                    # Detect hands
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    
                    if res.multi_hand_landmarks:
                        xs, ys = [], []
                        for hand_lms in res.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                            for lm in hand_lms.landmark:
                                xs.append(int(lm.x * w))
                                ys.append(int(lm.y * h))
                        
                        if xs and ys:  # Ensure we have landmarks
                            # Unified bounding box for hands
                            pad = 30  # Increased padding for better capture
                            x_min = max(min(xs) - pad, 0)
                            y_min = max(min(ys) - pad, 0)
                            x_max = min(max(xs) + pad, w)
                            y_max = min(max(ys) + pad, h)
                            
                            # Draw bounding rectangle
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            
                            # Add confidence display
                            cv2.putText(frame, "Hand Detected", (x_min, y_min-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            crop = frame[y_min:y_max, x_min:x_max]
                            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                                try:
                                    # Show ROI in top-right pane
                                    target_size = min(w // 2, h // 2)
                                    roi_disp = cv2.resize(crop, (target_size, target_size))
                                    
                                    # Prepare for model prediction
                                    roi_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    roi_resized = cv2.resize(roi_rgb, (INPUT_W, INPUT_H), 
                                                           interpolation=cv2.INTER_AREA)
                                    x_input = (roi_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
                                    
                                    # Make prediction
                                    probs = model.predict(x_input, verbose=0)[0]
                                    idx = int(np.argmax(probs))
                                    conf = float(probs[idx])
                                    
                                    # Display prediction and confidence on camera feed
                                    if conf >= 0.3:  # Show predictions above 30% confidence
                                        pred_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
                                        
                                        # Add prediction text above the bounding box
                                        pred_text = f"Pred: {pred_label} ({conf:.2f})"
                                        cv2.putText(frame, pred_text, (x_min, y_min-40), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                        
                                        # Add confidence bar
                                        bar_width = int((x_max - x_min) * conf)
                                        cv2.rectangle(frame, (x_min, y_min-25), (x_min + bar_width, y_min-15), 
                                                    (0, 255, 0) if conf >= 0.7 else (0, 255, 255), -1)
                                        cv2.rectangle(frame, (x_min, y_min-25), (x_max, y_min-15), (255, 255, 255), 2)
                                    
                                    # Higher confidence threshold for better accuracy
                                    if conf >= 0.90:  # Slightly lower threshold for better responsiveness
                                        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
                                        
                                        # Enhanced duplicate suppression
                                        should_insert = (label != self.last_label or 
                                                       self.frames_since_last_insert > 15)  # Increased frames
                                        
                                        if should_insert:
                                            self.text_box.insert(tk.END, label)
                                            self.text_box.see(tk.END)
                                            self.last_label = label
                                            self.frames_since_last_insert = 0
                                            
                                            # Update status with detected character
                                            self.update_status(f"✨ Detected: {label} (confidence: {conf:.2f})")
                                        
                                    self.frames_since_last_insert += 1
                                    
                                except Exception as e:
                                    print(f"Model prediction error: {e}")
                                    self.update_status(f"⚠️ Prediction error: {str(e)[:50]}...", '#ffc107')
                    
                    else:
                        # No hands detected
                        self.frames_since_last_insert = 0
                        self.last_label = None
                        
                except Exception as e:
                    print(f"Hand detection error: {e}")
                    self.update_status(f"⚠️ Detection error: {str(e)[:50]}...", '#ffc107')
            
            else:
                # Detection is paused
                self.frames_since_last_insert = 0
                self.last_label = None
            
            # ----- Update GUI images with error handling -----
            try:
                # Left: full annotated frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize frame to fit window better
                display_size = (400, 300)
                frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
                
                left_img = ImageTk.PhotoImage(frame_pil)
                self.cam_left_lbl.configure(image=left_img, text="")
                self.cam_left_lbl.imgtk = left_img  # Keep reference
                
                # Right: ROI (or mirror of left if no ROI)
                if roi_disp is not None:
                    roi_rgb = cv2.cvtColor(roi_disp, cv2.COLOR_BGR2RGB)
                    roi_pil = Image.fromarray(roi_rgb)
                    roi_pil = roi_pil.resize(display_size, Image.Resampling.LANCZOS)
                    right_img = ImageTk.PhotoImage(roi_pil)
                    self.cam_right_lbl.configure(image=right_img, text="")
                else:
                    # Show mirror of main view
                    right_img = ImageTk.PhotoImage(frame_pil)
                    self.cam_right_lbl.configure(image=right_img, text="")
                
                self.cam_right_lbl.imgtk = right_img  # Keep reference
                
            except Exception as e:
                print(f"GUI update error: {e}")
                self.update_status(f"⚠️ Display error: {str(e)[:50]}...", '#ffc107')
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            self.update_status(f"❌ Processing error: {str(e)[:50]}...", '#dc3545')
        
        finally:
            # Schedule next update (slightly slower for better performance)
            self.root.after(15, self.update_frame)

# -------------------- Run app --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignApp(root)
    root.mainloop()
