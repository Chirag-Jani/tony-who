# JARVIS hand HUD

Webcam + hand tracking: place anchors, draw lines, rectangles, and circles in an Iron Man–style overlay, plus an optional wireframe “tactical” globe.

## Setup

- Python 3.10+ recommended  
- Webcam  

```bash
pip install -r requirements.txt
```

(Minimum to run: `opencv-python`, `mediapipe`, `numpy`.)

## Run

```bash
python app.py
```

Quit with **`q`**.

## Controls (keyboard)

| Key | Action |
|-----|--------|
| **M** | Cycle shape mode: line → rectangle → circle |
| **G** | Toggle globe mode (wireframe sphere; resets position/scale on enter) |
| **C** | Clear everything |
| **B** / **R** / **O** | Undo last line / rectangle / circle |
| **Backspace** | Remove last placed anchor |

## Gestures

- **Pinch** (thumb + index) in empty space → drop an anchor; pinch near an anchor → grab and move.  
- **Both hands pinching** → commit the current shape (nearest anchor per hand).  
- **Open palm** near an anchor → delete that anchor.  

**Globe mode:** one-hand pinch drag = spin; two-hand pinch = move; spread or pinch closer = zoom (unbounded—press **G** again to reset if it gets huge).

If the wrong camera opens, change the device index in `app.py` (`VideoCapture(0)`).
