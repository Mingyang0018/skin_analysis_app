# Skin Analysis Tool

A Streamlit-based application that analyzes uploaded facial photos and generates scores (1-10) for:
- Redness
- Glow
- Dryness
- Texture
- Pores
- Dark circles

## Requirements

- Python 3.8 or higher
- dlib's 81-point shape prediction model: Download `shape_predictor_81_face_landmarks.dat` from [this GitHub repository](https://github.com/codeniko/shape_predictor_81_face_landmarks) and place it in the project root directory.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/codeniko/shape_predictor_81_face_landmarks.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Note for Windows users**: Installing dlib may require additional setup, including CMake and a C++ compiler. Download precompiled `.whl` files from [this repository](https://github.com/z-mahmud22/Dlib_Windows_Python3.x) and install using `pip install <wheel_file>`.

## Usage

1. Run the application:
   ```bash
   streamlit run skin_analysis_app.py
   ```
2. Open the displayed URL in your browser (typically `http://localhost:8501`).
3. Upload a clear facial photo to view the analysis results.