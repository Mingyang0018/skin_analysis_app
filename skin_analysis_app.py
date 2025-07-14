import streamlit as st
import cv2
import dlib
import numpy as np

# 初始化 dlib 的面部检测器和 81 点形状预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat")

def get_roi_mask(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points_array = np.array([(p.x, p.y) for p in points], dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    return mask
def map_score(value, min_val, max_val, low=1, high=10):
    value = np.clip(value, min_val, max_val)
    score = (value - min_val) / (max_val - min_val) * (high - low) + low
    return round(score, 2)

def analyze_redness(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = np.prod(roi.shape[:2])
    percentage = (red_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    # 将percentage映射到1-10分
    score = map_score(percentage, 0, 5)
    return score

def analyze_shine(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 255, 255]))
    bright_pixels = cv2.countNonZero(mask)
    total_pixels = np.prod(roi.shape[:2])
    percentage = (bright_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    # 将percentage映射到1-10分
    score = map_score(percentage, 0, 20)
    # score = min(10, max(1, (percentage / 5)))
    return score

def analyze_texture(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    score = map_score(std_dev, 5, 30)
    return score

def analyze_pores(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edge_density = np.mean(np.abs(edges))
    score = map_score(edge_density, 0, 2)
    return score

def analyze_dark_circles(under_eye_roi, cheek_roi):
    under_eye_gray = cv2.cvtColor(under_eye_roi, cv2.COLOR_BGR2GRAY)
    cheek_gray = cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2GRAY)
    under_eye_avg = np.mean(under_eye_gray)
    cheek_avg = np.mean(cheek_gray)
    difference = cheek_avg - under_eye_avg
    # 将差值映射到1-10分
    score = map_score(difference, 0, 20)
    return score

def draw_mask_contour(img, mask, label=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        x, y = cnt[0][0]
        if label:
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    st.title("skin_analysis_app")
    st.write("upload a face image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "bmp", "tiff", "webp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption=None)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_image = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # st.write(f"Image dtype: {image.dtype}, Shape: {image.shape}")
        # st.write(f"Gray image dtype: {gray.dtype}, Shape: {gray.shape}")
        if gray.dtype != np.uint8:
            st.warning("Gray image is not uint8, converting...")
            gray = gray.astype(np.uint8)
        faces = detector(gray)
        if len(faces) == 0:
            st.error("No face detected. Please upload a clear photo of a face.")
            return

        face = faces[0]
        landmarks = predictor(gray, face)

        # 额头
        forehead_points = [landmarks.part(i) for i in range(68, 81)] + [landmarks.part(22), landmarks.part(21)]
        forehead_mask = get_roi_mask(image, forehead_points)
        forehead_roi = cv2.bitwise_and(image, image, mask=forehead_mask)
        draw_mask_contour(annotated_image, forehead_mask, label="Forehead")

        # 左脸颊
        lx1 = min([landmarks.part(i).x for i in range(0, 4)])
        lx2 = landmarks.part(31).x
        ly1 = max([landmarks.part(i).y for i in range(36, 42)]) + 10
        ly2 = min([landmarks.part(i).y for i in range(48, 68)]) - 10
        left_cheek_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        left_cheek_mask[ly1:ly2, lx1:lx2] = 255
        left_cheek_roi = cv2.bitwise_and(image, image, mask=left_cheek_mask)
        draw_mask_contour(annotated_image, left_cheek_mask)

        # 右脸颊
        rx1 = landmarks.part(31).x
        rx2 = max([landmarks.part(i).x for i in range(12, 17)])
        ry1 = max([landmarks.part(i).y for i in range(42, 48)]) + 10
        ry2 = min([landmarks.part(i).y for i in range(48, 68)]) - 10
        right_cheek_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        right_cheek_mask[ry1:ry2, rx1:rx2] = 255
        right_cheek_roi = cv2.bitwise_and(image, image, mask=right_cheek_mask)
        draw_mask_contour(annotated_image, right_cheek_mask)

        # 左眼下
        lex1 = min([landmarks.part(i).x for i in range(36, 42)])
        lex2 = max([landmarks.part(i).x for i in range(36, 42)])
        ley1 = max([landmarks.part(i).y for i in range(36, 42)])
        ley2 = ley1 + 20
        left_eye_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        left_eye_mask[ley1:ley2, lex1:lex2] = 255
        left_eye_roi = cv2.bitwise_and(image, image, mask=left_eye_mask)
        draw_mask_contour(annotated_image, left_eye_mask)

        # 右眼下
        rex1 = min([landmarks.part(i).x for i in range(42, 48)])
        rex2 = max([landmarks.part(i).x for i in range(42, 48)])
        rey1 = max([landmarks.part(i).y for i in range(42, 48)])
        rey2 = rey1 + 20
        right_eye_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        right_eye_mask[rey1:rey2, rex1:rex2] = 255
        right_eye_roi = cv2.bitwise_and(image, image, mask=right_eye_mask)
        draw_mask_contour(annotated_image, right_eye_mask)

        # 鼻子
        nose_points = [landmarks.part(i) for i in range(27, 36)]
        nose_mask = get_roi_mask(image, nose_points)
        nose_roi = cv2.bitwise_and(image, image, mask=nose_mask)
        draw_mask_contour(annotated_image, nose_mask)

        # 分析
        redness_score = analyze_redness(cv2.bitwise_or(left_cheek_roi, nose_roi))
        shine_score = analyze_shine(cv2.bitwise_or(forehead_roi, nose_roi))
        texture_score = analyze_texture(left_cheek_roi)
        pores_score = analyze_pores(nose_roi)
        dark_circles_score = analyze_dark_circles(left_eye_roi, left_cheek_roi)
        dryness_score = texture_score

        # 显示带标注的图像
        st.image(annotated_image, caption=None, channels="BGR", )

        # 显示分析结果
        st.subheader("Skin Analysis Results")
        st.markdown(f"""
        - **Redness**: {redness_score}
        - **Shine**: {shine_score}
        - **Dryness**: {dryness_score}
        - **Texture**: {texture_score}
        - **Pores**: {pores_score}
        - **Dark Circles**: {dark_circles_score}
        """)

if __name__ == "__main__":
    main()

