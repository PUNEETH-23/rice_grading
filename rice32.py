import cv2
import numpy as np
import streamlit as st

def normalize_image(img, target_range=(0, 255)):
    if target_range == (0, 1):
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm_img

def get_classification(ratio):
    return "(Full)" if ratio > 1.36 else "(Broken)"

def assess_rice_quality(full_proportion):
    if full_proportion > 65:
        return "Good Quality Rice", "🌟", "#28a745"
    elif 50 < full_proportion <= 65:
        return "Average Quality Rice", "⭐", "#ffc107"
    else:
        return "Poor Quality Rice", "⚠️", "#dc3545"

def detect_stones_by_rgb(image):
    if len(image.shape) == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return [], []  # Return empty lists if image is not color

    height, width = image.shape[:2]
    rgb_values = rgb_image.reshape((-1, 3))

    dark_threshold = 100
    dark_mask = np.all(rgb_values < dark_threshold, axis=1)
    dark_mask = dark_mask.reshape((height, width))
    dark_mask = dark_mask.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stones = []
    stone_details = []

    MIN_STONE_AREA = 10
    MAX_STONE_AREA = 10000

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_STONE_AREA < area < MAX_STONE_AREA:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            if circularity > 0.5 and solidity > 0.8:
                stones.append(cnt)
                stone_details.append({
                    'area': area,
                    'circularity': round(circularity, 2),
                    'solidity': round(solidity, 2),
                    'classification': 'Stone'
                })

    return stones, stone_details, dark_mask

def detect_rice_grains(image, threshold_value):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), markers)
    grain_masks = []
    for label in range(2, markers.max() + 1):
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 255
        grain_masks.append(mask)
    
    valid_grains = []
    grain_details = []
    full_grains = []
    broken_grains = []
    MIN_AREA, MAX_AREA = 50, 5000
    
    for mask in grain_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if w > h else float(h) / w
                classification = get_classification(aspect_ratio)
                if classification == "(Full)":
                    full_grains.append(cnt)
                else:
                    broken_grains.append(cnt)
                valid_grains.append(cnt)
                grain_details.append({
                    'area': area,
                    'aspect_ratio': round(aspect_ratio, 2),
                    'classification': classification
                })
    
    stones, stone_details, _ = detect_stones_by_rgb(image)
    
    result_image = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, full_grains, -1, (0, 255, 0), 2)  # Green for full grains
    cv2.drawContours(result_image, broken_grains, -1, (0, 255, 255), 2)  # Yellow for broken grains
    cv2.drawContours(result_image, stones, -1, (0, 0, 255), 2)  # Red for stones
    
    return valid_grains, grain_details, stones, stone_details, result_image, binary, dist_transform, markers

def calculate_grain_statistics(grain_details):
    total_grains = len(grain_details)
    full_grains = sum(1 for grain in grain_details if grain['classification'] == "(Full)")
    broken_grains = total_grains - full_grains

    full_proportion = (full_grains / total_grains * 100) if total_grains > 0 else 0
    broken_proportion = (broken_grains / total_grains * 100) if total_grains > 0 else 0

    quality_assessment, quality_icon, quality_color = assess_rice_quality(full_proportion)
    
    return {
        'total': total_grains,
        'full': full_grains,
        'broken': broken_grains,
        'full_proportion': round(full_proportion, 1),
        'broken_proportion': round(broken_proportion, 1),
        'quality': quality_assessment,
        'quality_icon': quality_icon,
        'quality_color': quality_color
    }

def main():
    st.title("🌾 Rice Grain Analyzer")
    
    st.sidebar.header("Image Processing Settings")
    threshold_value = st.sidebar.slider("Threshold Value", 100, 255, 160)
    min_area = st.sidebar.number_input("Minimum Grain Area", 10, 200, 50)
    max_area = st.sidebar.number_input("Maximum Grain Area", 200, 10000, 5000)
    
    uploaded_file = st.file_uploader("Choose a rice image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        valid_grains, grain_details, stones, stone_details, result_image, binary, dist_transform, markers = detect_rice_grains(image, threshold_value)
        
        stats = calculate_grain_statistics(grain_details)
        
        # Display the counts together for full grains, broken grains, and stones
        st.write(f"**Total Grains:** {stats['total']}")
        st.write(f"**Full Grains:** {stats['full']} ({stats['full_proportion']}%)")
        st.write(f"**Broken Grains:** {stats['broken']} ({stats['broken_proportion']}%)")
        st.write(f"**Detected Stones:** {len(stones)}")
        
        st.image(result_image, caption="Detected Grains and Stones", use_column_width=True)
        st.image(binary, caption="Binary Thresholded Image", use_column_width=True)
        
        if stones:
            stone_image = np.copy(image)
            for stone in stones:
                if len(stone) == 4:  # Ensure we have (x, y, w, h)
                    x, y, w, h = stone
                    cv2.rectangle(stone_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            st.image(stone_image, caption="Detected Stones Highlighted", use_column_width=True)
        else:
            st.write("No stones detected.")
        
        normalized_dist_transform = normalize_image(dist_transform, target_range=(0, 255))
        st.image(normalized_dist_transform, caption="Distance Transform", use_column_width=True, clamp=False)
        
        normalized_markers = normalize_image(markers, target_range=(0, 255))
        st.image(normalized_markers, caption="Watershed Markers", use_column_width=True, clamp=False)

if __name__ == "__main__":
    main()


