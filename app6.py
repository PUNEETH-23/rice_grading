import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

def get_classification(ratio):
    ratio = round(ratio, 1)
    return "(Full)" if ratio > 1.36 else "(Broken)"

def assess_rice_quality(full_proportion):
    if full_proportion > 65:
        return "Good Quality Rice", "🌟", "#28a745"  # Green color for good quality
    elif 50 < full_proportion <= 65:
        return "Average Quality Rice", "⭐", "#ffc107"  # Yellow color for average quality
    else:
        return "Poor Quality Rice", "⚠️", "#dc3545"  # Red color for poor quality

def detect_rice_grains(image):
    # Read image in grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binary thresholding
    ret, binary = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find unknown region
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), markers)
    
    # Create mask for each grain
    grain_masks = []
    for label in range(2, markers.max() + 1):
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 255
        grain_masks.append(mask)
    
    # Process each grain
    valid_grains = []
    grain_details = []
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
                
                valid_grains.append(cnt)
                grain_details.append({
                    'area': area,
                    'aspect_ratio': round(aspect_ratio, 2),
                    'classification': classification
                })
    
    # Visualization
    result_image = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, valid_grains, -1, (0, 255, 0), 2)
    
    return valid_grains, grain_details, result_image, binary, dist_transform, markers

def calculate_grain_statistics(grain_details):
    total_grains = len(grain_details)
    full_grains = sum(1 for grain in grain_details if grain['classification'] == "(Full)")
    broken_grains = total_grains - full_grains
    
    # Calculate proportions
    full_proportion = (full_grains / total_grains * 100) if total_grains > 0 else 0
    broken_proportion = (broken_grains / total_grains * 100) if total_grains > 0 else 0
    
    # Assess quality
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
    
    # Sidebar for configuration
    st.sidebar.header("Image Processing Settings")
    threshold_value = st.sidebar.slider("Threshold Value", 100, 255, 160)
    min_area = st.sidebar.number_input("Minimum Grain Area", 10, 200, 50)
    max_area = st.sidebar.number_input("Maximum Grain Area", 200, 10000, 5000)

    # File uploader
    uploaded_file = st.file_uploader("Choose a rice image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read the image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Detect rice grains
        valid_grains, grain_details, result_image, binary, dist_transform, markers = detect_rice_grains(image)
        
        # Calculate statistics
        stats = calculate_grain_statistics(grain_details)
        
        # Display results
        st.subheader("Analysis Results")
        
        # Quality Assessment
        st.markdown(
            f"<div style='padding: 20px; border-radius: 10px; background-color: {stats['quality_color']}; color: white; text-align: center; margin-bottom: 20px;'>"
            f"<h3 style='margin: 0;'>{stats['quality_icon']} {stats['quality']}</h3>"
            "</div>",
            unsafe_allow_html=True
        )
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Grains", stats['total'])
        with col2:
            st.metric("Full Grains", f"{stats['full']} ({stats['full_proportion']}%)")
        with col3:
            st.metric("Broken Grains", f"{stats['broken']} ({stats['broken_proportion']}%)")
        
        # Grain Details
        st.subheader("Grain Characteristics")
        details_df = [
            {
                "Area": detail['area'], 
                "Aspect Ratio": detail['aspect_ratio'], 
                "Classification": detail['classification']
            } for detail in grain_details
        ]
        st.dataframe(details_df)
        
        # Image Visualization
        st.subheader("Visualization")
        cols = st.columns(3)
        
        with cols[0]:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with cols[1]:
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                     caption="Detected Grains", use_column_width=True)
        
        with cols[2]:
            st.image(binary, caption="Binary Image", use_column_width=True)

if __name__ == "__main__":
    main()
