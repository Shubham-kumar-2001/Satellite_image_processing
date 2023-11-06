import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import streamlit as st
from skimage import io
from skimage.filters import threshold_multiotsu
import numpy as np
from io import BytesIO

def convert_image_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def download_image(image, file_name):
    # Save the image to a BytesIO object
    image_buffer = BytesIO()
    plt.imsave(image_buffer, image, format="png")
    
    # Create a download link for the image
    st.download_button(
        label="Download Image",
        data=image_buffer.getvalue(),
        file_name=file_name,
        mime="image/png"
    )

def detect_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_keypoints = cv2.drawKeypoints(img, keypoints, None)
    
    # Apply Corner Harris detection
    gray = np.float32(gray)
    harris_response = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_response = cv2.dilate(harris_response, None)
    img_harris = np.copy(img)
    img_harris[harris_response > 0.01 * harris_response.max()] = [0, 0, 255]
    
    # Convert the images to RGB format for display in Streamlit
    img_keypoints_rgb = cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB)
    img_harris_rgb = cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB)
    
    # Return the images
    return img_keypoints_rgb, img_harris_rgb    


def split_channels(image):
    # Split image into red, green, blue channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return red_channel, green_channel, blue_channel, grayscale_image

def apply_gabor_filter(channel, lambda_val=10, theta=0, sigma=5, psi=0, gamma=0.5):
    gabor_kernel = cv2.getGaborKernel((50, 50), sigma, theta, lambda_val, gamma, psi, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(channel, cv2.CV_8UC3, gabor_kernel)
    return filtered_image

def apply_sobel_filter(image, channels=None):
    filtered_image = np.copy(image)
    if channels is None:
        channels = [0, 1, 2]  # Apply on all channels by default
    for channel in channels:
        filtered_image[:, :, channel] = cv2.Sobel(image[:, :, channel], cv2.CV_64F, 1, 1)
    return filtered_image


def apply_gaussian_filter(image, kernel_size, sigma, channels=None):
    filtered_image = np.copy(image)
    if channels is None:
        channels = [0, 1, 2]  # Apply on all channels by default
    for channel in channels:
        filtered_image[:, :, channel] = cv2.GaussianBlur(filtered_image[:, :, channel], (kernel_size, kernel_size), sigma)
    return filtered_image


def apply_canny_edge_detection(image, threshold1, threshold2):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    return edges

def generate_segmentation_colors(num_colors):
    colors = []
    np.random.seed(0)
    for _ in range(num_colors):
        color = np.random.randint(0, 256, size=3)
        colors.append(color)
    return colors
def threshold_region(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresholds = threshold_multiotsu(gray_image, classes=4)
    regions = np.digitize(gray_image, bins=thresholds)
    segm1 = (regions == 0)
    segm2 = (regions == 1)
    segm3 = (regions == 2)
    segm4 = (regions == 3)
    return regions
def threshold_segmentation(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresholds = threshold_multiotsu(gray_image, classes=4)
    regions = np.digitize(gray_image, bins=thresholds)
    segm1 = (regions == 0)
    segm2 = (regions == 1)
    segm3 = (regions == 2)
    segm4 = (regions == 3)
    segm5 = (regions == 4)
    return segm4

def map_labels_to_colors(labels, colors):
    height, width = labels.shape
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            label = labels[i, j]
            color = colors[label]
            segmented_image[i, j] = color
    return segmented_image



def kmeans_segmentation(image, n_clusters):
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)

    # Get the labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Assign each pixel to its corresponding centroid
    segmented_image = centroids[labels].reshape(image.shape)

    return segmented_image

def gmm_segmentation(image, n_components):
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Normalize pixel values to [0, 1]
    X = pixels.astype(float) / 255.0

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=40, init_params='random')
    gmm.fit(X)

    labels = gmm.predict(X)

    # Reshape the labels back to the original image shape
    labels = labels.reshape(image.shape[:2])

    return labels

def adjust_hsv(image, hue_shift, saturation_scale, value_scale):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply adjustments to the HSV channels
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180  # Shift hue
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_scale  # Scale saturation
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * value_scale  # Scale value

    # Convert the image back to BGR color space
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return adjusted_image
    
def channel_wise_operation(image, theta1, theta2, filter_type, kernel_size, sigma, threshold1, threshold2):
    if len(image.shape) == 3:
        # Split the image into channels
        b, g, r = cv2.split(image)
        channels = [r, g, b]
    else:
        # Grayscale image
        channels = [image]

    # Display the channel-wise operation results
    st.subheader("Channel-wise Operation")
    for i, channel in enumerate(channels):
        st.image(channel, caption=f"Channel {i+1}", use_column_width=True)

    if filter_type == "Gaussian":
        filtered_channels = []
        for channel in channels:
            if len(channel.shape) == 2:
                filtered_channel = cv2.GaussianBlur(channel, (kernel_size, kernel_size), sigma)
            else:
                filtered_channel = apply_gaussian_filter(channel, kernel_size, sigma)
            filtered_channels.append(filtered_channel)
        filtered_image = cv2.merge(filtered_channels)

        # Display the filtered image
        st.image(filtered_image, caption="Filtered Image", use_column_width=True)

    elif filter_type == "Canny":
        edges = apply_canny_edge_detection(image, threshold1, threshold2)
        filtered_image = cv2.bitwise_and(image, edges)

        # Display the filtered image
        st.image(filtered_image, caption="Filtered Image", use_column_width=True)

    elif filter_type == "Gabor":
        filtered_channels = []
        for channel in channels:
            if len(channel.shape) == 2:
                filtered_channel = apply_gabor_filter(channel, lambda_val=10, theta=theta1, sigma=sigma)
            else:
                filtered_channel = apply_gabor_filter(channel, lambda_val=10, theta=theta2, sigma=sigma)
            filtered_channels.append(filtered_channel)
        filtered_image = cv2.merge(filtered_channels)

        # Display the filtered image
        st.image(filtered_image, caption="Filtered Image", use_column_width=True)

    elif filter_type == "Sobel":
        filtered_image = apply_sobel_filter(image)

        # Display the filtered image
        st.image(filtered_image, caption="Filtered Image", use_column_width=True)


def perform_channel_wise_operation(image):
    st.subheader("Channels")
    b, g, r = cv2.split(image)
    st.image(r, caption="Red Channel", use_column_width=True)
    st.image(g, caption="Green Channel", use_column_width=True)
    st.image(b, caption="Blue Channel", use_column_width=True)
    

    theta1 = st.slider("Theta 1", 0, 360, int(1*np.pi/4), key="theta1_slider")
    theta2 = st.slider("Theta 2", 0, 360, int(3*np.pi/4), key="theta2_slider")
    filter_type = st.selectbox("Select Filter", ["Gaussian", "Canny", "Gabor", "Sobel"])

    if filter_type == "Gaussian":
        kernel_size = st.slider("Kernel Size", 1, 15, 3)
        sigma = st.slider("Sigma", 0.1, 5.0, 1.0)
        gaussian_channel = st.selectbox("Select Channel for Gaussian Filtering", ["Red", "Green", "Blue"])

        channel_wise_operation(image, theta1, theta2, filter_type, kernel_size, sigma, None, None)

    elif filter_type == "Canny":
        threshold1 = st.slider("Threshold 1", 0, 255, 100)
        threshold2 = st.slider("Threshold 2", 0, 255, 200)
        edges = apply_canny_edge_detection(image, threshold1, threshold2)
        st.image(edges, caption="Canny Edge Image", use_column_width=True)
        download_image(edges, "canny.png")  # Download edge image

    elif filter_type == "Gabor":
        red_channel, green_channel, blue_channel, grayscale_image = split_channels(image)

        channel_options = ["Red", "Green", "Blue", "Grayscale"]
        selected_channel = st.selectbox("Select a channel for Gabor filtering:", channel_options)

        if selected_channel == "Red":
            channel = red_channel
        elif selected_channel == "Green":
            channel = green_channel
        elif selected_channel == "Blue":
            channel = blue_channel
        else:
            channel = grayscale_image

        st.sidebar.subheader("Gabor Filter Parameters")
        lambda_val = st.sidebar.slider("Lambda", 0.1, 50.0, 10.0)
        theta = st.sidebar.slider("Theta", 0, 180, 0)
        sigma = st.sidebar.slider("Sigma", 0.1, 10.0, 5.0)
        psi = st.sidebar.slider("Psi", 0, 180, 0)
        gamma = st.sidebar.slider("Gamma", 0.1, 1.0, 0.5)
        filtered_image = apply_gabor_filter(channel, lambda_val=lambda_val, theta=theta, sigma=sigma, psi=psi, gamma=gamma)
        st.image(filtered_image, caption="Gabor Filtered Image", use_column_width=True)
        download_image(filtered_image, "gabor_filtered.png")  # Download filtered image
         

    elif filter_type == "Sobel":
        sobel_channels = st.multiselect("Select Channels for Sobel Filtering", ["Red", "Green", "Blue"])

        channels_dict = {"Red": 0, "Green": 1, "Blue": 2}
        channel_indices = [channels_dict[channel] for channel in sobel_channels]

        filtered_image = apply_sobel_filter(image, channels=channel_indices)
        download_image(filtered_image, "sobel_filtered.png")  # Download filtered image


        channel_wise_operation(image, theta1, theta2, filter_type, None, None, None, filtered_image)





def perform_image_operations(image):
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_column_width=True)
    if image.shape[2] != 3:
        # Convert the image to RGB if it has a different number of channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to a common size if necessary
    if image.shape[0] > 800 or image.shape[1] > 800:
        image = cv2.resize(image, (800, 800))

    st.title("Image Operations")
    operation_selection = st.selectbox("Select Operation", ["K-Means Segmentation", "GMM Segmentation", "HSV Conversion", "Threshold Region", "Threshold Segmentation", " SIFT Detection"])

    if operation_selection == "K-Means Segmentation":
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="kmeans_slider")
        segmented_image = kmeans_segmentation(image, n_clusters)

        # Normalize the pixel values to [0, 255]
        segmented_image = (segmented_image * 255).astype(np.uint8)

        st.subheader("K-Means Segmented Image")
        st.image(segmented_image, caption="K-Means Segmented Image", use_column_width=True)
        download_image(segmented_image, "kmeans_segmented.png")  # Download segmented image

    elif operation_selection == "GMM Segmentation":
        n_components = st.slider("Number of Components", 2, 10, 3, key="gmm_slider")
        segmented_labels = gmm_segmentation(image, n_components)
        colors = generate_segmentation_colors(n_components)  # Assuming the implementation of this function exists
        segmented_image = map_labels_to_colors(segmented_labels, colors)

        st.subheader("GMM Segmented Image")
        st.image(segmented_image, caption="GMM Segmented Image", use_column_width=True)

        # Download segmented image
        download_image(segmented_image, "gmm_segmented.png")

    elif operation_selection == "Threshold Region":
        # Perform threshold segmentation
        segmented_labels = threshold_region(image)
        colors = generate_segmentation_colors(4)  # Assuming 4 classes from the threshold
        segmented_image = map_labels_to_colors(segmented_labels, colors)
        st.subheader("Threshold Segmented Image")
        st.image(segmented_image, caption="Threshold Segmented Image", use_column_width=True)
        download_image(segmented_image, "Threshold_region.png")

    elif operation_selection == "Threshold Segmentation":
        # Perform threshold segmentation
        segmented_labels = threshold_segmentation(image)
        colors = generate_segmentation_colors(4)  # Assuming 4 classes from the threshold
        segmented_image = map_labels_to_colors(segmented_labels, colors)
        st.subheader("Threshold Segmented Image")
        st.image(segmented_image, caption="Threshold Segmented Image", use_column_width=True)
        download_image(segmented_image, "Threshold_segmentation.png")

    elif operation_selection == "HSV Conversion":
        hue_shift = st.slider("Hue Shift", -180, 180, 0)
        saturation_scale = st.slider("Saturation Scale", 0.0, 5.0, 1.0)
        value_scale = st.slider("Value Scale", 0.0, 5.0, 1.0)
        adjusted_image = adjust_hsv(image, hue_shift, saturation_scale, value_scale)

        st.subheader("Adjusted HSV Image")
        st.image(adjusted_image, caption="Adjusted HSV Image", use_column_width=True)
        download_image(adjusted_image, "HSV_image.png")  # Download adjusted image

        # Display the original image and adjusted image side by side
        st.subheader("Comparison")
        col1, col2 = st.columns(2)
        col1.subheader("Original Image")
        col1.image(image, use_column_width=True)
        col2.subheader("Adjusted Image")
        col2.image(adjusted_image, use_column_width=True)

    elif operation_selection == " SIFT Detection":
        keypoints_image, harris_image = detect_keypoints(image)
        st.subheader("Image with Keypoints")
        st.image(keypoints_image, caption="Image with Keypoints", use_column_width=True)
        download_image(keypoints_image, "image_with_keypoints.png")  # Download image with keypoints

        st.subheader("Harris Image")
        st.image(harris_image, caption="Harris  Corner Image", use_column_width=True)
        download_image(harris_image, "harris_image.png")  # Download Harris image

    
def home_page():
    st.title("Welcome to the App")
    st.write("This is the home page")

def display_uploaded_image(image):
    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)


def browse_image_page():
    st.title("UPLOAD IMAGES")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Upload an image file
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = io.imread(uploaded_file)

        # Return the image
        return image


def main():
    menu = ["Home", "Upload Image", "Image Operation", "Channel-wise Operation"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Display selected page based on user choice
    if choice == "Home":
        home_page()
    elif choice == "Upload Image":
        browse_image_page()
    elif choice == "Image Operation":
        st.title("IMAGE OPERATIONS")
        st.write("Select an image from the 'Upload Image' page to perform operations.")
        image = browse_image_page()  # Get the uploaded image
        if image is not None:
            perform_image_operations(image) 
    
    elif choice == "Channel-wise Operation":
        st.title("CHANNEL-WISE OPERATION")
        st.write("Select an image from the 'Upload Image' page to perform channel-wise operation.")
        image = browse_image_page()  # Get the uploaded image
        if image is not None:
            perform_channel_wise_operation(image)  # Perform channel-wise operation


if __name__ == "__main__":
    main()