import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.signal import convolve2d
from sklearn.decomposition import PCA as SkPCA
import time
import io
import base64

st.set_page_config(page_title="Interactive NumPy Guide", layout="wide")
# --- Custom CSS for styling ---
def apply_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #0e1117;
                color: #e6e6e6;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .css-1d391kg {
                background-color: #1a1c23 !important;
                color: white;
            }
            button[kind="primary"] {
                background-color: #61dafb !important;
                color: #0e1117 !important;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #61dafb !important;
            }
            textarea, input[type=text] {
                background-color: #2a2c33 !important;
                color: white !important;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #007bff !important;
            }
            textarea, input[type=text] {
                background-color: #f8f9fa !important;
                color: black !important;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Sidebar for navigation and settings
st.sidebar.title("Navigate Sections")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "1️⃣ Matrix Multiplication",
        "2️⃣ Broadcasting",
        "3️⃣ Singular Value Decomposition (SVD)",
        "4️⃣ Fast Fourier Transform (FFT)",
        "5️⃣ Principal Component Analysis (PCA)",
        "6️⃣ Linear Algebra",
        "7️⃣ Random Module",
        "8️⃣ Multidimensional Arrays",
        "9️⃣ Library Integrations",
        "📖 Help & Cheat Sheet",
    ],
)
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
apply_theme(theme)

language = st.sidebar.selectbox("Language", ["English", "Hindi"])
def translate(text, lang):
    if lang == "Hindi":
        translations = {
            "Welcome to the Interactive NumPy Guide 🚀": "इंटरएक्टिव NumPy गाइड में आपका स्वागत है 🚀",
            "Matrix Multiplication": "मैट्रिक्स गुणन",
            "Broadcasting in NumPy": "NumPy में ब्रॉडकास्टिंग",
            "Singular Value Decomposition (SVD)": "सिंगुलर वैल्यू डीकंपोजिशन (SVD)",
            "Fast Fourier Transform (FFT)": "फास्ट फूरियर ट्रांसफॉर्म (FFT)",
            "Principal Component Analysis (PCA)": "प्रिंसिपल कंपोनेंट एनालिसिस (PCA)",
            "Linear Algebra with NumPy": "NumPy के साथ लीनियर अलजेब्रा",
            "NumPy Random Module": "NumPy रैंडम मॉड्यूल",
            "Multidimensional Arrays": "मल्टीडायमेंशनल ऐरेज़",
            "Library Integrations": "लाइब्रेरी इंटीग्रेशन्स",
            "Help & Cheat Sheet": "सहायता और चीट शीट",
            "Try Your Own Code": "अपना कोड आज़माएं",
            "Run": "चलाएं",
            "Quiz": "क्विज़",
        }
        return translations.get(text, text)
    return text

# Helper for safe code execution
def run_user_code(code, global_vars=None):
    try:
        local_vars = {}
        exec(code, global_vars if global_vars else {}, local_vars)
        return local_vars, None
    except Exception as e:
        return None, str(e)

# Helper for Jupyter download
def download_jupyter(code_content, filename="section.ipynb"):
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# NumPy Section Tutorial"]},
            {"cell_type": "code", "metadata": {}, "source": [code_content], "outputs": [], "execution_count": None}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    nb_str = str(nb).replace("'", '"')
    b64 = base64.b64encode(nb_str.encode()).decode()
    href = f'<a download="{filename}" href="data:file/nbformat;base64,{b64}">Download Jupyter Notebook</a>'
    st.markdown(href, unsafe_allow_html=True)

# Cache expensive functions
@st.cache_data
def compute_svd(A):
    return np.linalg.svd(A, full_matrices=False)

@st.cache_data
def generate_random_data(dist, params, n_samples):
    if dist == "Uniform":
        return np.random.uniform(*params, n_samples)
    elif dist == "Normal":
        return np.random.normal(*params, n_samples)
    elif dist == "Poisson":
        return np.random.poisson(params[0], n_samples)
    elif dist == "Binomial":
        return np.random.binomial(*params, n_samples)

# --- Home ---
if page == "🏠 Home":
    st.title(translate("Welcome to the Interactive NumPy Guide 🚀", language))
    st.markdown(
        """
        This app provides interactive tutorials on NumPy, from beginner to advanced.
        **Features**:
        - Interactive sliders, editable code, Plotly visuals
        - Quizzes, file uploads, light/dark themes, Jupyter exports
        - Sections: Matrix ops, broadcasting, SVD, FFT, PCA, linear algebra, random, arrays, integrations
        - New: Help section with NumPy cheat sheet
        Navigate using the sidebar.
        """
    )
    st.image("https://numpy.org/images/logo.svg", width=200, caption="NumPy Logo")

# --- 1. Matrix Multiplication ---
# --- 1. Matrix Multiplication ---
elif page == "1️⃣ Matrix Multiplication":
    st.title(translate("Matrix Multiplication", language))
    st.markdown(
        """
        **Mathematics**: For A (m×n) and B (n×p), C[i,j] = Σ_k A[i,k] * B[k,j]. Derives from linear transformations.
        **NumPy**: `np.dot(A, B)` or `A @ B`. Uses BLAS for speed.
        **Use Cases**: Neural networks, graphics transformations.
        **Pitfall**: Ensure inner dimensions match to avoid ValueError.
        """
    )

    tab1, tab2 = st.tabs(["Main Demo", "Convolution Application"])
    with tab1:
        animate_size = st.checkbox("Animate matrix size (cycles 2 to 5)", value=False)
        if 'current_size' not in st.session_state:
            st.session_state.current_size = 2

        if animate_size:
            size = st.session_state.current_size
            st.write(f"Animating matrix size: {size} x {size}")
            time.sleep(0.7)
            st.session_state.current_size = st.session_state.current_size % 5 + 1 if st.session_state.current_size < 5 else 2
            st.rerun()
        else:
            size = st.slider("Matrix size (NxN)", 2, 5, 3)

        A = np.random.randint(1, 10, (size, size))
        B = np.random.randint(1, 10, (size, size))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Matrix A")
            st.write(A)
        with col2:
            st.subheader("Matrix B")
            st.write(B)
        with col3:
            result_matrix = np.dot(A, B)
            st.subheader("Full Result Matrix")
            st.write(result_matrix)

        st.subheader("Step-by-Step Multiplication Visualization")
        animate_steps = st.checkbox("Animate steps (cycles automatically)", value=False)
        max_steps = size * size
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1

        if animate_steps:
            steps_to_show = st.session_state.current_step
            time.sleep(0.5)
            st.session_state.current_step = (steps_to_show % max_steps) + 1
            st.rerun()
        else:
            steps_to_show = st.slider("Number of steps to show", 1, max_steps, size)

        def multiply_stepwise(A, B, steps):
            result = np.zeros((size, size), dtype=int)
            step_outputs = []
            count = 0
            for i in range(size):
                for j in range(size):
                    if count >= steps:
                        return result, step_outputs
                    val = np.sum(A[i, :] * B[:, j])
                    result[i, j] = val
                    step_outputs.append((i, j, val))
                    count += 1
            return result, step_outputs

        partial_result, step_values = multiply_stepwise(A, B, steps_to_show)
        st.write("Partial Result:")
        st.write(partial_result)

        # Visual with highlight
        fig_partial = px.imshow(partial_result, text_auto=True, color_continuous_scale='Viridis', title="Partial Result Heatmap")
        if step_values:
            last_i, last_j, last_val = step_values[-1]
            fig_partial.add_annotation(x=last_j, y=last_i, text=f"Step {len(step_values)}: {last_val}", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_partial, use_container_width=True)

        st.subheader("Step Details")
        for idx, (i, j, val) in enumerate(step_values):
            st.write(f"Step {idx+1}: Result[{i},{j}] = {val} (Σ A[{i},k] * B[k,{j}])")

        st.subheader(translate("Try Your Own Code", language))
        code = f"""# Matrix multiplication example
import numpy as np

A = np.array({A.tolist()})  # First matrix
B = np.array({B.tolist()})  # Second matrix
result = np.dot(A, B)  # Compute dot product; alternative: A @ B
print("Result:")
print(result)
"""
        edited_code = st.text_area("Edit and run this code:", code, height=200)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                if 'result' in local_vars:
                    st.write(local_vars['result'])
                else:
                    st.write("Code ran but no 'result' variable found.")

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("What is the shape of A (2x3) @ B (3x4)?", ["2x4", "3x3", "Error", "4x2"])
        if quiz_q == "2x4":
            st.success("Correct! Outer dimensions determine the result shape.")
        elif quiz_q:
            st.error("Incorrect. Hint: Inner dimensions must match, outer give result.")

        download_jupyter(edited_code, "matrix_multiplication.ipynb")

    with tab2:
        st.subheader("Application: Convolutions for Image Processing")
        st.markdown("**Math**: Slide kernel over image, compute dot product at each position. Used in CNNs.")
        image_size = st.slider("Image size (NxN)", 5, 20, 10)
        image = np.random.rand(image_size, image_size)
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # Edge detection
        result_conv = convolve2d(image, kernel, mode='same')
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image Array:")
            st.write(image)
        with col2:
            st.write("Convolved Result:")
            st.write(result_conv)
        fig_conv = px.imshow(result_conv, color_continuous_scale='gray', title="Edge-Detected Image")
        st.plotly_chart(fig_conv, use_container_width=True)

# --- 2. Broadcasting ---
elif page == "2️⃣ Broadcasting":
    st.title(translate("Broadcasting in NumPy", language))
    st.markdown(
        """
        **Mathematics**: Expands arrays with compatible shapes for element-wise operations. Rules: Align right-to-left; dims equal or 1.
        **NumPy**: Implicit in `+`, `-`, `*`, `/`. Uses C for speed.
        **Use Cases**: Normalizing data, adding biases in ML.
        **Pitfall**: Shape mismatch causes ValueError; use `np.newaxis`.
        """
    )

    tab1, tab2 = st.tabs(["Main Demo", "Visual Explanation"])
    with tab1:
        shape_A_rows = st.slider("Rows in Array A", 1, 5, 3)
        shape_A_cols = st.slider("Columns in Array A", 1, 5, 1)
        shape_B_size = st.slider("Size of Array B", 1, 5, 3)

        A = np.arange(shape_A_rows * shape_A_cols).reshape(shape_A_rows, shape_A_cols)
        B = np.arange(shape_B_size)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Array A")
            st.write(A)
        with col2:
            st.subheader("Array B")
            st.write(B)
        with col3:
            st.subheader("Full Result (A + B)")
            try:
                full_result = A + B
                st.write(full_result)
            except Exception as e:
                st.error(f"Broadcasting error: {e}")
                full_result = None

        st.subheader("Step-by-Step Addition Visualization")
        animate_steps = st.checkbox("Animate steps (cycles automatically)", value=False)
        max_steps = A.size
        if 'current_broadcast_step' not in st.session_state:
            st.session_state.current_broadcast_step = 1

        if animate_steps:
            steps_to_show = st.session_state.current_broadcast_step
            time.sleep(0.5)
            st.session_state.current_broadcast_step = (steps_to_show % max_steps) + 1
            st.rerun()
        else:
            steps_to_show = st.slider("Number of steps to show", 1, max_steps, min(max_steps, 5))

        if full_result is not None:
            partial_result = np.zeros_like(full_result)
            count = 0
            for i in range(shape_A_rows):
                for j in range(shape_A_cols):
                    if count >= steps_to_show:
                        break
                    # Simulate broadcasting: B expanded to match A
                    expanded_b = B[j % shape_B_size] if shape_B_size > 1 else B[0]
                    partial_result[i, j] = A[i, j] + expanded_b
                    count += 1

            st.write("Partial Result:")
            st.write(partial_result)

            # Visual with highlight
            fig_partial = px.imshow(partial_result, text_auto=True, color_continuous_scale='Greens', title="Partial Broadcasted Addition")
            if count > 0:
                last_i = (count - 1) // shape_A_cols
                last_j = (count - 1) % shape_A_cols
                fig_partial.add_annotation(x=last_j, y=last_i, text=f"Step {count}", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_partial, use_container_width=True)

        st.subheader(translate("Try Your Own Code", language))
        code = f"""# Broadcasting example
import numpy as np

A = np.arange({shape_A_rows * shape_A_cols}).reshape({shape_A_rows}, {shape_A_cols})
B = np.arange({shape_B_size})

result = A + B  # B expands to match A's shape
print(result)
"""
        edited_code = st.text_area("Edit and run this code:", code, height=200)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                if 'result' in local_vars:
                    st.write(local_vars['result'])
                else:
                    st.write("Code ran but no 'result' variable found.")

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("Does np.array([1,2,3]) + 5 work?", ["Yes, scalar broadcasting", "No, shape mismatch", "Only if reshaped"])
        if quiz_q == "Yes, scalar broadcasting":
            st.success("Correct! Scalars broadcast to any shape.")
        elif quiz_q:
            st.error("Incorrect. Hint: NumPy treats scalars as expandable.")

        download_jupyter(edited_code, "broadcasting.ipynb")

    with tab2:
        st.subheader("Visual Demo of Broadcasting")
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=A, colorscale='Blues', name='A', text=[[str(val) for val in row] for row in A], showscale=False))
        fig.add_trace(go.Heatmap(z=B.reshape(1, -1), colorscale='Reds', name='B (Expanded)', text=[str(val) for val in B], showscale=False))
        if full_result is not None:
            fig.add_trace(go.Heatmap(z=full_result, colorscale='Greens', name='Result', text=[[str(val) for val in row] for row in full_result], showscale=False))
        fig.update_layout(title="Broadcasting Visualization")
        st.plotly_chart(fig, use_container_width=True)


# --- 3. Singular Value Decomposition (SVD) ---
elif page == "3️⃣ Singular Value Decomposition (SVD)":
    st.title(translate("Singular Value Decomposition (SVD)", language))
    st.markdown(
        """
        **Mathematics**: A = U Σ V^T, where U/V orthogonal, Σ diagonal (singular values). Derives from A^T A eigenvalues.
        **NumPy**: `np.linalg.svd(A, full_matrices=False)` for compact form.
        **Use Cases**: Image compression, PCA, recommendation systems.
        **Pitfall**: Large matrices are computationally expensive.
        """
    )

    tab1, tab2 = st.tabs(["Main Demo", "Image Compression"])
    with tab1:
        rows = st.slider("Rows", 2, 6, 4)
        cols = st.slider("Columns", 2, 6, 4)

        A = np.random.randn(rows, cols)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matrix A")
            st.write(A)
        with col2:
            U, S, VT = compute_svd(A)
            st.subheader("Singular Values")
            st.write(S)

        st.subheader("Left Singular Vectors (U)")
        st.write(U)
        st.subheader("Right Singular Vectors (VT)")
        st.write(VT)

        st.subheader(translate("Try Your Own Code", language))
        code = f"""# SVD example
import numpy as np

A = np.array({A.tolist()})

U, S, VT = np.linalg.svd(A, full_matrices=False)  # Compact SVD
print("Singular Values:", S)
"""
        edited_code = st.text_area("Edit and run this code:", code, height=200)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                if 'S' in local_vars:
                    st.write("Singular Values:", local_vars['S'])
                else:
                    st.write("Code ran but no 'S' variable found.")

        st.subheader("Singular Values Plot")
        fig = px.bar(x=list(range(1, len(S) + 1)), y=S, labels={'x': 'Index', 'y': 'Singular Value'}, title="Singular Values")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("What does Σ represent in SVD?", ["Singular values (importance)", "Orthogonal matrices", "Original matrix"])
        if quiz_q == "Singular values (importance)":
            st.success("Correct! They indicate dimension significance.")
        elif quiz_q:
            st.error("Incorrect. Hint: Diagonal matrix in decomposition.")

        download_jupyter(edited_code, "svd.ipynb")

    with tab2:
        st.subheader("Application: Image Compression via SVD")
        img_size = st.slider("Image size (NxN)", 10, 50, 20)
        k = st.slider("Top k singular values", 1, img_size, 5)
        img = np.random.rand(img_size, img_size)
        U, S, VT = compute_svd(img)
        compressed = np.dot(U[:, :k] * S[:k], VT[:k, :])
        col1, col2 = st.columns(2)
        with col1:
            fig_orig = px.imshow(img, color_continuous_scale='gray', title="Original Image")
            st.plotly_chart(fig_orig)
        with col2:
            fig_comp = px.imshow(compressed, color_continuous_scale='gray', title=f"Compressed (k={k})")
            st.plotly_chart(fig_comp)

# --- 4. Fast Fourier Transform (FFT) ---
elif page == "4️⃣ Fast Fourier Transform (FFT)":
    st.title(translate("Fast Fourier Transform (FFT)", language))
    st.markdown(
        """
        **Mathematics**: X_k = Σ_n x_n * e^{-2πi k n / N}, O(n log n) algorithm for frequency domain.
        **NumPy**: `np.fft.fft(signal)`.
        **Use Cases**: Audio processing, image filtering.
        **Pitfall**: Assumes periodicity; use windowing for real signals.
        """
    )

    tab1, tab2 = st.tabs(["Main Demo", "Signal Filtering"])
    with tab1:
        length = st.slider("Signal length", 64, 512, 256, step=64)
        freq1 = st.slider("Frequency 1", 1, 30, 5)
        freq2 = st.slider("Frequency 2", 1, 30, 12)

        t = np.linspace(0, 1, length)
        signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Signal (Time Domain)")
            fig_signal = px.line(x=t, y=signal, labels={'x': 'Time', 'y': 'Amplitude'}, title="Original Signal")
            st.plotly_chart(fig_signal)
        with col2:
            fft_vals = np.fft.fft(signal)
            freq = np.fft.fftfreq(length, d=1/length)
            magnitude = np.abs(fft_vals)
            st.subheader("FFT Magnitude Spectrum")
            fig_fft = px.bar(x=freq[:length//2], y=magnitude[:length//2], labels={'x': 'Frequency', 'y': 'Magnitude'}, title="FFT Spectrum")
            st.plotly_chart(fig_fft)

        st.subheader(translate("Try Your Own Code", language))
        code = f"""# FFT example
import numpy as np

length = {length}
t = np.linspace(0, 1, length)
signal = np.sin(2 * np.pi * {freq1} * t) + 0.5 * np.sin(2 * np.pi * {freq2} * t)

fft_vals = np.fft.fft(signal)
freq = np.fft.fftfreq(length, d=1/length)
magnitude = np.abs(fft_vals[:length//2])
print(magnitude)
"""
        edited_code = st.text_area("Edit and run this code:", code, height=220)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                if 'magnitude' in local_vars:
                    st.write(local_vars['magnitude'])
                else:
                    st.write("Code ran but no 'magnitude' variable found.")

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("What does FFT convert?", ["Time to frequency domain", "Frequency to time", "Spatial to temporal"])
        if quiz_q == "Time to frequency domain":
            st.success("Correct! Decomposes signals into frequencies.")
        elif quiz_q:
            st.error("Incorrect. Hint: Used in spectrum analysis.")

        download_jupyter(edited_code, "fft.ipynb")

    with tab2:
        st.subheader("Application: Signal Filtering")
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.2)
        noisy_signal = signal + noise_level * np.random.randn(length)
        fft_noisy = np.fft.fft(noisy_signal)
        fft_filtered = fft_noisy.copy()
        fft_filtered[int(length*0.2):] = 0  # Low-pass filter
        filtered_signal = np.fft.ifft(fft_filtered).real
        col1, col2 = st.columns(2)
        with col1:
            fig_noisy = px.line(x=t, y=noisy_signal, title="Noisy Signal")
            st.plotly_chart(fig_noisy)
        with col2:
            fig_filtered = px.line(x=t, y=filtered_signal, title="Filtered Signal")
            st.plotly_chart(fig_filtered)

# --- 5. Principal Component Analysis (PCA) ---
elif page == "5️⃣ Principal Component Analysis (PCA)":
    st.title(translate("Principal Component Analysis (PCA)", language))
    st.markdown(
        """
        **Mathematics**: Eigen decomposition of covariance: Cov = (1/n) X^T X, project onto top eigenvectors.
        **NumPy**: `np.cov`, `np.linalg.eigh`.
        **Use Cases**: Dimensionality reduction, data visualization.
        **Pitfall**: Center data; assumes linear relationships.
        """
    )

    tab1, tab2 = st.tabs(["Main Demo", "With Uploaded Data"])
    with tab1:
        n_points = st.slider("Number of 3D data points", 10, 200, 50)
        mean = [0, 0, 0]
        cov = [[3, 1, 0], [1, 2, 0.5], [0, 0.5, 1]]
        data = np.random.multivariate_normal(mean, cov, n_points)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Generated Data (first 5 rows)")
            st.write(data[:5])
        with col2:
            data_centered = data - np.mean(data, axis=0)
            cov_matrix = np.cov(data_centered, rowvar=False)
            eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
            order = eig_vals.argsort()[::-1]
            eig_vals = eig_vals[order]
            eig_vecs = eig_vecs[:, order]
            st.subheader("Eigenvalues")
            st.write(eig_vals)

        st.subheader("Covariance Matrix")
        st.write(cov_matrix)
        st.subheader("Eigenvectors")
        st.write(eig_vecs)

        st.subheader("PCA Projection (3D to 2D)")
        proj = data_centered @ eig_vecs[:, :2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=proj[:, 0], y=proj[:, 1], mode='markers', name='Projected Data'))
        fig.update_layout(title="PCA 2D Projection", xaxis_title="PC1", yaxis_title="PC2", height=500)
        st.plotly_chart(fig)

        st.subheader("3D Data Plot")
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode='markers', name='Original Data'))
        fig_3d.update_layout(title="3D Data", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        st.plotly_chart(fig_3d)

        st.subheader(translate("Try Your Own Code", language))
        code = f"""# PCA example
import numpy as np

data = np.random.multivariate_normal({mean}, {cov}, {n_points})
data_centered = data - np.mean(data, axis=0)
cov_matrix = np.cov(data_centered, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
order = eig_vals.argsort()[::-1]
eig_vals = eig_vals[order]
eig_vecs = eig_vecs[:, order]
print("Eigenvalues:", eig_vals)
print("Eigenvectors:", eig_vecs)
"""
        edited_code = st.text_area("Edit and run this code:", code, height=220)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                for key in ['eig_vals', 'eig_vecs']:
                    if key in local_vars:
                        st.write(f"{key.capitalize()}:", local_vars[key])

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("Why center data before PCA?", ["To remove mean variance", "For numerical stability", "It's optional"])
        if quiz_q == "To remove mean variance":
            st.success("Correct! Centers data at origin for covariance.")
        elif quiz_q:
            st.error("Incorrect. Hint: Covariance is mean-centered.")

        download_jupyter(edited_code, "pca.ipynb")

    with tab2:
        st.subheader("Upload Your Own Data for PCA")
        uploaded_file = st.file_uploader("Upload CSV (numeric columns only)", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                data = df.select_dtypes(include=[np.number]).values
                if data.shape[1] < 2:
                    st.error("Need at least 2 numeric columns.")
                else:
                    pca = SkPCA(n_components=2)
                    transformed = pca.fit_transform(data)
                    fig_upload = px.scatter(x=transformed[:,0], y=transformed[:,1], title="PCA on Uploaded Data")
                    st.plotly_chart(fig_upload)
                    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

# --- 6. Linear Algebra ---
elif page == "6️⃣ Linear Algebra":
    st.title(translate("Linear Algebra with NumPy", language))
    st.markdown(
        """
        **Mathematics**: Solve Ax=b via Gaussian elimination; eigenvalues from det(A - λI) = 0.
        **NumPy**: `np.linalg.solve`, `det`, `eig`.
        **Use Cases**: Physics simulations, optimization.
        **Pitfall**: Singular matrices (det=0) raise LinAlgError.
        """
    )

    tab1, tab2 = st.tabs(["Main Demo", "Physics Simulation"])
    with tab1:
        n = st.slider("Matrix size", 2, 6, 3)
        A = np.random.randint(-10, 10, (n, n))
        b = np.random.randint(-10, 10, n)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matrix A")
            st.write(A)
            st.subheader("Vector b")
            st.write(b)
        with col2:
            try:
                x = np.linalg.solve(A, b)
                st.subheader("Solution x (Ax = b)")
                st.write(x)
            except np.linalg.LinAlgError as e:
                st.error(f"Error: {e}")
            det = np.linalg.det(A)
            st.subheader("Determinant")
            st.write(det)
            eig_vals, eig_vecs = np.linalg.eig(A)
            st.subheader("Eigenvalues")
            # Handle complex eigenvalues
            st.text(str(eig_vals))  # Use st.text to avoid Arrow error
            st.subheader("Eigenvectors")
            st.text(str(eig_vecs))  # Use st.text for complex arrays

        st.subheader(translate("Try Your Own Code", language))
        code = f"""# Linear algebra example
import numpy as np

A = np.array({A.tolist()})
b = np.array({b.tolist()})

x = np.linalg.solve(A, b)  # Solve Ax = b
det = np.linalg.det(A)
eig_vals, eig_vecs = np.linalg.eig(A)
print("Solution x:", x)
print("Determinant:", det)
print("Eigenvalues:", eig_vals)
"""
        edited_code = st.text_area("Edit and run this code:", code, height=220)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                for key in ['x', 'det', 'eig_vals']:
                    if key in local_vars:
                        if key == 'eig_vals':
                            st.write("Eigenvalues:", str(local_vars[key]))  # Convert to string
                        else:
                            st.write(f"{key.capitalize()}:", local_vars[key])

        fig_eig = px.bar(y=np.abs(eig_vals), title="Eigenvalue Magnitudes")
        st.plotly_chart(fig_eig)

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("What if det(A) = 0?", ["Infinite solutions or none", "Unique solution", "Always solvable"])
        if quiz_q == "Infinite solutions or none":
            st.success("Correct! Singular matrix.")
        elif quiz_q:
            st.error("Incorrect. Hint: Indicates dependent equations.")

        download_jupyter(edited_code, "linear_algebra.ipynb")

    with tab2:
        st.subheader("Application: Projectile Motion Simulation")
        st.markdown("**Physics**: Model trajectory using linear equations for position.")
        v0 = st.slider("Initial velocity (m/s)", 10, 50, 20)
        theta = st.slider("Angle (degrees)", 0, 90, 45)
        g = 9.81
        t = np.linspace(0, 2 * v0 * np.sin(np.deg2rad(theta)) / g, 100)  # Time to hit ground
        theta_rad = np.deg2rad(theta)
        x = v0 * np.cos(theta_rad) * t
        y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
        fig_proj = px.line(x=x, y=y, labels={'x': 'Distance (m)', 'y': 'Height (m)'}, title="Projectile Trajectory")
        st.plotly_chart(fig_proj)

# --- 7. Random Module ---
elif page == "7️⃣ Random Module":
    st.title(translate("NumPy Random Module", language))
    st.markdown(
        """
        **Mathematics**: PRNG (Mersenne Twister) for uniform; Box-Muller for normal.
        **NumPy**: `np.random.uniform`, `randn`, `poisson`, etc.
        **Use Cases**: Simulations, ML data augmentation, Monte Carlo methods.
        **Pitfall**: Set seed for reproducibility; not cryptographically secure.
        """
    )

    tab1, tab2 = st.tabs(["Distributions", "Monte Carlo Simulation"])
    with tab1:
        dist = st.selectbox("Distribution", ["Uniform", "Normal", "Poisson", "Binomial"])
        n_samples = st.slider("Samples", 100, 5000, 1000)
        seed = st.number_input("Seed (optional)", value=None, step=1)
        if seed is not None:
            np.random.seed(int(seed))

        # Define parameters first to avoid NameError
        params = {}
        if dist == "Uniform":
            low, high = st.slider("Bounds", -10.0, 10.0, (0.0, 1.0))
            params = {"low": low, "high": high}
            data = generate_random_data(dist, (low, high), n_samples)
        elif dist == "Normal":
            mean = st.slider("Mean", -5.0, 5.0, 0.0)
            std = st.slider("Std Dev", 0.1, 5.0, 1.0)
            params = {"mean": mean, "std": std}
            data = generate_random_data(dist, (mean, std), n_samples)
        elif dist == "Poisson":
            lam = st.slider("Lambda", 1.0, 10.0, 4.0)
            params = {"lam": lam}
            data = generate_random_data(dist, (lam,), n_samples)
        elif dist == "Binomial":
            n_trials = st.slider("Trials", 1, 20, 10)
            p = st.slider("Probability", 0.0, 1.0, 0.5)
            params = {"n_trials": n_trials, "p": p}
            data = generate_random_data(dist, (n_trials, p), n_samples)

        st.subheader(f"{dist} Samples")
        st.write(data[:10])

        st.subheader("Histogram")
        fig_hist = px.histogram(data, nbins=50, title=f"{dist} Distribution Histogram")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader(translate("Try Your Own Code", language))
        code_params = {
            "Uniform": f"np.random.uniform(low={params.get('low', 0.0)}, high={params.get('high', 1.0)}, size={n_samples})",
            "Normal": f"np.random.normal(loc={params.get('mean', 0.0)}, scale={params.get('std', 1.0)}, size={n_samples})",
            "Poisson": f"np.random.poisson(lam={params.get('lam', 4.0)}, size={n_samples})",
            "Binomial": f"np.random.binomial(n={params.get('n_trials', 10)}, p={params.get('p', 0.5)}, size={n_samples})",
        }
        code = f"""# Random {dist} distribution
import numpy as np
np.random.seed({seed if seed else 'None'})

data = {code_params[dist]}
print(data[:10])
"""
        edited_code = st.text_area("Edit and run this code:", code, height=200)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                if 'data' in local_vars:
                    st.write(local_vars['data'][:10])
                else:
                    st.write("Code ran but no 'data' variable found.")

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("What does np.random.seed do?", ["Sets reproducibility", "Generates seeds", "Randomizes more"])
        if quiz_q == "Sets reproducibility":
            st.success("Correct! Fixes the random sequence.")
        elif quiz_q:
            st.error("Incorrect. Hint: Ensures consistent results.")

        download_jupyter(edited_code, "random_module.ipynb")

    with tab2:
        st.subheader("Application: Monte Carlo Estimation of π")
        n_points = st.slider("Points for Monte Carlo", 1000, 50000, 10000)
        points = np.random.uniform(-1, 1, (n_points, 2))
        inside = np.sum(points[:, 0]**2 + points[:, 1]**2 <= 1)
        pi_est = 4 * inside / n_points
        st.write(f"Estimated π: {pi_est}")
        fig_mc = go.Figure()
        colors = ['blue' if x**2 + y**2 <= 1 else 'red' for x, y in points]
        fig_mc.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(color=colors, size=3)))
        fig_mc.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line_color="blue")
        fig_mc.update_layout(title="Monte Carlo π Estimation", height=500)
        st.plotly_chart(fig_mc)

# --- 8. Multidimensional Arrays ---
elif page == "8️⃣ Multidimensional Arrays":
    st.title(translate("Multidimensional Arrays", language))
    st.markdown(
        """
        **Core**: ndarrays with shape, dtype, strides for memory layout.
        **Efficiency**: Contiguous C-order memory; views save memory.
        **Use Cases**: Tensors in deep learning, scientific data.
        **Pitfall**: Modifying views affects original; use `.copy()`.
        """
    )

    tab1, tab2 = st.tabs(["Manipulation Demo", "Memory Layout Visual"])
    with tab1:
        dims = st.selectbox("Dimensions", [2, 3])
        shape = tuple(st.slider(f"Size Dim {i+1}", 2, 5, 3) for i in range(dims))
        arr = np.arange(np.prod(shape)).reshape(shape)

        st.subheader("Array")
        st.write(arr)

        reshape_str = st.text_input("Reshape (e.g., 3,3 for 2D)", ",".join(map(str, shape)))
        try:
            new_shape = tuple(map(int, reshape_str.split(",")))
            if np.prod(new_shape) == np.prod(shape):
                reshaped = arr.reshape(new_shape)
                st.subheader("Reshaped")
                st.write(reshaped)
            else:
                st.error("New shape must have same total size.")
        except:
            st.error("Invalid shape.")

        slice_str = st.text_input("Slice (e.g., :2,:,1 for 3D)", ":,:")
        try:
            sliced = eval(f"arr[{slice_str}]")
            st.subheader("Sliced")
            st.write(sliced)
        except:
            st.error("Invalid slice.")

        st.subheader(translate("Try Your Own Code", language))
        code = f"""# Multidimensional array example
import numpy as np

arr = np.arange({np.prod(shape)}).reshape{shape}
reshaped = arr.reshape({shape})  # Reshape array
sliced = arr[:2]  # Slice creates view
print("Reshaped:", reshaped)
print("Sliced:", sliced)
"""
        edited_code = st.text_area("Edit and run this code:", code, height=200)
        if st.button(translate("Run", language)):
            local_vars, error = run_user_code(edited_code, {"np": np})
            if error:
                st.error(f"Error in code: {error}")
            else:
                st.write("Output:")
                for key in ['reshaped', 'sliced']:
                    if key in local_vars:
                        st.write(f"{key.capitalize()}:", local_vars[key])

        if dims == 2:
            fig_arr = px.imshow(arr, text_auto=True, title="2D Array Heatmap")
            st.plotly_chart(fig_arr)

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("What are strides?", ["Memory steps per dim", "Array shape", "Data type"])
        if quiz_q == "Memory steps per dim":
            st.success("Correct! Defines memory traversal.")
        elif quiz_q:
            st.error("Incorrect. Hint: Related to views.")

        download_jupyter(edited_code, "multidim_arrays.ipynb")

    with tab2:
        st.subheader("Memory Layout: Strides Demo")
        st.write("Original Strides:", arr.strides)
        transposed = arr.T if dims == 2 else arr.transpose(2,1,0)
        st.write("Transposed Strides:", transposed.strides)
        st.markdown("**Explanation**: Transpose shares memory but adjusts strides for access.")

# --- 9. Library Integrations ---
elif page == "9️⃣ Library Integrations":
    st.title(translate("Library Integrations", language))
    st.markdown(
        """
        **Concepts**: NumPy integrates with Pandas (dataframes), SciPy (math), Scikit-learn (ML).
        **Use Cases**: Data pipelines, optimization, modeling.
        **Pitfall**: Ensure compatible dtypes; `.values` for NumPy arrays.
        """
    )

    tab1, tab2 = st.tabs(["Pandas + NumPy", "SciPy/Scikit-learn Demo"])
    with tab1:
        st.subheader("Integration: NumPy + Pandas")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Pandas DataFrame:")
                st.dataframe(df.head())
                data_np = df.select_dtypes(include=[np.number]).values
                st.subheader("NumPy Array from DF")
                st.write(data_np[:5])
                mean = np.mean(data_np, axis=0)
                std = np.std(data_np, axis=0)
                normalized = (data_np - mean) / (std + 1e-8)  # Avoid div by zero
                st.subheader("Normalized (NumPy)")
                st.write(normalized[:5])
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

        st.subheader(translate("Try Your Own Code", language))
        code = """# Pandas to NumPy example
import numpy as np
import pandas as pd

# Simulate df = pd.read_csv('file.csv')
df = pd.DataFrame(np.random.rand(5,3), columns=['A','B','C'])
data_np = df.values
mean = np.mean(data_np, axis=0)
print("Mean:", mean)
"""
        edited_code = st.text_area("Edit and run (simulates df if no file):", code, height=200)
        if st.button(translate("Run", language)):
            sim_df = pd.DataFrame(np.random.rand(5,3), columns=['A','B','C'])
            local_vars, error = run_user_code(edited_code, {"np": np, "pd": pd, "df": sim_df})
            if error:
                st.error(f"Error: {error}")
            else:
                st.write("Output (using simulated df):")
                st.write(local_vars)

        st.subheader(translate("Quiz", language))
        quiz_q = st.radio("How to get NumPy from Pandas DF?", ["df.values", "df.array", "np.from_df"])
        if quiz_q == "df.values":
            st.success("Correct! Returns underlying NumPy array.")
        elif quiz_q:
            st.error("Incorrect. Hint: Attribute for array view.")

        download_jupyter(edited_code, "integrations.ipynb")

    with tab2:
        st.subheader("SciPy Optimization & Scikit-learn PCA")
        from scipy.optimize import minimize
        def objective(x):
            return x[0]**2 + x[1]**2
        result = minimize(objective, [1,1])
        st.write("SciPy Minimize (x^2 + y^2 from [1,1]):", result.x)

        data = np.random.rand(100, 5)
        pca = SkPCA(n_components=2)
        transformed = pca.fit_transform(data)
        fig_sk = px.scatter(x=transformed[:,0], y=transformed[:,1], title="Sklearn PCA on Random Data")
        st.plotly_chart(fig_sk)

# --- 10. Help & Cheat Sheet ---
elif page == "📖 Help & Cheat Sheet":
    st.title(translate("Help & Cheat Sheet", language))
    st.markdown(
        """
        **NumPy Cheat Sheet** (Key Functions):
        - **Creation**: `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`
        - **Math**: `np.add`, `np.multiply`, `np.sin`, `np.exp`, `np.log`
        - **Stats**: `np.mean`, `np.std`, `np.median`, `np.corrcoef`
        - **Linear Algebra**: `np.dot`, `np.linalg.inv`, `np.linalg.eig`, `np.linalg.svd`
        - **Random**: `np.random.rand`, `np.random.randn`, `np.random.choice`
        - **Manipulation**: `np.reshape`, `np.transpose`, `np.concatenate`
        **Resources**:
        - [NumPy Docs](https://numpy.org/doc/stable/)
        - [SciPy](https://scipy.org), [Pandas](https://pandas.pydata.org), [Scikit-learn](https://scikit-learn.org)
        **Animation Tips**: Use checkboxes for animation control and session state for smooth cycling without errors.
        """
    )