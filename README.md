# CircuitVision: AI-Powered Electronic Circuit Analysis

## 🚀 Overview
CircuitVision is a cutting-edge application that revolutionizes the way electronic circuit images are analyzed. By intelligently combining state-of-the-art AI models, CircuitVision transforms static images into actionable insights, automatically detecting components, understanding their connections, and generating SPICE-compatible netlists. Whether you're reverse-engineering a PCB, digitizing an old schematic, or quickly needing to simulate a photographed circuit, CircuitVision provides a powerful and intuitive solution.

It leverages **YOLO** for initial component detection, **SAM2 (Segment Anything Model 2)** for precise circuit segmentation and intelligent cropping, and **Gemini** for advanced component value extraction, delivering a comprehensive analysis pipeline.

## ✨ Key Features

### 👁️ Advanced Component Detection
*   **Precision with YOLO:** Utilizes YOLO (You Only Look Once) for rapid and accurate identification of various electronic components.
*   **Refined by NMS:** Employs Non-Maximum Suppression (NMS) based on confidence scores to eliminate redundant bounding boxes, ensuring cleaner and more precise detection.
*   **Adaptive Bounding Boxes:** Component bounding boxes dynamically adjust if SAM2-based intelligent cropping is applied, maintaining accuracy on the focused region of interest.

### 📐 Intelligent Image Segmentation & Cropping
*   **SAM2 for Accuracy:** Integrates the Segment Anything Model 2 (SAM2) for highly detailed segmentation, accurately isolating the primary circuit board area from its surroundings.
*   **Smart Cropping:** The image and SAM2 mask are intelligently cropped based on the segmented circuit extent (with optimal padding). This focuses subsequent analysis solely on the relevant circuit area, boosting performance and accuracy.
*   **Graceful Fallback:** If SAM2 models are unavailable or disabled, the system seamlessly defaults to analyzing the original, uncropped image.

### 🔗 Robust Node Analysis
*   **Mask-Driven Connections:** Node connection analysis leverages the cropped SAM2 binary mask (after excluding component areas) and adjusted YOLO bounding boxes for superior precision in identifying conductive paths and nodes.
*   **Overhauled Corner Detection:** Features a custom pixel-based corner finding algorithm, replacing older methods for improved robustness across diverse image types. Includes fallback mechanisms and empty coordinate checks.

### 📝 Automated Netlist Generation
*   **Two-Stage Process for Detail:**
    1.  **Initial Structural Netlist:** An initial "valueless" netlist is generated based on detected components and their geometric interconnections.
    2.  **Gemini-Powered Value Enrichment:** An enumerated image (marking each component) is sent to a Gemini vision model to accurately identify component types (Resistors, Capacitors, ICs, etc.) and their values (10kΩ, 100uF, etc.), producing a final, value-enriched netlist.
*   **SPICE-Ready Output:** The final netlist is filtered to remove invalid entries, ensuring a clean and reliable output ready for SPICE simulation.

### 🖥️ Intuitive User Interface & Visualization
*   **Effortless Workflow:** Analysis auto-triggers on file upload, eliminating manual steps. A custom loading animation provides feedback.
*   **Comprehensive Results Display:**
    *   Visualize each step: annotated component detections, SAM2 segmentation masks, node connections.
    *   Detailed timings for each major analysis stage (YOLO, SAM2, Cropping, Nodes, Netlist).
    *   Display of original image properties and extracted EXIF data.
*   **EXIF-Aware Processing:** Images are automatically rotated based on EXIF orientation data before analysis.
*   **Debugging Aids:** Expander sections reveal debug images and the enumerated image sent to Gemini.

## 📈 Recent Enhancements
CircuitVision is continuously evolving! We've recently rolled out significant updates to the core analysis pipeline, user interface, and netlist generation capabilities. For a comprehensive list of changes, including technical deep-dives and commit references, please see our detailed **[CHANGELOG.md](./CHANGELOG.md)**.

Key highlights since the introduction of advanced NMS include:
*   **SAM2 Integration:** For intelligent circuit segmentation and cropping.
*   **Gemini-Powered Netlists:** For accurate component value extraction.
*   **UI Overhaul:** Streamlined workflow and richer visualizations.
*   **Corner Detection Rework:** For more robust node analysis.

## 🛠️ Setup

1.  **Models:** Ensure the following models are accessible to the application:
    *   YOLO detection model (`.pt` or equivalent)
    *   SAM2 segmentation model files (if using this feature)
    *   Access configured for the Gemini API (for component value extraction)

    You can download the required YOLO and SAM2 models by running the provided script:
    ```bash
    python download_models.py
    ```
    This will create `models/YOLO` and `models/SAM2` directories and populate them with the necessary model files. Make sure you have `requests` and `gdown` installed (`pip install requests gdown`) or run the script in an environment where these are available (like the Docker container after initial setup).

2.  **Dependencies:** Install all required Python packages.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configuration:** Review and adjust any necessary paths or API keys in the application's configuration settings.

### 🐳 Using Docker

For a containerized setup, you can use Docker and Docker Compose. This is the recommended way to manage dependencies and ensure a consistent environment.

1.  **Build the Docker Image:**
    Navigate to the project root directory (where the `Dockerfile` is located) and run:
    ```bash
    docker build -t circuitvision .
    ```
    (If a `docker-compose.yml` file is provided, you can often skip the manual build and just use `docker-compose up --build`.)

2.  **Run with Docker Compose (Recommended if `docker-compose.yml` exists):**
    If a `docker-compose.yml` file is present in the project root, you can typically build and run the application with a single command:
    ```bash
    docker-compose up --build
    ```
    To run in detached mode (in the background):
    ```bash
    docker-compose up -d --build
    ```

3.  **Run the Docker Container (Manual, if not using Docker Compose):**
    After building the image, you can run it as a container. You'll need to map the necessary ports (e.g., for the web UI) and potentially mount volumes for models or configuration if they are not already copied into the image.
    Example (assuming the app runs on port 8501 internally):
    ```bash
    docker run -p 8501:8501 -v ./models:/app/models circuitvision
    ```
    *Adjust port mappings and volume mounts (`-v`) as per your application's specific needs and how the `Dockerfile` is structured.* 

    *Inside the Docker environment, you might still need to run `python download_models.py` if the models are not part of the Docker image build process, or ensure your volume mounts point to a location where models have been downloaded on your host machine.*

## 🚀 Usage

1.  **Upload Image:** Launch the CircuitVision application and upload an image of the electronic circuit you wish to analyze.
2.  **Automatic Analysis:** The system will automatically:
    *   Pre-process the image (including EXIF-based rotation).
    *   Detect components using YOLO and refine with NMS.
    *   Segment the circuit area using SAM2 (if available) and crop the image.
    *   Perform node analysis on the processed image.
    *   Generate an initial structural netlist.
    *   Extract component values using Gemini to produce the final netlist.
3.  **Review Results:** Examine the detailed visualizations for each analysis stage, component statistics, timing information, and both the initial and final netlists.
4.  **SPICE Simulation:** Utilize the generated final netlist with your preferred SPICE simulation software.

---

We hope you find CircuitVision powerful and easy to use! 