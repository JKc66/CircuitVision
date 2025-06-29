# CircuitVision ⚡🌿: AI-Powered Electrical Circuit Analysis & Netlist Generation

<p align="center">
  <img width="60%" src="static/images/CircuitVision-nobg_dark.png" />
</p>

<p align="center">
  <em>From picture to simulation. An AI-powered app that automatically analyzes circuit diagrams and generates SPICE netlists.</em>
</p>

<p align="center">
    <a href="https://app.jawadk.me/CircuitVision/" target="_blank">
        <img src="https://img.shields.io/badge/Live_Demo-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Live Demo">
    </a>
    <a href="https://github.com/JKc66/CircuitVision" target="_blank">
        <img src="https://img.shields.io/badge/View_on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="View on GitHub">
    </a>
</p>

<p align="center">
  <a href="https://github.com/JKc66/CircuitVision/blob/main/LICENSE" target="_blank">
    <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-green.svg" />
  </a>
  <a href="https://github.com/JKc66/CircuitVision/issues" target="_blank">
    <img alt="Issues" src="https://img.shields.io/github/issues/JKc66/CircuitVision" />
  </a>
  <a href="https://deepwiki.com/JKc66/CircuitVision" target="_blank">
    <img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg" />
  </a>
</p>


<p align="center">
  <a href="https://github.com/mah-sam">
    <img alt="Mahmoud" src="https://img.shields.io/badge/Mahmoud-100000?style=for-the-badge&logo=github&logoColor=white&labelColor=black" />
  </a>
  <a href="https://github.com/JKc66">
    <img alt="Jawad" src="https://img.shields.io/badge/Jawad-100000?style=for-the-badge&logo=github&logoColor=white&labelColor=black" />
  </a>
</p>

## 📚 Table of Contents
- [CircuitVision ⚡🌿: AI-Powered Electrical Circuit Analysis \& Netlist Generation](#circuitvision--ai-powered-electrical-circuit-analysis--netlist-generation)
  - [📚 Table of Contents](#-table-of-contents)
  - [🚀 Overview](#-overview)
  - [✨ Key Features \& The Engineering Behind Them](#-key-features--the-engineering-behind-them)
    - [👁️ Advanced Component Detection (Fine-Tuned YOLOv11)](#️-advanced-component-detection-fine-tuned-yolov11)
    - [📐 Intelligent Cropping \& Precise Segmentation (YOLO + SAM 2)](#-intelligent-cropping--precise-segmentation-yolo--sam-2)
    - [🔗 Custom-Developed Node \& Connection Analysis](#-custom-developed-node--connection-analysis)
    - [📝 Automated \& Enriched Netlist Generation](#-automated--enriched-netlist-generation)
    - [🖥️ Intuitive User Interface \& Rich Visualization](#️-intuitive-user-interface--rich-visualization)
  - [📈 Our Development Journey \& Enhancements](#-our-development-journey--enhancements)
  - [🛠️ Technology Stack](#️-technology-stack)
  - [🛠️ Setup](#️-setup)
    - [🐳 Using Docker (Recommended)](#-using-docker-recommended)
  - [🚀 Usage](#-usage)
  - [🧠 Training Code](#-training-code)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [✒️ Citation](#️-citation)

## 🚀 Overview

CircuitVision is an innovative application designed to bridge the gap between visual **electrical circuit** diagrams and functional simulations. It intelligently analyzes images of **electrical circuits**—whether hand-drawn, photographed, or from schematics—and transforms them into SPICE-compatible netlists, providing an easy access to the operating point parameters of the circuit. This project leverages a sophisticated pipeline of **fine-tuned and custom-adapted AI models**, including **YOLOv11** for component detection, an **adapted SAM 2 (Segment Anything Model 2)** for precise segmentation and intelligent cropping, and **Google's Gemini** for its robust OCR ability.

Our goal is to automate the tedious process of manual circuit transcription, enabling engineers, students, and hobbyists to quickly digitize, understand, and simulate **electrical circuits** with unprecedented ease and accuracy.

## ✨ Key Features & The Engineering Behind Them

<p align="center">
  <b>Phase 1: Component Detection</b><br>
  <img width="100%" src="static/Figures/Darkmode_CircuitVision2x_phase1.png" />
  <b>Phase 2: Topology Analysis</b><br>
  <img width="100%" src="static/Figures/Darkmode_CircuitVision2x_phase2.png" />
  <b>Phase 3: Netlist Generation</b><br>
  <img width="100%" src="static/Figures/Darkmode_CircuitVision2x_phase3.png" />
  <b>Phase 4: Simulation</b><br>
  <img width="100%" src="static/Figures/Darkmode_CircuitVision2x_phase4.png" />
</p>
<p align="center"><i>The four phases of the CircuitVision pipeline: Detection, Topology Analysis, Netlist Generation, and Simulation.</i></p>

### 👁️ Advanced Component Detection (Fine-Tuned YOLOv11)
* Utilizes a **YOLOv11 model fine-tuned specifically on electrical circuit component datasets**. This training enables robust identification of diverse **electrical components** even in complex or noisy images. **We obtained a mean average percision (mAp-50) of 0.9313, which is on bar with SOTA detection models.**
* Employs Non-Maximum Suppression (NMS) based on confidence scores to eliminate redundant detections, ensuring a clean and precise component map.
* Bounding boxes are re-calculated and adjusted after the initial crop to maintain coordinate accuracy relative to the new, focused region of interest.

### 📐 Intelligent Cropping & Precise Segmentation (YOLO + SAM 2)
* **Intelligent YOLO-Based Cropping:** Before segmentation, the application performs an intelligent crop based on the collective bounding box of all components detected by YOLO. This crucial first step isolates the relevant circuit area, significantly boosting the performance and accuracy of subsequent steps by removing background noise.
* **Precise Segmentation with SAM 2:** After cropping, a fine-tuned **Segment Anything Model 2 (SAM 2)** is applied to the focused circuit area. This generates a highly detailed binary mask of the conductive traces, which is essential for accurate node and connection analysis. Our fine-tuned SAM 2 model achieves **98.7% accuracy** on circuit segmentation from a small dataset of only 267 images. (This work will be submitted to a journal soon).


### 🔗 Custom-Developed Node & Connection Analysis
* Our **custom-developed node connection algorithm** leverages the SAM 2-generated binary mask (after excluding component areas) and the adjusted YOLO bounding boxes. This allows for superior precision in identifying conductive traces and connection nodes within the **electrical circuit**.
* Features a **newly implemented pixel-based corner finding algorithm**, replacing previous methods for enhanced robustness and accuracy across diverse circuit image styles and qualities. Includes comprehensive fallback mechanisms and empty coordinate checks.

### 📝 Automated & Enriched Netlist Generation
* **Two-Stage Intelligent Process:**
    1.  **Structural Netlist:** An initial "valueless" netlist is generated based on the detected components (from fine-tuned YOLO) and their geometric interconnections (from custom node analysis).
    2.  **Gemini-Powered Value & Type Enrichment:** An enumerated image, clearly marking each detected component with a unique ID, is passed to the **Gemini Pro Vision model**. Gemini then performs advanced OCR and contextual understanding to accurately identify component types (Resistors, Capacitors, Voltage/Current Sources, etc.) and extract their corresponding values (e.g., 10kΩ, 100µF, 12V). This step also serves as a cross-validation for component types initially detected by YOLO.
* **SPICE-Ready Output:** The final netlist is meticulously filtered to remove invalid or incomplete entries, ensuring a clean, reliable, and simulation-ready output for SPICE-based tools.

### 🖥️ Intuitive User Interface & Rich Visualization
* **Streamlined Workflow:** Analysis is automatically triggered upon image upload, with a custom loading animation providing real-time feedback.
* **Comprehensive Results Dashboard:**
    *   Visualize each critical step: annotated component detections (YOLO), SAM 2 segmentation masks, identified node connections.
    *   Detailed performance timings for each major analysis stage (YOLO, SAM 2, Cropping, Node Analysis, Netlist Generation).
    *   Display of original image properties and extracted EXIF data for traceability.
* **Intelligent Preprocessing:** Images are automatically rotated based on EXIF orientation data before any analysis begins.
* **Debugging & Transparency:** Dedicated expander sections reveal intermediate debug images, including the enumerated image sent to Gemini, providing insight into the AI's "reasoning."

## 📈 Our Development Journey & Enhancements
CircuitVision is the culmination of significant research and development in applying and adapting cutting-edge AI to the specialized domain of **electrical circuit analysis**. We've moved beyond off-the-shelf model usage to:
*   **Model Fine-Tuning:** Extensive work was done to **fine-tune YOLOv11** on specific **electrical circuit datasets** to maximize component recognition accuracy.
*   **SAM 2 Adaptation:** We've **custom-adapted the application of SAM 2** for optimal circuit segmentation, leading to the intelligent cropping feature which is key to downstream accuracy.
*   **Algorithm Development:** The node analysis and corner detection algorithms are **custom-built** for the unique challenges of **electrical circuit diagrams**.
*   **Multi-Model Orchestration:** Integrating YOLO, SAM 2, and Gemini into a cohesive and efficient pipeline represents a complex engineering achievement.

## 🛠️ Technology Stack
<p align="center">
  <a href="https://www.python.org/" target="_blank"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" /></a>
  <a href="https://pytorch.org/" target="_blank"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" /></a>
  <a href="https://streamlit.io/" target="_blank"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" /></a>
  <a href="https://www.docker.com/" target="_blank"><img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" /></a>
  <a href="https://deepmind.google/technologies/gemini/" target="_blank"><img alt="Gemini" src="https://img.shields.io/badge/Gemini-8E77F0?style=for-the-badge&logo=google&logoColor=white" /></a>
</p>

## 🛠️ Setup

1.  **Models & API Access:**
    *   **YOLO & SAM 2 Models:** Our system relies on specific pre-trained and fine-tuned model weights.
        Run the provided script to download the necessary model files:
        ```bash
        python download_models.py
        ```
        This will create `models/YOLO` and `models/SAM2` directories and populate them. Ensure you have `requests` and `gdown` installed (`pip install requests gdown`).
    *   **Gemini API Key:** You will need a Google Gemini API key. Set it as an environment variable `GEMINI_API_KEY` or configure it within the application as required.

2.  **Dependencies:** Install all required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### 🐳 Using Docker (Recommended)

For a consistent and isolated environment, we recommend using Docker.

1.  **Build the Docker Image:**
    From the project root directory (containing the `Dockerfile`):
    ```bash
    docker build -t circuitvision .
    ```

2.  **Run with Docker Compose (`docker-compose.yml`):**
    This is often the simplest way to manage multi-container applications or complex setups.
    ```bash
    docker compose up --build
    ```
    To run in detached mode:
    ```bash
    docker compose up -d --build
    ```
    *(Ensure your `GEMINI_API_KEY` is available to the Docker environment, e.g., via an `.env` file used by Docker Compose or by passing it as an environment variable in the `docker-compose.yml`)*

## 🚀 Usage

1.  **Launch Application:** Start the CircuitVision Streamlit app (e.g., `streamlit run app.py` or via Docker).
2.  **Upload Image:** Use the file uploader to select an image of the **electrical circuit** you wish to analyze.
3.  **Automatic Analysis:** The system will automatically:
    *   Pre-process the image (EXIF rotation).
    *   Perform component detection (YOLO).
    *   Segment and crop the circuit (SAM 2).
    *   Execute custom node analysis.
    *   Generate the initial structural netlist.
    *   Query Gemini for component values and types.
    *   Produce the final, enriched netlist.
    
## 🧠 Training Code

Explore the core training pipelines used to build the intelligence behind CircuitVision:

* **🟪 SAM 2 Training**
  Custom training notebook for **Segment Anything Model 2**, tailored specifically for electrical circuit segmentation.
  
  [![Open in Kaggle](https://img.shields.io/badge/Open%20in%20Kaggle-20BEFF?logo=kaggle&logoColor=fff)](https://www.kaggle.com/code/mah01sam/training-sam-2/notebook?scriptVersionId=237405492)


* **🟧 YOLOv11 Training**
  End-to-end training and inference pipeline for detecting circuit elements with **YOLOv11**.

  [![Open in Kaggle](https://img.shields.io/badge/Open%20in%20Kaggle-20BEFF?logo=kaggle&logoColor=fff)](https://www.kaggle.com/code/mah01sam/yolov11-training-and-project-pipeline)
 

---

## 🙏 Acknowledgements

We extend our deepest gratitude to our supervisor, Prof. Adel Abdennour, for his exceptional guidance, support, and invaluable insights throughout this Senior Design Project.

---

## ✒️ Citation

If you find CircuitVision useful in your research or work, please cite it as follows:

```bibtex
@misc{CircuitVision,
  author = {Sameh, Mahmoud and Khan, Jawad},
  title = {CircuitVision: AI-Powered Electrical Circuit Analysis & Netlist Generation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JKc66/CircuitVision}}
}
```
