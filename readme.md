## CircuitVision: Recognition and Analysis of Hand-Drawn Electrical Circuits using Artificial Intelligence

### Project Description

The project aims to develop an AI system named CircuitVision, capable of recognizing and analyzing hand-drawn electrical circuits from images. The system leverages a combination of computer vision techniques and artificial intelligence to:

-   Detect and classify circuit components
-   Extract component values
-   Generate netlist representations
-   Perform circuit simulations

### Key Features

-   **Hand-Drawn Circuit Recognition:** CircuitVision can accurately identify and classify various electrical components in hand-drawn circuit diagrams, including resistors, capacitors, inductors, voltage sources, current sources, and diodes.
-   **Component Value Extraction:** The system extracts numerical values associated with components, enabling the generation of accurate netlists.
-   **Netlist Generation:** CircuitVision automatically generates netlist representations of recognized circuits, facilitating circuit simulation and analysis using tools like SPICE.
-   **Circuit Simulation and Analysis:** The generated netlists can be used to perform circuit simulations, providing valuable insights into circuit behavior, such as node voltages, branch currents, and overall circuit performance.

### Code Structure

The code is structured into several modules, each responsible for a specific task in the circuit recognition and analysis pipeline.

-   **Dataset Preparation:** Modules for creating and managing datasets of hand-drawn circuit images and their corresponding annotations. This includes functions for data augmentation, preprocessing, and conversion to YOLO format.
-   **Object Detection (YOLO):** Implementation of the YOLO object detection model for identifying and localizing circuit components within images.
-   **Component Enumeration and Labeling:** Functions to assign unique identifiers to detected components and extract their values using OCR or vision language models like Gemini.
-   **Netlist Generation:** Modules for converting the detected components and their connections into a structured netlist representation.
-   **Circuit Simulation (PySpice):** Integration with PySpice for parsing the generated netlist and performing circuit simulations to determine electrical characteristics.
-   **Visualization and Analysis:** Tools for visualizing the recognized circuits, detected components, generated netlists, and simulation results.

### Usage Instructions

1.  **Dataset Setup:** Prepare a dataset of hand-drawn circuit images and annotations. Use provided scripts for data augmentation and conversion to YOLO format.
2.  **Model Training:** Train the YOLO object detection model using the prepared dataset. Adjust hyperparameters as needed for optimal performance.
3.  **Circuit Recognition and Analysis:** Use the trained YOLO model to process new circuit images. The system will automatically detect components, extract values, generate netlists, and perform simulations.
4.  **Visualize Results:** Examine the visualized outputs, including annotated images, generated netlists, and simulation plots, to gain insights into the recognized circuits.

### Dependencies

-   Python 3.x
-   OpenCV
-   TensorFlow or PyTorch (for YOLO)
-   PySpice
-   Ultralytics YOLO
-   Scikit-learn
-   Matplotlib
-   PIL

### Future Work

-   **Improved Component Recognition:** Enhance the accuracy of component detection and classification, especially for complex or uncommon components.
-   **Robust Value Extraction:** Develop more robust methods for extracting component values from handwritten text, potentially using advanced OCR techniques or vision language models.
-   **Enhanced Netlist Generation:** Improve the netlist generation process to handle a wider variety of circuit topologies and component connections.
-   **Integration with Circuit Design Tools:** Explore the integration of CircuitVision with existing circuit design software to provide a seamless workflow from hand-drawn sketches to simulation and analysis.

### Acknowledgments

This project was inspired by the research and advancements in the fields of computer vision, artificial intelligence, and circuit analysis. We acknowledge the contributions of researchers and developers who have paved the way for automated circuit recognition and analysis systems.

### Contact

For any questions or inquiries, please contact:

Mahmoud Sameh
Jawad Khan
Advisor: Prof. Adel Abdennour

### Contributing

Contributions to this project are welcome. Please follow the standard GitHub workflow for submitting pull requests.

### License

This project is licensed under the MIT License.

This response draws exclusively from the information provided in the sources. 
