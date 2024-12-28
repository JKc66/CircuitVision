from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import numpy as np
import cv2
import os
import shutil
from .utilities import summarize_components, gemini_labels
from .circuit_analyzer import CircuitAnalyzer
from copy import deepcopy
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Simulation import CircuitSimulation

analyzer = CircuitAnalyzer(yolo_path='sdp_demo/best_large_model.pt', debug=False)
files_location = 'sdp_demo/static/assets/uploads/'
components_dict = {
    'gnd': 'Ground: A reference point in an electrical circuit. has no direction nor value (both None).',
    'voltage.ac': 'AC Voltage source, has no direction (None). if its value is written in phasor, write it as magnitude:phase',
    'voltage.dc': 'DC Voltage source, has a direction, should be specified as either up, down, left or right, depending on the direction of the + sign',
    'voltage.battery': 'Battery Voltage source, has the direction of the largest horizontal bar of its symbol (up, down, left or right)',
    'resistor': 'Resistor: A passive component with no direction (None)',
    'voltage.dependent': 'Voltage-Dependent Source: A voltage source whose output voltage depends on another voltage or current in the circuit. has the direction of the + sign (up, down, left or right).',
    'current.dc': 'DC Current: Direct current, where the current flows in one direction consistently, has the direction of the arrow inside the symbol (up, down, left or right).',
    'current.dependent': 'Current-Dependent Source: A current source whose output current depends on another current or voltage in the circuit. has the direction of the arrow inside the symbol (up, down, left or right).',
    'capacitor': 'Capacitor: A passive component with no direction (None)',
    'inductor': 'Inductor: A passive component with no direction (None)',
    'diode': 'Diode: has the direction of the symbol which is like an arrow (up, down, left or right)'
}

def home(request):
    return render(request, 'C:/Users/mahmo/OneDrive/Desktop/SDP/webapp_demo/sdp_demo/sdp_demo/templates/index.html')

@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        filename = uploaded_file.name

        try: 
            # Read the in-memory file as a NumPy array for use in OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # or cv2.IMREAD_GRAYSCALE for grayscale
            print(filename)
            print("Image Shape:", image.shape) #e.g (height, width, channels)

            
            #save the image locally 
            shutil.rmtree(files_location)
            os.mkdir(files_location)
            image_path = files_location + '1.' + filename.split('.')[-1]
            cv2.imwrite(image_path, image)

            #detection bboxes section
            bboxes = analyzer.bboxes(image)
            detection_summary = summarize_components(bboxes)
            print(detection_summary)
            annotated_image = analyzer.get_annotated(image, bboxes)
            annotated_path = files_location + '2.' + filename.split('.')[-1]
            cv2.imwrite(annotated_path, annotated_image)

            #node section
            nodes, emptied_mask, enhanced, contour_image, corners_image, final_visualization = analyzer.get_node_connections(image, bboxes)
            emptied_path = files_location + '3.' + filename.split('.')[-1]
            cv2.imwrite(emptied_path, emptied_mask)
            contour_path = files_location + '4.' + filename.split('.')[-1]
            contour_text = f"{len(nodes)} nodes detected, excluding any noise initially marked as a node"
            cv2.imwrite(contour_path, contour_image)
            final_path = files_location + '5.' + filename.split('.')[-1]
            cv2.imwrite(final_path, final_visualization)

            #value-less netlist
            valueless_netlist = analyzer.generate_netlist_from_nodes(nodes)
            valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
            netlist = deepcopy(valueless_netlist)

            #fixed netlist
            enum_img, bbox_ids= analyzer.enumerate_components(image, bboxes)
            enum_path = files_location + '6.' + filename.split('.')[-1]
            print(enum_img.shape)
            cv2.imwrite(enum_path, enum_img)
            gemini_info = gemini_labels(enum_img)
            print(gemini_info)
            analyzer.fix_netlist(netlist, gemini_info)
            netlist_text = '\n'.join([analyzer.stringify_line(line) for line in netlist])

            # spice
            net_text = '.title detected_circuit\n' + netlist_text
            parser = SpiceParser(source=net_text)
            bootstrap_circuit = parser.build_circuit()
            print(bootstrap_circuit)
            simulator = bootstrap_circuit.simulator()
            analysis = simulator.operating_point()
            analysis_text = f"Node Voltages: {analysis.nodes}\n\nBranch currents: {analysis.branches}"

        except Exception as e:
            return HttpResponse(f"Error processing image: {e}")

        return JsonResponse({'status': 'success', 'detection_path': annotated_path, 'detection_summary':detection_summary, 'emptied_path': emptied_path, "contour_path": contour_path, "contour_text":contour_text, 'final_path':final_path, 'valueless_netlist':valueless_netlist_text, 'enum_path':enum_path, 'fixed_netlist':netlist_text, "analysis_text":analysis_text})
    return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)