import numpy as np
import cv2
import os
import shutil
from .utills import summarize_components, gemini_labels
from .circuit_analyzer import CircuitAnalyzer
from copy import deepcopy
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Simulation import CircuitSimulation
from pathlib import Path

analyzer = CircuitAnalyzer(yolo_path='models/models/YOLO/best_large_model_yolo.pt', debug=False)
files_location = 'static/assets/uploads/'
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

