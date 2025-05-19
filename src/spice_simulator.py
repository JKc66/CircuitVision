import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import u_Hz
from src.utils import _parse_vlm_ac_string, safe_to_complex

def perform_dc_spice_analysis(current_netlist_content, app_logger):
    """Performs DC SPICE analysis on the provided netlist content."""
    try:
        if current_netlist_content:
            app_logger.debug("Running DC SPICE analysis with netlist:")

            # Pre-process netlist for DC analysis: comment out C and L with reactance values
            processed_lines_for_dc = []
            for line in current_netlist_content.split('\n'):
                line_stripped = line.strip()
                if not line_stripped:
                    processed_lines_for_dc.append(line)
                    continue

                parts = line_stripped.split()
                if not parts:
                    processed_lines_for_dc.append(line)
                    continue
                
                # Ensure parts[0] exists and is not empty before accessing parts[0][0]
                component_char = ''
                if parts[0]:
                    component_char = parts[0][0].upper()
                
                is_problematic_for_dc = False
                if component_char in ['C', 'L'] and len(parts) >= 4:
                    value_part = parts[3]
                    if value_part.startswith('j') or value_part.startswith('-j'):
                        is_problematic_for_dc = True
                
                if is_problematic_for_dc:
                    # Comment out the line for DC analysis
                    processed_lines_for_dc.append(f"* {line} ; DC analysis: reactance value ignored")
                    app_logger.info(f"DC Analysis: Ignoring line due to reactance value: {line_stripped}")
                else:
                    processed_lines_for_dc.append(line)
            
            current_netlist_content_dc_safe = "\n".join(processed_lines_for_dc)
            app_logger.debug("DC-safe netlist:")
            app_logger.debug(current_netlist_content_dc_safe)

            net_text_dc = (
                '.title detected_circuit_dc\n'
                + current_netlist_content_dc_safe
                + '\n.end\n'
            )
            
            with st.expander("🔍 Debug: Complete SPICE Netlist (DC)"):
                st.code(net_text_dc, language="verilog") # Changed language for syntax highlighting
            
            app_logger.debug("Full DC SPICE netlist with control commands:")
            app_logger.debug(net_text_dc)
            
            parser = SpiceParser(source=net_text_dc)
            bootstrap_circuit = parser.build_circuit()
            
            app_logger.debug("Circuit elements (DC):")
            for element in bootstrap_circuit.elements:
                app_logger.debug(f"  {element}")
            
            simulator = bootstrap_circuit.simulator(
                temperature=27, nominal_temperature=27, gmin=1e-12,
                abstol=1e-12, reltol=1e-6, chgtol=1e-14,
                trtol=7, itl1=100, itl2=50, itl4=10
            )
            
            app_logger.debug("Running operating point analysis (DC)...")
            analysis = simulator.operating_point()
            
            app_logger.debug("Raw DC analysis results:")
            app_logger.debug("Nodes:")
            for node, value in analysis.nodes.items():
                app_logger.debug(f"  {node}: {value} (type: {type(value)})")
            app_logger.debug("Branches:")
            for branch, value in analysis.branches.items():
                app_logger.debug(f"  {branch}: {value} (type: {type(value)})")
            
            node_voltages_dc = {}
            for node, value in analysis.nodes.items():
                try:
                    voltage = float(value._value if hasattr(value, '_value') else value.get_value() if hasattr(value, 'get_value') else value)
                    node_voltages_dc[str(node)] = f"{voltage:.3f}V"
                except Exception as e_val:
                    app_logger.error(f"Error converting DC node voltage {node}: {str(e_val)}")
                    node_voltages_dc[str(node)] = "Error"
            
            branch_currents_dc = {}
            for branch, value in analysis.branches.items():
                try:
                    current = float(value._value if hasattr(value, '_value') else value.get_value() if hasattr(value, 'get_value') else value)
                    branch_currents_dc[str(branch)] = f"{current*1000:.3f}mA"
                except Exception as e_val:
                    app_logger.error(f"Error converting DC branch current {branch}: {str(e_val)}")
                    branch_currents_dc[str(branch)] = "Error"
            
            col1_dc, col2_dc = st.columns(2)
            with col1_dc:
                st.markdown("### Node Voltages (DC)")
                st.json(node_voltages_dc)
            with col2_dc:
                st.markdown("### Branch Currents (DC)")
                st.json(branch_currents_dc)
        else:
            st.error("Netlist is empty. Please generate or edit the netlist before running DC SPICE analysis.")
    
    except Exception as e_dc:
        error_msg_dc = f"❌ DC SPICE Analysis Error: {str(e_dc)}"
        app_logger.error(error_msg_dc, exc_info=True)
        st.error(error_msg_dc)

def perform_ac_spice_analysis(active_results_data, analyzer_instance, current_ac_frequency_hz, app_logger):
    """Performs AC SPICE analysis, including parsing, simulation, and plotting."""
    try:
        if active_results_data.get('netlist'):
            sim_netlist_data = deepcopy(active_results_data['netlist'])
            spice_body_lines = []
            
            for line_dict in sim_netlist_data:
                if line_dict.get('class') == 'gnd':
                    continue

                original_value = str(line_dict.get('value', ''))
                component_type_prefix = line_dict.get('component_type', '')

                if component_type_prefix in ['V', 'I']:
                    parsed_params = _parse_vlm_ac_string(original_value) # Imported from src.utils
                    if parsed_params:
                        line_dict['value'] = f"{parsed_params['dc_offset']} AC {parsed_params['mag']} {parsed_params['phase']}"
                        app_logger.debug(f"Processed AC source {line_dict.get('component_type','')}{line_dict.get('component_num','')}: original='{original_value}', spice_val='{line_dict['value']}'")
                    else:
                        if original_value.lower().strip().startswith('ac') or ':' in original_value:
                            default_ac_val = "0 AC 1 0"
                            st.warning(f"Could not parse AC parameters for {component_type_prefix}{line_dict.get('component_num','')}: '{original_value}'. Using default: {default_ac_val}")
                            app_logger.warning(f"Could not parse AC parameters for {component_type_prefix}{line_dict.get('component_num','')}: '{original_value}'. Using default: {default_ac_val}")
                            line_dict['value'] = default_ac_val
                elif component_type_prefix == 'C':
                    val_lower = original_value.lower()
                    component_name_debug = f"{component_type_prefix}{line_dict.get('component_num','')}"
                    if val_lower.startswith("-j"):
                        try:
                            reactance_str = val_lower[2:]
                            Xc = float(reactance_str) if reactance_str else 1.0
                            if Xc <= 0 or current_ac_frequency_hz <= 0:
                                app_logger.warning(f"Invalid Xc ({Xc}) or freq ({current_ac_frequency_hz}Hz) for {component_name_debug}. Using original: '{original_value}'")
                            else:
                                capacitance = 1 / (2 * np.pi * current_ac_frequency_hz * Xc)
                                line_dict['value'] = capacitance
                                app_logger.info(f"Processed Capacitor {component_name_debug}: original='{original_value}', Xc={Xc}, freq={current_ac_frequency_hz}Hz, C={capacitance:.4e}F")
                        except ValueError:
                            app_logger.warning(f"Could not parse Xc from '{original_value}' for {component_name_debug}. Using original.")
                elif component_type_prefix == 'L':
                    val_lower = original_value.lower()
                    component_name_debug = f"{component_type_prefix}{line_dict.get('component_num','')}"
                    Xl, parsed_Xl = None, False
                    if val_lower.startswith("j"):
                        try:
                            Xl = float(val_lower[1:]) if val_lower[1:] else 1.0
                            parsed_Xl = True
                        except ValueError:
                            app_logger.warning(f"Could not parse Xl from '{original_value}' (jXl format) for {component_name_debug}. Using original.")
                    elif 'j' in val_lower and val_lower.endswith('j'):
                        try:
                            Xl = float(val_lower[:-1]) if val_lower[:-1] else 1.0
                            parsed_Xl = True
                        except ValueError:
                            app_logger.warning(f"Could not parse Xl from '{original_value}' (Xlj format) for {component_name_debug}. Using original.")
                    if parsed_Xl and Xl is not None:
                        if Xl <= 0 or current_ac_frequency_hz <= 0:
                            app_logger.warning(f"Invalid Xl ({Xl}) or freq ({current_ac_frequency_hz}Hz) for {component_name_debug}. Using original: '{original_value}'")
                        else:
                            inductance = Xl / (2 * np.pi * current_ac_frequency_hz)
                            line_dict['value'] = inductance
                            app_logger.info(f"Processed Inductor {component_name_debug}: original='{original_value}', Xl={Xl}, freq={current_ac_frequency_hz}Hz, L={inductance:.4e}H")
                
                spice_line = analyzer_instance.stringify_line(line_dict)
                if spice_line:
                    spice_body_lines.append(spice_line)

            netlist_content_for_pyspice = "\n".join(spice_body_lines)
            
            if not netlist_content_for_pyspice.strip():
                 st.error("Netlist for AC analysis is effectively empty after processing. Cannot simulate.")
                 return

            net_text_ac = (
                '.title detected_circuit_ac\n'
                + netlist_content_for_pyspice
                + f'\n* Equivalent SPICE command being executed:\n* .ac lin 1 {current_ac_frequency_hz} {current_ac_frequency_hz}\n'
                + '\n.end\n'
            )
            
            with st.expander("🔍 Debug: Complete SPICE Netlist (AC)"):
                st.code(net_text_ac, language="verilog") # Changed language
            
            app_logger.debug("Running AC SPICE analysis with netlist:")
            app_logger.debug(net_text_ac)
            
            parser_ac = SpiceParser(source=net_text_ac)
            circuit_ac = parser_ac.build_circuit()
            app_logger.debug("Circuit elements (AC):")
            for element in circuit_ac.elements:
                app_logger.debug(f"  {element}")

            simulator_ac = circuit_ac.simulator(temperature=27, nominal_temperature=27)
            app_logger.debug(f"Running AC analysis at {current_ac_frequency_hz} Hz...")
            analysis_ac = simulator_ac.ac(
                variation='lin',
                start_frequency=u_Hz(current_ac_frequency_hz),
                stop_frequency=u_Hz(current_ac_frequency_hz),
                number_of_points=1
            )
            
            node_voltages_ac_display = {}
            for node_name_ac, val_waveform_ac in analysis_ac.nodes.items():
                value_to_convert, is_complex = None, False
                if hasattr(val_waveform_ac, 'as_ndarray'):
                    np_array = val_waveform_ac.as_ndarray()
                    if (np_array.dtype == np.complex64 or np_array.dtype == np.complex128) and len(np_array) > 0:
                        value_to_convert, is_complex = np_array[0], True
                if not is_complex and hasattr(val_waveform_ac, '__len__') and len(val_waveform_ac) > 0:
                    value_to_convert = val_waveform_ac[0]
                
                if value_to_convert is not None:
                    complex_voltage = safe_to_complex(value_to_convert) # Imported from src.utils
                    mag, phase = np.abs(complex_voltage), np.angle(complex_voltage, deg=True)
                    node_voltages_ac_display[str(node_name_ac)] = f"{mag:.3f} ∠ {phase:.2f}° V"
                else:
                    node_voltages_ac_display[str(node_name_ac)] = "Error (no data)"

            branch_currents_ac_display = {}
            for branch_name, val_waveform in analysis_ac.branches.items():
                value_to_convert_b, is_complex_b = None, False
                if hasattr(val_waveform, 'as_ndarray'):
                    np_array_b = val_waveform.as_ndarray()
                    if (np_array_b.dtype == np.complex64 or np_array_b.dtype == np.complex128) and len(np_array_b) > 0:
                        value_to_convert_b, is_complex_b = np_array_b[0], True
                if not is_complex_b and hasattr(val_waveform, '__len__') and len(val_waveform) > 0:
                    value_to_convert_b = val_waveform[0]

                if value_to_convert_b is not None:
                    complex_current = safe_to_complex(value_to_convert_b)
                    mag, phase = np.abs(complex_current), np.angle(complex_current, deg=True)
                    branch_currents_ac_display[str(branch_name)] = f"{mag:.3f} ∠ {phase:.2f}° A"
                else:
                    branch_currents_ac_display[str(branch_name)] = "Error (no data)"

            col1_ac, col2_ac = st.columns(2)
            with col1_ac:
                st.markdown("### Node Voltages (AC)")
                st.json(node_voltages_ac_display)
            with col2_ac:
                st.markdown("### Branch Currents (AC)")
                st.json(branch_currents_ac_display)

            st.markdown("### AC Analysis Plots")
            plot_tabs = st.tabs(["Phasor Diagram"])
            with plot_tabs[0]:
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})
                    max_v_mag = 0
                    for node, val_waveform in analysis_ac.nodes.items():
                        if len(val_waveform) > 0 and str(node) != '0':
                            complex_val = safe_to_complex(val_waveform[0])
                            mag, angle = np.abs(complex_val), np.angle(complex_val)
                            max_v_mag = max(max_v_mag, mag)
                            ax1.plot([0, angle], [0, mag], label=f'V({node})', marker='o', linewidth=2)
                    ax1.set_title('Voltage Phasors'); ax1.set_rmax(max_v_mag * 1.2 if max_v_mag > 0 else 1); ax1.grid(True); ax1.legend()
                    
                    max_i_mag = 0
                    for branch, val_waveform in analysis_ac.branches.items():
                        if len(val_waveform) > 0:
                            complex_val = safe_to_complex(val_waveform[0])
                            mag, angle = np.abs(complex_val), np.angle(complex_val)
                            max_i_mag = max(max_i_mag, mag)
                            ax2.plot([0, angle], [0, mag], label=str(branch), marker='o', linewidth=2)
                    ax2.set_title('Current Phasors'); ax2.set_rmax(max_i_mag * 1.2 if max_i_mag > 0 else 1); ax2.grid(True); ax2.legend()
                    
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                except Exception as e_plot:
                    st.error(f"Error generating phasor plots: {str(e_plot)}")
        else:
            st.error("Netlist not available for AC analysis. Please ensure previous steps completed.")

    except Exception as e_ac:
        error_msg_ac = f"❌ AC SPICE Analysis Error: {str(e_ac)}"
        app_logger.error(error_msg_ac, exc_info=True)
        st.error(error_msg_ac)
        st.info("💡 Tip: Check AC source parameters, frequency, and circuit connectivity.")
