import numpy as np
import matplotlib.pyplot as plt
import sys

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# For drawing circuit
import schemdraw
import schemdraw.elements as elm

logger = Logging.setup_logging()

circuit = Circuit('Voltage Divider')


# circuit.V('input', 'in', circuit.gnd, 10@u_V)
# circuit.R(1, 'in', 'out', 9@u_kOhm)
# circuit.R(2, 'out', circuit.gnd, 1@u_kOhm)

circuit.V('input', 1, circuit.gnd, 10@u_V)
circuit.R(1, 1, 2, 2@u_kΩ)
circuit.R(2, 2, circuit.gnd, 1@u_kΩ)

print("The Curcuit/Netlist:\n\n", circuit)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)

print("The simulator:\n\n", simulator)

analysis = simulator.operating_point()

print(analysis)

for node in analysis.nodes.values():
    print('Node {}: {:5.2f} V'.format(str(node), float(node))) # Fixme: format value + unit

# Function to draw the circuit
def draw_circuit(circuit):
    with schemdraw.Drawing() as d:
        for element in circuit.elements:
            if element.name.startswith('R'):
                d += elm.Resistor().label(element.name)
            elif element.name.startswith('C'):
                d += elm.Capacitor().label(element.name)
            elif element.name.startswith('V'):
                d += elm.SourceV().label(element.name)
        d.draw()

draw_circuit(circuit)


exit()