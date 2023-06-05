import ahkab
import matplotlib
from ahkab import circuit, printing, time_functions
import numpy as np
import matplotlib.pyplot as plt

print("We're using ahkab %s" % ahkab.__version__)
print("\n \n")

#This section of the code starts the RV circuit analysys of matrix and ahkab
"""
RV Circuit
Matrix Method & Ahkab Method
"""
#numpy array of the ciruit loops
rv = np.array([[1500, -300, 0, -700],
              [-300, 1000, -200, 0],
              [0, -200, 1150, -300],
              [-700, 0, -300, 1700]])

irv = np.linalg.inv(rv)

vs = np.array([[50], [-30], [0], [0]])

ans = irv.dot(vs)
print("The following values represents the OP simulation values for 'RV Circuit: Matrix Method \n")
print(ans)
print("\n \n")
print("-------------------------------------------------------------------------------------------  \n \n")



mycir3 = ahkab.Circuit('RV Circuit: Ahkab Method')

mycir3.add_vsource('V1', 'n1', mycir3.gnd, dc_value=50)
mycir3.add_resistor('R1', 'n1', 'n3', value=500)
mycir3.add_resistor('R2', 'n3', 'n4', value=700)
mycir3.add_resistor('R3', 'n4', mycir3.gnd, value=300)
mycir3.add_resistor('R4', 'n4', 'n5', value=200)
mycir3.add_resistor('R5', 'n5', 'n2', value=500)
mycir3.add_vsource('V2', 'n2', mycir3.gnd, dc_value=30)
mycir3.add_resistor('R6', 'n3', 'n8', value=200)
mycir3.add_resistor('R7', 'n8', 'n7', value=500)
mycir3.add_resistor('R8', 'n7', 'n4', value=300)
mycir3.add_resistor('R9', 'n7', 'n6', value=500)
mycir3.add_resistor('R10', 'n6', 'n5', value=150)

opa3 = ahkab.new_op()
r4 = ahkab.run(mycir3, opa3)['op']

print("The following table represents the OP simulation values for 'RV Circuit: Ahkab Method \n")
print(r4)
print("\n \n")
print("-------------------------------------------------------------------------------------------  \n \n \n")

#This line ends the the RV circuit analysys of matrix and ahkab



#This section of the code starts the RCLcircuit analysys
"""
RCL Circuit Analyses
OP Simulation, TRAN, AC an Pole Zero Analyses and plots
"""

mycir = circuit.Circuit('RCL Circuit AC and Tran')
mycir2 = ahkab.Circuit('RCL Circuit PZ')

# Circuit labeling and node conections
# Split into two identical circuits to solve compliling errors
mycir.add_resistor('R1', 'n1', 'n3', value=1)
mycir.add_resistor('R2', 'n3', 'n4', value=2)
mycir.add_capacitor('C1', 'n4', mycir.gnd, value=20e-12)
mycir.add_resistor('R4', 'n4', 'n5', value=2)
mycir.add_resistor('R5', 'n5', mycir.gnd, value=5)
mycir.add_resistor('R6', 'n3', 'n8', value=2)
mycir.add_inductor('L1', 'n7', 'n8', value=30e-12)
mycir.add_capacitor('C2', 'n7', 'n4', value=3e-12)
mycir.add_inductor('L2', 'n6', 'n7', value=1e-3)
mycir.add_resistor('R10', 'n6', 'n5', value=2)

mycir2.add_resistor('R1', 'n1', 'n3', value=1)
mycir2.add_resistor('R2', 'n3', 'n4', value=2)
mycir2.add_capacitor('C1', 'n4', mycir.gnd, value=20e-12)
mycir2.add_resistor('R4', 'n4', 'n5', value=2)
mycir2.add_resistor('R5', 'n5', mycir.gnd, value=5)
mycir2.add_resistor('R6', 'n3', 'n8', value=2)
mycir2.add_inductor('L1', 'n7', 'n8', value=30e-12)
mycir2.add_capacitor('C2', 'n7', 'n4', value=3e-12)
mycir2.add_inductor('L2', 'n6', 'n7', value=1e-3)
mycir2.add_resistor('R10', 'n6', 'n5', value=2)

voltage_step = time_functions.pulse(v1=0, v2=1, td=500e-9, tr=1e-12,pw=1, tf=1e-12, per=2)
mycir.add_vsource('V1', 'n1', mycir.gnd, dc_value=1, ac_value=1, function=voltage_step)
mycir2.add_vsource('V1', 'n1', mycir.gnd, dc_value=1, ac_value=1)

opa = ahkab.new_op()
aca = ahkab.new_ac(start=1e3, stop=1e5, points=100)
trana = ahkab.new_tran(tstart=0, tstop=1.2e-3, tstep=1e-6, x0=None)
pza = ahkab.new_pz('V1', ('n5', mycir2.gnd), x0=None, shift=1e3)
r = ahkab.run(mycir, an_list=[opa, aca, trana])
r2 = ahkab.run(mycir2, pza)['pz']
r3 = ahkab.run(mycir, opa)['op']

print("The following table represents the OP simulation values for the RCL Circuit Analyses \n")
print(r3)
print("\n")
print("-------------------------------------------------------------------------------------------  \n \n")


figsize = (5, 5)
plt.figure(figsize=figsize)

# plot o's for zeros and x's for poles
for x, v in r2:
    plt.plot(np.real(v), np.imag(v), 'bo'*(x[0]=='z')+'rx'*(x[0]=='p'))
# set axis limits and print some thin axes
xm = 1e9
plt.xlim([-xm*10., xm*10.])
plt.plot(plt.xlim(), [0,0], 'k', alpha=.5, lw=.5)
plt.plot([0,0], plt.ylim(), 'k', alpha=.5, lw=.5)

# plot the lines from origin to poles or zeros
plt.plot([np.real(r2['p0']), 0], [np.imag(r2['p0']), 0], 'k--', alpha=.5)
plt.text(np.real(r2['p0']), np.imag(r2['p0'])*1.1, '$p_0$', ha='center',
     fontsize=20)
plt.plot([np.real(r2['p1']), 0], [np.imag(r2['p1']), 0], 'k--', alpha=.5)
plt.text(np.real(r2['p1']), np.imag(r2['p1'])*1.1, '$p_1$', ha='center',
     fontsize=20)
plt.plot([np.real(r2['p2']), 0], [np.imag(r2['p2']), 0], 'k--', alpha=.5)
plt.text(np.real(r2['p2']), np.imag(r2['p2'])*1.1, '$p_2$', ha='center',
     fontsize=20)
plt.plot([np.real(r2['p3']), 0], [np.imag(r2['p3']), 0], 'k--', alpha=.5)
plt.text(np.real(r2['p3']), np.imag(r2['p3'])*1.1, '$p_3$', ha='center',
     fontsize=20)

# print the distance between p0 and p1
plt.plot([np.real(r2['p1']), np.real(r2['p0'])],[np.imag(r2['p1']), np.imag(r2['p0'])],'k-', alpha=.5, lw=.5)

# label the singularities from the ciruit
plt.text(np.real(r2['z0']), np.imag(r2['z0']), '$z_0$', ha='center', fontsize=20)
plt.text(np.real(r2['z1']), np.imag(r2['z1']), '$z_1$', ha='center', fontsize=20)
plt.text(.4e6, .4e7, '$z_0$', ha='center', fontsize=20)

# lables y axis
plt.xlabel('Real [Hz]'); plt.ylabel('Imag [Hz]'); plt.title('Singularities');


# Plot the trasient plot of the circuit
fig = plt.figure()
plt.title(mycir.title + " - TRAN Simulation")
plt.plot(r['tran']['T'], r['tran']['VN1'], label="Input voltage")
#plt.hold(True)
plt.plot(r['tran']['T'], r['tran']['VN5'], label="output voltage")
plt.legend()
#plt.hold(False)
plt.grid(True)
plt.ylim([0, 1.2])
plt.ylabel('Step response')
plt.xlabel('Time [s]')
fig.savefig('tran_plot.png')

#plot the AC simulation of the circuit
fig = plt.figure()
plt.subplot(211)
plt.semilogx(r['ac']['f'], np.abs(r['ac']['VN5']), 'o-')
plt.ylabel('abs(V(n5)) [V]')
plt.title(mycir.title + " - AC Simulation")
plt.subplot(212)
plt.grid(True)
plt.semilogx(r['ac']['f'], np.angle(r['ac']['VN5']), 'o-')
plt.xlabel('frequency [rad/s]')
plt.ylabel('arg(V(n5)) [rad]')
fig.savefig('ac_plot.png')
plt.show()
# end of code
