import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sys

# ------------------------------------------------------------------------------
# Constants

oxygen_mm = 15.999
nitrogen_mm = 14.007
carbon_mm = 12.011
hydrogen_mm = 1.008
R = 8.3145

# ------------------------------------------------------------------------------
# Importing NASA Polynoimals

# Products
prodcol = ['Names','a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7']
prodn = 17
df = pd.read_csv('data_polynomial_1', names=prodcol, nrows=prodn)
product_name = df.Names.tolist()  # name list as reference
my_data = np.genfromtxt('data_polynomial_1', delimiter=',', names=True)

# -----------------------------------------------------------------------------
# Molar Mass constants:

MM_canola = 881.4  # [g/mol]
MM_soybean = 879.4
MM_sunflower = 881.4
MM_hydrogen = 2 * 1.008
MM_oxygen = 2 * 15.999
MM_water = 2 * 1.008 + 15.999
MM_carbon = 12.0107
MM_CO = 28.01
MM_CO2 = 44.01
MM_CH4 = MM_carbon + 4 * MM_hydrogen
MM_C3H8 = 3 * MM_carbon + 8 * MM_hydrogen
MM_C4H8 = 4 * MM_carbon + 8 * MM_hydrogen
MM_C4H10 = 4 * MM_carbon + 10 * MM_hydrogen
MM_C5H12 = 5 * MM_carbon + 12 * MM_hydrogen
MM_C7H16 = 7 * MM_carbon + 16 * MM_hydrogen
MM_C8H18 = 8 * MM_carbon + 18 * MM_hydrogen
MM_C9H19 = 9 * MM_carbon + 19 * MM_hydrogen
MM_C10H21 = 10 * MM_carbon + 21 * MM_hydrogen
MM_C12H10 = 12 * MM_carbon + 10 * MM_hydrogen

# ------------------------------------------------------------------------------
# Thermodynamics section:
# Evaluation of NASA Polynomials

c_k = np.zeros(prodn) # no need for 'np.array'
h_k = np.zeros(prodn)
s_k = np.zeros(prodn)
g_k = np.zeros(prodn)

# solve heat capacity equation
def nasa_equation_c0(a, T):
    for i in range(prodn):
        c_k[i] = R * (a[i][1] + a[i][2] * T + a[i][3] * (T ** 2) + a[i][4] * (T ** 3) + a[i][5] * (T ** 4))
    return c_k

# solve enthalpy equation
def nasa_equation_h0(a, T):
    for i in range(prodn):
        h_k[i] = R * T * (a[i][0] + a[i][1] * (T / 2) + a[i][2] * (T ** 2) / 3 + a[i][3] * (T ** 3) / 4 + a[i][4] * (T ** 4) / 5 + (a[i][5] / T))
    return h_k

# solve entropy equations
def nasa_equation_s0(a, T):
    for i in range(prodn):
        s_k[i] = R * (a[i][0] * np.log(T) + a[i][1] * T + a[i][2] * (T ** 2) / 2 + a[i][3] * (T ** 3) / 3 + a[i][4] * (T ** 4) / 4 + a[i][6])
    return s_k

# solve gibbs free energy equations
def nasa_equation_g0(h_k, s_k, T):
    return h_k - T * s_k

h_k = np.zeros(prodn)
s_k = np.zeros(prodn)
g_k = np.zeros(prodn)

# ------------------------------------------------------------------------------
# Select Variables
# Oil type?
canola_oil = 0
soybean_oil = 1
sunflower_oil = 2
oil = sunflower_oil # <-- Choose which oil type

#  coefficients derived based on 'polynomials.py'
canola_coef = [6563.548500464801, -42.77290483530693, 0.13306620612062095, -0.0001600501861677549, 6.26936437885385e-08, 100, -25991.07764471329]
soybean_coef = [18761.142240552716, -172.1026308518384, 0.6278840533138913, -0.0009954537950874374, 5.876984146879762e-07, 100, -69711.96306328745]
sunflower_coef = [-1962.856179185774, 47.19617002965326, -0.21779742427724272, 0.00044124565883593226, -3.195571607031591e-07, 100, 4652.592711592474]

a = [[-1962.856179185774, 47.19617002965326, -0.21779742427724272, 0.00044124565883593226, -3.195571607031591e-07, 100, 4652.592711592474],#change this line
[3.33727920E+00, -4.94024731E-05, 4.99456778E-07, -1.79566394E-10, 2.00255376E-14, -9.50158922E+02, -3.20502331E+00],
[3.28253784E+00, 1.48308754E-03, -7.57966669E-07, 2.09470555E-10, -2.16717794E-14, -1.08845772E+03, 5.45323129E+00],
[3.03399249E+00, 2.17691804E-03, -1.64072518E-07, -9.70419870E-11, 1.68200992E-14, -3.00042971E+04, 4.96677010E+00],
[2.49266888E+00, 4.79889284E-05, -7.24335020E-08, 3.74291029E-11, -4.87277893E-15, 8.54512953E+04, 4.80150373E+00],
[2.71518561E+00, 2.06252743E-03, -9.98825771E-07, 2.30053008E-10, -2.03647716E-14, -1.41518724E+04, 7.81868772E+00],
[3.85746029E+00, 4.41437026E-03, -2.21481404E-06, 5.23490188E-10, -4.72084164E-14, -4.87591660E+04, 2.27163806E+00],
[7.48514950E-02, 1.33909467E-02, -5.73285809E-06, 1.22292535E-09, -1.01815230E-13, -9.46834459E+03, 1.84373180E+01],
[6.66789363E+00, 2.06120214E-02, -7.36553027E-06, 1.18440761E-09, -7.06963210E-14, -1.62748521E+04, -1.31859503E+01],
[8.02147991E+00, 2.26010707E-02, -8.31284033E-06, 1.37803072E-09, -8.42175469E-14, -4.30852153E+03, -17.11706975],
[9.44535834E+00, 2.57858073E-02, -9.23619122E-06, 1.48632755E-09, -8.87897158E-14, -20138.21655, -2.63470076E+01],
[1.35469980E+01, 2.84217860E-02, -9.41746480E-06, 1.38935890E-09, -7.42126090E-14, -2.45776800E+04, -4.70211850E+01],
[1.85354704E+01, 3.91420468E-02, -1.38030268E-05, 2.22403874E-09, -1.33452580E-13, -3.19500783E+04, -7.01902840E+01],
[2.21755407E+01, 4.24428161E-02, -1.49161103E-05, 2.40376673E-09, -1.44359037E-13, -3.61030944E+04, -88.0854457],
[1.91952670E+01, 5.54392490E-02, -2.14368010E-05, 3.78851440E-09, -2.80029870E-13, -1.43737110E+04,-8.60562950E+01],
[2.13221280E+01, 6.15735240E-02, -2.38494830E-05, 4.22091160E-09, -2.78893070E-13, -1.79678090E+04, -7.56437890E+01],
[2.28964892E+01, 3.68452570E-02, -1.35016270E-05, 2.20802808E-09, -1.33358223E-13, 1.07394499E+04, -1.00510148E+02]]

if oil == canola_oil:
    MM_oil = MM_canola
    oil_coef = canola_coef
    oil_c = 57
    oil_h = 100
    oil_o = 6
elif oil == soybean_oil:
    MM_oil = MM_soybean
    oil_coef = soybean_coef
    oil_c = 57
    oil_h = 98
    oil_o = 6
else:
    MM_oil = MM_sunflower
    oil_coef = sunflower_coef
    oil_c = 57
    oil_h = 100
    oil_o = 6

# Reaction temperature?
T_set = 600

# ------------------------------------------------------------------------------
# Customization section:
computing_time = 20 # second(s)
i = 0
alpha = 200 # larger number will restrict
# Initial temperature
initial_temperature = 1000
# Fractional reduction every cycle
def cooling(n):
    return 1 /(1+alpha*math.log(1+n))

def limit_cooling(n):
    return 0.2 #- 1/(20 * n) + 0.5
# ------------------------------------------------------------------------------
# GUESS
# Set the number of variables and their boundaries
number_variables = 17
lower_bounds = [0.00, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
upper_bounds = [0.30, 4.0, 1.2, 1.5, 5.0, 2.5, 2.0, 1.0, 1.5, 4.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
mole_weight = [MM_oil, MM_hydrogen, MM_oxygen, MM_water, MM_carbon, MM_CO, MM_CO2, MM_CH4, MM_C3H8, MM_C4H8, \
               MM_C4H10, MM_C5H12, MM_C7H16,MM_C8H18, MM_C9H19, MM_C10H21, MM_C12H10]

# ------------------------------------------------------------------------------
# a single objective function relating mole and gibbs energy required
H_k = np.zeros(prodn)
S_k = np.zeros(prodn)
G_k = np.zeros(prodn)

nasa_equation_h0(a, T_set)
nasa_equation_s0(a, T_set)

def objective_function(mole):
    for j in range(len(mole)):
        H_k[j] = h_k[j] * mole[j] # enthalpy function
        if mole[j] == 0:
            S_k[j] = 0
        else:
            S_k[j] = mole[j] * (s_k[j] - R * np.log(mole[j] / sum(mole[:])))
    G_k = H_k - S_k * T_set
    return sum(G_k[:])

# ------------------------------------------------------------------------------
# Simulated Annealing Algorithm:
initial_solution = np.zeros(prodn)

for v in range(prodn):
    initial_solution[v] = lower_bounds[v] #random.uniform(lower_bounds[v], upper_bounds[v])

def positive_negative():
    return 1 if random.random() < 0.5 else -1

current_solution = initial_solution
best_solution = initial_solution

n = 1  # no of solutions accepted
best_fitness = objective_function(best_solution)
current_temperature = initial_temperature  # current temperature
start = time.time()
no_attempts = 100  # number of attempts in each level of temperature
record_best_fitness = []
record_best_solution = []

i = 0
j = 0

mole_reactant = [1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mass_reactant = [mole_reactant[0] * mole_weight[0], mole_reactant[1] * mole_weight[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mass_total_r = sum(mass_reactant)
mass_product = np.zeros(17)
mass = np.zeros(17)

def find_mass(weight, mole):
    for i in range(len(mole)):
        mass[i] = weight[i] * mole[i]
        mass_total = sum(mass[:])
    return mass_total

reac_carbon = oil_c * mole_reactant[0]
reac_hydrogen = oil_h * mole_reactant[0] + 2 * mole_reactant[1]
reac_oxygen = oil_o * mole_reactant[0]
reac_species = [reac_carbon, reac_hydrogen, reac_oxygen]

product_count = [0, 0, 0]
def number_species(mole): # products that have C, H and O
    product_count[0] = oil_c * mole[0] + mole[4] + mole[5] + mole[6] + mole[7] + 3 * mole[8] + 4 * mole[9] \
                    + 4 * mole[10] + 5 * mole[11] + 7 * mole[12] + 8 * mole[13] + 9 * mole[14] + 10 * mole[15] + 12 * mole[16] # carbon
    product_count[1] = oil_h * mole[0] + 2 * mole[1] + 2 * mole[3] + 4 * mole[7] + 8 * mole[8] + 8 * mole[9] + 10 * mole[10]\
                    + 12 * mole[11] + 16 * mole[12] + 18 * mole[13] + 19 * mole[14] + 21 * mole[15] + 10 * mole[16] # hydrogen
    product_count[2] = oil_o * mole[0] + 2 * mole[2] + mole[3] + mole[5] + 2 * mole[6] # oxygen
    return product_count

# certainty range
q = 0.9
w = 1.1

record_solutions = math.pi * np.ones((50, 17))

for i in range(100001): # arbitrarily large number
    for j in range(no_attempts): # number of attempts in each temperature level
        for k in range(prodn): # 17 variables

            current_solution[k] = best_solution[k] + 0.1 * positive_negative() * (random.uniform(lower_bounds[k], upper_bounds[k]))
            current_solution[k] = max(min(current_solution[k], upper_bounds[k]), lower_bounds[k])  # repair the solution respecting the bounds

            current_fitness = objective_function(current_solution)
            E = abs(current_fitness - best_fitness)
            if i == 0 and j == 0:
                EA = E

        if mass_total_r * q < find_mass(mole_weight, current_solution) < mass_total_r * w \
            and reac_carbon * q < number_species(current_solution)[0] < reac_carbon * w \
            and reac_hydrogen * q < number_species(current_solution)[1] < reac_hydrogen * w \
            and reac_oxygen * q < number_species(current_solution)[2] < reac_oxygen * w:

            # making sure the current_solution is within the set bounds

            p = math.exp(-E / (EA * current_temperature))

            if current_fitness < best_fitness:  # mass and mole conservation
                if random.random() < p:
                    best_solution = current_solution  # update the best solution
                    best_fitness = objective_function(best_solution)
                    record_solutions[n][:] = best_solution
                    n = n + 1  # count the solutions accepted
                    EA = (EA * (n - 1) + E) / n  # update EA
                    print(best_solution)

                    for k in range(number_variables):
                        if best_solution[k] == upper_bounds[k] and lower_bounds[k] != upper_bounds[k]:
                            lower_bounds[k] = upper_bounds[k] * (1 + limit_cooling(i))
                            upper_bounds[k] = upper_bounds[k] * (1 - limit_cooling(i))  # increases the upper bound by x
                        elif best_solution[k] == lower_bounds[k] and best_solution[k] != 0 and lower_bounds[k] != upper_bounds[k]:
                            upper_bounds[k] = lower_bounds[k] * (1 + limit_cooling(i))
                            lower_bounds[k] = lower_bounds[k] * (1 - limit_cooling(i)) # decreases the lower bound by x

    # Cooling the temperature
    current_temperature = initial_temperature * cooling(i)
    # Stop by computing time
    end = time.time()
    if end - start >= computing_time:
        print('Time end')
        break
    elif current_temperature < 0.0001:
        print('Temperature end')
        break


# Dynamic array
def clean_data(x):
    for i in range(50):
        for j in range(17):
            if x[i][j] == math.pi:
                x[i][j] = x[i - 1][j]
            else:
                pass

clean_data(record_solutions)
best_solution = record_solutions[-1][:]

def write_csv(data):
    with open('sunflower_600', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

if best_solution[0] == math.pi:
    print('No solution found')
    solution_found = 0
    sys.exit()
else:
    write_csv(best_solution)
    print('Solution found')
    solution_found = 1