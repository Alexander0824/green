import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

colnames = ['temperature', 'almond', 'canola', 'corn', 'grapeseed', 'hazelnut', 'olive', 'peanut', 'safflower', 'sesame', 'soybean', 'sunflower', 'walnut']
n = 30

df = pd.read_csv('data', names=colnames, nrows=n)

temp_list = df.temperature.tolist()

for j in range(len(temp_list)):
    #  converting temperature list from C to K
    temp_list[j] = temp_list[j] + 298

# note: units of heat capacity are kJ/kg.K [J/g.K]
canola_list = df.canola.tolist()
soybean_list = df.soybean.tolist()
sunflower_list = df.sunflower.tolist()

MM_canola = 876.6  # g/mol
MM_soybean = 920
MM_sunflower = 876.16
R = 8.3145  # J/mol.K

#  kJ/kg.K --> J/mol.K
for l in range(len(canola_list)):
    canola_list[l] = canola_list[l] * MM_canola
    soybean_list[l] = soybean_list[l] * MM_soybean
    sunflower_list[l] = sunflower_list[l] * MM_sunflower

canola_h = np.zeros(len(temp_list))
soybean_h = np.zeros(len(temp_list))
sunflower_h = np.zeros(len(temp_list))

#  calculation of a_6 term (based on Rob's simulations)

eV_c = -53.910040 / 4
eV_o = -9.311737 / 2
eV_h = -6.769542 / 2

# ∆H_f = E[oil] - n_C*E[C] - n_H*E[H] - n_O*E[O]

eV_canola = -925.939389 - 57 * eV_c - 100 * eV_h - 6 * eV_o
eV_soybean = -917.524825 - 57 * eV_c - 98 * eV_h - 6 * eV_o
eV_sunflower = -926.635501 - 57 * eV_c - 100 * eV_h - 6 * eV_o

h_canola = eV_canola * 96.487 * 1000  # J/mol
h_canola_1 = eV_canola * 96.487
h_soybean = eV_soybean * 96.487 * 1000
h_sunflower = eV_sunflower * 96.487 * 1000

#  calculation of a_7 term (based on Gibbs free energy)

g_canola = -264.1e3  # J/mol.K
g_soybean = -240.7e3
g_sunflower = -219.6e3

#  calculation of NASA Polynomial
z_canola = np.polyfit(temp_list, canola_list, 4)  # fourth order polynomial (5 coefficients)
p_canola = np.poly1d(z_canola)  # polynomial using z coefficients

z_soybean = np.polyfit(temp_list, soybean_list, 4)  # fourth order polynomial (5 coefficients)
p_soybean = np.poly1d(z_soybean)  # polynomial using z coefficients

z_sunflower = np.polyfit(temp_list, sunflower_list, 4)  # fourth order polynomial (5 coefficients)
p_sunflower = np.poly1d(z_sunflower)  # polynomial using z coefficients

# NASA coefficients for canola oil - based on enthalpy
T_ref = 298.15
a_5_c = z_canola[0]
a_4_c = z_canola[1]
a_3_c = z_canola[2]
a_2_c = z_canola[3]
a_1_c = z_canola[4]

# NASA coefficients for soybean oil - based on enthalpy
a_5_soy = z_soybean[0]
a_4_soy = z_soybean[1]
a_3_soy = z_soybean[2]
a_2_soy = z_soybean[3]
a_1_soy = z_soybean[4]

# NASA coefficients for sunflower oil - based on enthalpy
a_5_sun = z_sunflower[0]
a_4_sun = z_sunflower[1]
a_3_sun = z_sunflower[2]
a_2_sun = z_sunflower[3]
a_1_sun = z_sunflower[4]

def poly_c(T, a_1, a_2, a_3, a_4, a_5, a_6, a_7):
    return a_1 + a_2 * T + a_3 * T ** 2 + a_4 * T ** 3 + a_5 * T ** 4

def a6_finder(T, a_1, a_2, a_3, a_4, a_5, h):
    # h is defined at temperature T
    return 1/R * (h - R * (a_1 * T + (a_2 / 2) * T ** 2 + (a_3 / 3) * T ** 3 + (a_4 / 4) * T ** 4 + (a_5 / 5) * T ** 5))

a_6_c = a6_finder(0, a_1_c, a_2_c, a_3_c, a_4_c, a_5_c, h_canola)
a_6_soy = a6_finder(0, a_1_soy, a_2_soy, a_3_soy, a_4_soy, a_5_soy, h_soybean)
a_6_sun = a6_finder(0, a_1_sun, a_2_sun, a_3_sun, a_4_sun, a_5_sun, h_sunflower)

def poly_h(T, a_1, a_2, a_3, a_4, a_5, a_6, a_7):
    return R * (a_1 * T + (a_2 / 2) * T ** 2 + (a_3 / 3) * T ** 3 + (a_4 / 4) * T ** 4 + (a_5 / 5) * T ** 5 + a_6)

def poly_s(T, a_1, a_2, a_3, a_4, a_5, a_6, a_7):
    return R * (a_1 * np.log(T) + a_2 * T + (a_3 / 2) * T ** 2 + (a_4 / 3) * T ** 3 + (a_5 / 4) * T ** 4 + a_7)

def poly_g(T, a_1, a_2, a_3, a_4, a_5, a_6, a_7):
    return R * (a_1 * T + (a_2 / 2) * T ** 2 + (a_3 / 3) * T ** 3 + (a_4 / 4) * T ** 4 + (a_5 / 5) * T ** 5 + a_6 / T) - \
        T * R * (a_1 * np.log(T) + a_2 * T + (a_3 / 2) * T ** 2 + (a_4 / 3) * T ** 3 + (a_5 / 4) * T ** 4 + a_7)

def a7_finder(T, a_1, a_2, a_3, a_4, a_5, a_6, g):
    # g is defined at temperature T
    return 1/(T*R) * (R * (a_1 * T + (a_2 / 2) * T ** 2 + (a_3 / 3) * T ** 3 + (a_4 / 4) * T ** 4 + (a_5 / 5) * T ** 5 + a_6 / T) - \
        T * R * (a_1 * np.log(T) + a_2 * T + (a_3 / 2) * T ** 2 + (a_4 / 3) * T ** 3 + (a_5 / 4) * T ** 4) - g)

a_7_c = a7_finder(298, a_1_c, a_2_c, a_3_c, a_4_c, a_5_c, a_6_c, g_canola)
a_7_soy = a7_finder(298, a_1_soy, a_2_soy, a_3_soy, a_4_soy, a_5_soy, a_6_soy, g_soybean)
a_7_sun = a7_finder(298, a_1_sun, a_2_sun, a_3_sun, a_4_sun, a_5_sun, a_6_sun, g_sunflower)

print('NASA coefficients for canola oil are', a_1_c, a_2_c, a_3_c, a_4_c, a_5_c, a_6_c, a_7_c)
print('NASA coefficients for soybean oil are', a_1_soy, a_2_soy, a_3_soy, a_4_soy, a_5_soy, a_6_soy, a_7_soy)
print('NASA coefficients for sunflower oil are', a_1_sun, a_2_sun, a_3_sun, a_4_sun, a_5_sun, a_6_sun, a_7_sun)

xp = np.linspace(200, 800, 100)

plt.ylabel('Specific heat (kJ/kg K)')
plt.xlabel('Temperature (K)')
plt.plot(xp, poly_c(xp, a_1_soy, a_2_soy, a_3_soy, a_4_soy, a_5_soy, a_6_soy, a_7_soy), '-', label='Soybean')
plt.plot(xp, poly_c(xp, a_1_c, a_2_c, a_3_c, a_4_c, a_5_c, a_6_c, a_7_c), '-', label='Canola')
plt.plot(xp, poly_c(xp, a_1_sun, a_2_sun, a_3_sun, a_4_sun, a_5_sun, a_6_sun, a_7_sun), '-', label='Sunflower')
plt.plot(temp_list, canola_list, label='Canola Data')
plt.plot(temp_list, soybean_list, label='Soybean Data')
plt.plot(temp_list, sunflower_list, label='Sunflower Data')
plt.legend(loc='best')
plt.show()

plt.ylabel('Enthalpy (kJ/kmol)')
plt.xlabel('Temperature (K)')
plt.plot(xp, poly_h(xp, a_1_soy, a_2_soy, a_3_soy, a_4_soy, a_5_soy, a_6_soy, a_7_soy), '-', label='Soybean')
plt.plot(xp, poly_h(xp, a_1_c, a_2_c, a_3_c, a_4_c, a_5_c, a_6_c, a_7_c), '-', label='Canola')
plt.plot(xp, poly_h(xp, a_1_sun, a_2_sun, a_3_sun, a_4_sun, a_5_sun, a_6_sun, a_7_sun), '-', label='Sunflower')
plt.legend(loc='upper left')
plt.show()

plt.ylabel('Entropy (kJ/kmol.K)')
plt.xlabel('Temperature (K)')
plt.plot(xp, poly_s(xp, a_1_soy, a_2_soy, a_3_soy, a_4_soy, a_5_soy, a_6_soy, a_7_soy), '-', label='Soybean')
plt.plot(xp, poly_s(xp, a_1_c, a_2_c, a_3_c, a_4_c, a_5_c, a_6_c, a_7_c), '-', label='Canola')
plt.plot(xp, poly_s(xp, a_1_sun, a_2_sun, a_3_sun, a_4_sun, a_5_sun, a_6_sun, a_7_sun), '-', label='Sunflower')
plt.legend(loc='upper left')
plt.show()
#
plt.ylabel('Gibbs Free Energy (kJ/kmol.K)')
plt.xlabel('Temperature (K)')
plt.plot(xp, poly_g(xp, a_1_soy, a_2_soy, a_3_soy, a_4_soy, a_5_soy, a_6_soy, a_7_soy), '-', label='Soybean')
plt.plot(xp, poly_g(xp, a_1_c, a_2_c, a_3_c, a_4_c, a_5_c, a_6_c, a_7_c), '-', label='Canola')
plt.plot(xp, poly_g(xp, a_1_sun, a_2_sun, a_3_sun, a_4_sun, a_5_sun, a_6_sun, a_7_sun), '-', label='Sunflower')
plt.legend(loc='lower left')
plt.show()