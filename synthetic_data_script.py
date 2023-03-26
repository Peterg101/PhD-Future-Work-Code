import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt
complex_number = np.complex(0,1)
original_real_perm_ramp_up = pd.read_excel('cb_10_original.xlsx', sheet_name='real_perm_ramp_up', engine = 'openpyxl').set_index('Frequency').to_numpy()
original_real_perm_ramp_down = pd.read_excel('cb_10_original.xlsx', sheet_name='real_perm_ramp_down', engine = 'openpyxl').set_index('Frequency').to_numpy()
original_imag_perm_ramp_up = pd.read_excel('cb_10_original.xlsx', sheet_name='imag_perm_ramp_up', engine = 'openpyxl').set_index('Frequency').to_numpy()
original_imag_perm_ramp_down = pd.read_excel('cb_10_original.xlsx', sheet_name='imag_perm_ramp_down', engine = 'openpyxl').set_index('Frequency').to_numpy()
ramp_up_comp_perm = (original_real_perm_ramp_up - (complex_number * original_imag_perm_ramp_up))
ramp_down_comp_perm = (original_real_perm_ramp_down - (complex_number * original_imag_perm_ramp_down))
frequencies = pd.read_excel('cb_10_original.xlsx', sheet_name='real_perm_ramp_up', engine = 'openpyxl')['Frequency']
#print(frequencies)
#print(original_real_perm_ramp_up)
#print(ramp_down_comp_perm)
multiplier = 0.5
freq = 10.00
list_1 = []
for i in range(0, 101):
    constant = multiplier + (0.01*i)
    new_array = ramp_up_comp_perm*constant
    df = pd.DataFrame(new_array, columns=['RT','HT_22', 'HT_100', 'HT_200', 'HT_300', 'HT_400', 'HT_500', 'HT_600', 'HT_700', 'HT_800', 'HT_900', 'HT_950'])
    df = pd.concat([df, frequencies], axis = 1)
    filtered = df[df['Frequency'] == freq]['RT'].to_list()[0]
    list_1.append(filtered)
array_1 = np.array(list_1)

new_array = np.stack((array_1.real, array_1.imag), axis = 1)
new_array_df = pd.DataFrame(new_array)
new_array_df.to_csv('separated_comp_vals.csv')
shuffler = np.random.permutation(len(array_1))
shuffler_2 = np.random.permutation(len(array_1))
shuffler_3 = np.random.permutation(len(array_1))

combinations = np.array(np.meshgrid(array_1, array_1, array_1))
my_data = np.genfromtxt('second_df.csv', delimiter=',', dtype = str)
complex_data = np.array([6.605033914285714-3.8842239857142857j, 
 7.472361600000001-4.394273600000001j,
 8.206254257142858-4.825854042857143j,
 4.937096057142857-2.903359342857143j, 
 4.536790971428571-2.6679518285714283j,
 6.138011314285714-3.609581885714286j,
 3.536028257142857-2.079433042857143j,
 7.138774028571429-4.198100671428572j,
 6.004576285714286-3.5311127142857144j,
 4.736943514285714-2.7856555857142857j,
 5.070531085714285-2.9818285142857146j,
 9.340451999999999-5.4928420000000004j,
 8.673276857142858-5.100496142857144j, 7.005339-4.119631500000001j,
 8.4064068-4.943557800000001j, 9.140299457142858-5.375138242857144j,
 8.873429400000001-5.218199900000001j,
 3.8696158285714284-2.2756059714285715j,
 3.7361808000000005-2.1971368000000004j,
 5.270683628571429-3.0995322714285716j,
 7.605796628571429-4.472742771428572j, 7.939384199999999-4.6689157j,
 6.6717514285714286-3.9234585714285717j,
 7.539079114285713-4.4335081857142855j,
 5.470836171428572-3.217236028571429j,
 9.540604542857144-5.610545757142858j,
 4.603508485714285-2.7071864142857143j,
 8.606559342857143-5.061261557142857j,
 8.072819228571428-4.747384871428571j,
 7.405644085714285-4.3550390142857145j,
 5.2039661142857145-3.060297685714286j,
 4.6702259999999995-2.7464210000000002j,
 8.739994371428573-5.139730728571429j,
 3.3358757142857143-1.9617292857142858j,
 5.6709887142857145-3.3349397857142864j,
 7.272209057142856-4.276569842857143j, 5.1372486-3.0210631j,
 3.4025932285714284-2.0009638714285716j,
 4.269920914285715-2.511013485714286j,
 6.404881371428571-3.7665202285714288j,
 3.8028983142857147-2.236371385714286j,
 5.804423742857143-3.4134089571428574j,
 5.003813571428571-2.9425939285714287j,
 5.404118657142858-3.1780014428571435j,
 6.738468942857143-3.9626931571428576j,
 5.337401142857143-3.1387668571428575j,
 9.407169514285716-5.532076585714287j,
 9.94090962857143-5.845953271428572j,
 6.204728828571428-3.6488164714285714j,
 5.871141257142857-3.452643542857143j,
 9.073581942857142-5.335903657142857j,
 6.5383164-3.8449894000000002j,
 4.336638428571429-2.550248071428572j,
 6.271446342857143-3.6880510571428573j,
 7.672514142857143-4.511977357142857j,
 5.537553685714286-3.256470614285715j,
 6.871903971428572-4.041162328571429j,
 5.937858771428571-3.491878128571429j,
 4.470073457142857-2.6287172428571433j,
 3.469310742857143-2.0401984571428575j,
 4.2032034-2.4717789000000003j, 8.53984182857143-5.022026971428572j,
 3.9363333428571425-2.314840557142857j,
 8.940146914285714-5.257434485714286j,
 7.805949171428571-4.590446528571428j,
 6.471598885714285-3.8057548142857143j,
 7.072056514285714-4.158866085714286j,
 8.139536742857143-4.786619457142858j,
 6.805186457142857-4.001927742857143j,
 9.00686442857143-5.296669071428572j, 5.6042712-3.2957052000000004j,
 6.0712938-3.5703473000000003j,
 3.6027457714285718-2.118667628571429j,
 5.737706228571429-3.3741743714285715j,
 4.069768371428571-2.393309728571429j,
 8.27297177142857-4.865088628571429j,
 9.874192114285714-5.806718685714286j,
 3.669463285714286-2.1579022142857145j,
 4.003050857142857-2.354075142857143j,
 8.806711885714286-5.178965314285715j,
 8.473124314285714-4.982792385714286j,
 7.2054915428571435-4.237335257142858j,
 8.006101714285716-4.708150285714287j,
 6.938621485714286-4.080396914285715j,
 7.872666685714287-4.629681114285715j,
 9.273734485714286-5.453607414285715j,
 7.338926571428572-4.315804428571429j,
 9.607322057142857-5.649780342857143j,
 9.740757085714286-5.728249514285714j, 9.807474599999999-5.7674841j,
 6.338163857142857-3.727285642857143j,
 4.870378542857143-2.864124757142857j,
 9.20701697142857-5.4143728285714285j,
 10.007627142857142-5.885187857142857j,
 9.674039571428573-5.68901492857143j,
 7.739231657142858-4.551211942857144j,
 9.473887028571427-5.5713111714285715j,
 4.136485885714285-2.4325443142857144j,
 4.803661028571429-2.8248901714285717j,
 8.339689285714286-4.904323214285714j,
 4.403355942857143-2.5894826571428573j])
new_df = pd.DataFrame(complex_data)
new_df.to_csv('funky_dataset.csv')


complex_array = np.stack((complex_data.real, complex_data.imag), axis = 1)
complex_array_df = pd.DataFrame(new_array)
complex_array_df.to_csv('complex_separated_comp_vals.csv')

#print(combinations)
#print(combinations.shape)

#print(array_1)
#print(array_1.shape)


# only for example, use your grid
z = np.linspace(0, 100, 101)
x = np.linspace(0, 100, 101)
y = np.linspace(0, 100, 101)
#print(z.shape)




#print(z[shuffler])

X, Y, Z = np.meshgrid(x, y, z)
#X1, Y1, Z1 = np.meshgrid(array_1, array_1, array_1)
X1, Y1, Z1 = np.meshgrid(complex_data, complex_data, complex_data )




#Calculations

j = complex(0,1)
e1 = X1
u1= 1-(1*j)
#material_2
e2 = Y1
u2= 1-(1*j)
#material_3 = data[11]   
e3 = Z1
u3= 1-(1*j)

#constants
initial_theta = 45
c = 3e10
f = 8.2e9
#print(f)
w = 2*np.pi*f
wl = c/f
#print("wl")
#print(wl)
d1=0.3
d2 = 0.3
d3 = 0.3
zm1 = np.sqrt(u1/e1)
zm2= np.sqrt(u2/e2)
zm3= np.sqrt(u3/e3)

pi = np.pi
#R_air
#Nlayer = np.sqrt(self.layer_e/self.layer_u)

u0 = 1.25663706e-06
e0 = 8.85418782e-12
z0 = np.sqrt(u0/e0)

z_in_1 = zm1*np.tanh((2*np.pi*np.sqrt(e1*u1)*d1*1/wl))
#print(z_in_1)
RL_1 = 20*np.log(abs((z_in_1-1)/(z_in_1+1)))
z_in_2 = zm2 * ((z_in_1 +(zm2*np.tanh((2*np.pi*np.sqrt(e2*u2)*d2)/wl)/(zm2+(z_in_1*np.tanh((2*np.pi*np.sqrt(e2*u2)*d2)/wl))))))
#print(z_in_2)
RL_2 = 20*np.log(abs((z_in_2-1)/(z_in_2+1)))

z_in_3 = zm3 * ((z_in_2 +(zm3*np.tanh((2*np.pi*np.sqrt(e3*u3)*d3)/wl)/(zm3+(z_in_2*np.tanh((2*np.pi*np.sqrt(e3*u3)*d3)/wl))))))                               
#print(z_in_3)

RL_ = 20*np.log(abs(((z_in_3)-1)/((z_in_3)+1)))
calc_perc = 0.002
additional_nums = random.randint(-20, 20)
percentage_reversal = round((calc_perc/100)*1030301)

new= RL_.flatten()
new_1 = np.sort(new)
first = 180

print('new db')
print(new_1[percentage_reversal+additional_nums])
print(percentage_reversal)
percentage = (first/1030301)*100
print(percentage)
#print('peter')

#print(X)




"""
# Creating figure
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel('Layer 1 (Bottom)') 
ax.set_ylabel('Layer 2 (Middle)')
ax.set_zlabel('Layer 3 (Top)')

# Creating plot
img = ax.scatter3D(X, Y, Z, c=RL_, alpha=1,  marker='.')

fig.colorbar(img)
#plt.show()

"""