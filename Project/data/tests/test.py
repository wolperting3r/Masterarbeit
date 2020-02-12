from data_generation_Rmax import generate_data as gd_rmax
from data_generation_x_c import generate_data as gd_xc
import os

print('Generating images')
gd_rmax(N_values=1, st_sz=[5, 5], equal_kappa=False)
gd_xc(N_values=30, st_sz=[5, 5], equal_kappa=False)

print('Making animated gif...')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'x_c')
path_gif = os.path.join(path, 'animated.gif')
path_png = os.path.join(path, '*.png')
if os.path.isfile(path_gif):
    os.remove(path_gif)
os.system('convert -delay 1 -loop 0 ' + path_png + ' ' + path_gif)
print('Finished!')
