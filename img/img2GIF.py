import os
import imageio

gif_name = 'robot_freezing_2'

start = 19
end = 73
frames = [ imageio.v2.imread(f'./img/img_{t}.png') for t in range(start,end+1) ]

imageio.mimsave(f'./{gif_name}.gif' , 
                frames,          
                fps = 5)        

for t in range(start,end + 1):
    os.remove(f'./img/img_{t}.png') 
