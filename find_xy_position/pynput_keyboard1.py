# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:44:54 2019

@author: mwdocter

more advanced listener, you can move a selected points udlr with the arrow keys, and press esc when done
"""
from pynput.keyboard import Key, Listener
import matplotlib.pyplot as plt
import time

points_left=[[ii, ii+10] for ii in range(11)]
points_right=points_left.copy()

fig = plt.figure(1)
ax1 = fig.add_subplot(1,2,1)  
#ax1=plt. subplot(1,2,1)
[ax1.plot(xx,yy,markersize=10, c='k', marker='o', fillstyle='none') for xx,yy in points_left]
#ax2=plt. subplot(1,2,2)
ax2 = fig.add_subplot(1,2,2)  
xp=[xx for xx,yy in points_right]
yp=[yy for xx,yy in points_right]
line1,=ax2.plot(xp,yp,markersize=10, c='k', marker='o', fillstyle='none', linestyle='none') 
fig.canvas.draw()
fig.canvas.flush_events()
plt.pause(1)

def on_release(key):
        global points_right
        
        if key == Key.up:
            points_right= [ [xx,yy+1] for xx,yy in points_right]
        elif key == Key.down:
            points_right= [ [xx,yy-1] for xx,yy in points_right]
        elif key == Key.right:
            points_right= [ [xx+1,yy] for xx,yy in points_right]
        elif key == Key.left:
            points_right= [ [xx-1,yy] for xx,yy in points_right]
        
        
        
        if key == Key.esc or (key==Key.up) or (key == Key.down) or (key == Key.right) or (key == Key.left):
            # Stop listener
            return False

xp_new=0*xp
while xp!=xp_new or yp!=yp_new:
    xp=[xx for xx,yy in points_right]
    yp=[yy for xx,yy in points_right]
    # Collect events until released
    with Listener(on_release=on_release) as listener:
        listener.join()

    xp_new=[xx for xx,yy in points_right]
    yp_new=[yy for xx,yy in points_right]
    print(xp_new[0],yp_new[0])

    line1.set_xdata(xp)
    line1.set_ydata(yp)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  