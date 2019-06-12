# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:44:54 2019

@author: mwdocter

simple listener, in order to give keyboard commands and respond to it
"""

from pynput.keyboard import Key, Listener

def on_press(key):
    print('{0} pressed'.format(
        key))
    return key

def on_release(key):
    print('{0} release'.format(
        key))
    return False
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
    