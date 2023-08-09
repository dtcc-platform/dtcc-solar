#!/usr/bin/env python3


'''

import trimesh
import glooey
import pyglet

class WesnothLabel(glooey.Label):
    custom_font_name = 'Lato Regular'
    custom_font_size = 10
    custom_color = '#b9ad86'
    custom_alignment = 'center'

class WesnothButton(glooey.Button):
    Foreground = WesnothLabel

    class Base(glooey.Image):
        custom_image = pyglet.resource.image('base.png')

    class Over(glooey.Image):
        custom_image = pyglet.resource.image('over.png')

    class Down(glooey.Image):
        custom_image = pyglet.resource.image('down.png')


default_path = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceL.stl'
mesh = trimesh.load_mesh(default_path)

scene = trimesh.Scene()
scene.camera.z_far = 10000
scene.add_geometry(mesh)       
scene.show()

window = pyglet.window.Window(1600, 1000)
gui = glooey.Gui(window)

button1 = WesnothButton("Click here!")
button1.push_handlers(on_click=lambda w: print(f"{w} clicked!"))

button2 = WesnothButton("Click here!")
button2.push_handlers(on_click=lambda w: print(f"{w} clicked!"))

vbox = glooey.VBox()

vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(button1)
vbox.add(button2)
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())
vbox.add(glooey.Placeholder())

grid = glooey.Grid()
grid.add(0, 0, glooey.Placeholder())
grid.add(0, 1, vbox)
grid.add(1, 0, glooey.Placeholder())
grid.add(1, 1, glooey.Placeholder())

grid.padding = 10

grid.set_row_height(1,200)
grid.set_col_width(1,150)

gui.add(grid)



pyglet.app.run()
'''