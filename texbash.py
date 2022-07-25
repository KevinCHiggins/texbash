import pygame
from pygame.locals import *
import numpy
import math

from pgu import gui

app = gui.Desktop()

size = (320, 320)

app.connect(gui.QUIT, app.quit, None)
top_con = gui.Table(width = 1300, height = 660, style = {"padding": 10})

# these two classes are a straight lift from examples/gui9.py in the PGU repo
class OpenDialog(gui.Dialog):
    def __init__(self,**params):
        title = gui.Label("Open Picture")
        
        t = gui.Table()
        
        self.value = gui.Form()
        
        t.tr()
        t.td(gui.Label("Open: "))
        t.td(gui.Input(name="fname"),colspan=3)
        
        t.tr()
        e = gui.Button("Okay")
        e.connect(gui.CLICK,self.send,gui.CHANGE)
        t.td(e,colspan=2)
        
        e = gui.Button("Cancel")
        e.connect(gui.CLICK,self.close,None)
        t.td(e,colspan=2)
        
        gui.Dialog.__init__(self,title,t)

class SaveDialog(gui.Dialog):
    def __init__(self,**params):
        title = gui.Label("Save As...")
        
        t = gui.Table()
        
        self.value = gui.Form()
        
        t.tr()
        t.td(gui.Label("Save: "))
        t.td(gui.Input(name="fname"),colspan=3)
        
        t.tr()
        e = gui.Button("Okay")
        e.connect(gui.CLICK,self.send,gui.CHANGE)
        t.td(e,colspan=2)
        
        e = gui.Button("Cancel")
        e.connect(gui.CLICK,self.close,None)
        t.td(e,colspan=2)
        
        gui.Dialog.__init__(self,title,t)


class TransformationList(gui.Container):
	def __init__(self):
		super().__init__(width = 800, height = 660, valign = -1)

		self._transform_list = []
		self._transformed_grids = []


	def add(self, combo):
		value = combo.value
		print("Adding", value)
		if value == "rotation":
			t = Rotation(self)
		elif value == "scale":
			t = Scale(self)
		elif value == "stretch":
			t = StretchFixedAngle(self)
		self._transform_list.append(t)
		self._transformed_grids.append(range_2d())
		list_pos = len(self._transform_list) - 1
		super().add(t, x=0, y = 140 * list_pos)
		self.recalculate(t)
		self.subject.refresh(self)

	def remove(self, t):
		print("Removing")

		super().remove(t)
		list_pos = self._transform_list.index(t)
		self._transform_list.remove(t)
		self._transformed_grids.pop(list_pos)
		for i in range(list_pos, len(self._transform_list)):
			print("Adjust yourself")
			self._transform_list[i].style.y -= 140
		self.subject.refresh(self)

	def recalculate(self, t):
		list_pos = self._transform_list.index(t)
		for i in range(list_pos, len(self._transform_list)):
			
			g = self.get_grid_feeding(i)
			for row in range(len(g)):
				for col in range(len(g[0])):
					self._transformed_grids[i][row][col] = self._transform_list[i].transform(g[row][col], row, col)
			print(self._transformed_grids[list_pos][0][0][0])
		self.subject.refresh(self)

	def get_final_grid(self):
		i = len(self._transformed_grids)
		if i == 0:
			return range_2d()
		else: return self._transformed_grids[i - 1]
	def get_grid_feeding(self, i):
		if i == 0:
			return range_2d()
		else:
			return self._transformed_grids[i - 1]

	def test_recalc(self, w):
	
		print("Changed widget val: ", w.value)

class Transformation(gui.Table):
	def __init__(self, parent):
		self.parent = parent
		self.floor = 0
		self.floor_ceil_diff = 0
		super().__init__(width = 800, height = 140, style = {"padding": 10})
		
		self.editable_params = []
		self.slider_params = [] # because sliders broadcast so many changes, they need different treatment

		self.tr()
		#self.up_btn = gui.Button("Move up")
		#self.td(self.up_btn)
		update_btn = gui.Button("Update") # temp expedient
		update_btn.connect(gui.CLICK, transformation_list.recalculate, self)
		self.td(update_btn)

		#transform_container.td(gui.Label("Transform", cls = "h2", style = {"padding": 10}))
		self.td(gui.Label("Floor"), width = 100, align = -1)
		self.floor_inp = KeyUpInput(value = 1, size = 4)
		self.editable_params.append(self.floor_inp)
		self.td(self.floor_inp)
		self.td(gui.Label("Ceiling"))
		self.ceil_inp = KeyUpInput(value = 1, size = 4)
		self.editable_params.append(self.ceil_inp)
		self.td(self.ceil_inp)

		self.td(gui.Label("Transform X"))
		self.t_x_slider = gui.HSlider(value = 160, min = 0, max = 319, size = 10, width = 320)
		self.td(self.t_x_slider, colspan = 4)
		self.slider_params.append(self.t_x_slider)

		# these widgets are added later but need to be defined now
		# so the mask can take their values in its constructor
		self.thick_inp = KeyUpInput(value = 100, size = 4)
		self.m_x_slider = gui.HSlider(value = 160, min = 0, max = 319, size = 10, width = 320)
		self.m_y_slider = gui.HSlider(value = 160, min = 0, max = 319, size = 10, width = 320)

		self.ang_inp = KeyUpInput(value = 10, size = 4)
		self.ang_rads = degs_to_rads(float(self.ang_inp.value)) # messy having this here

		self.mask = Mask(self)
		self.mask.refresh()
		self.td(self.mask, rowspan = 4) #, valign = 1)
		self.tr()
		self.dup_btn = gui.Button("Duplicate")
		self.td(self.dup_btn)
		self.other_label = gui.Label("Generic")
		self.td(self.other_label)
		self.other_inp = KeyUpInput(value = 1, size = 4)
		self.editable_params.append(self.other_inp)
		self.td(self.other_inp)
		self.td(gui.Spacer(1,1), colspan = 2)
		self.td(gui.Label("Transform Y"))
		self.t_y_slider = gui.HSlider(value = 160, min = 0, max = 319, size = 10, width = 320)
		self.slider_params.append(self.t_y_slider)
		self.td(self.t_y_slider, colspan = 4)


		self.tr()
		self.del_btn = gui.Button("Delete")
		self.del_btn.connect(gui.CLICK, self.parent.remove, self)
		self.td(self.del_btn)

		#mask_container.td(gui.Label("Mask", cls = "h2", style = {"padding": 10}))
		self.td(gui.Label("Radial"))

		self.type_switch = gui.Switch(False)
		self.editable_params.append(self.type_switch)
		self.td(self.type_switch, width = 100)
		self.td(gui.Label("Thickness"))

		self.editable_params.append(self.thick_inp)
		self.td(self.thick_inp)

		self.td(gui.Label("Mask X"))

		self.slider_params.append(self.m_x_slider)
		self.td(self.m_x_slider, colspan = 4)
		self.tr()

		self.down_btn = gui.Button("Move down")
		self.td(self.down_btn)
		self.td(gui.Label("Angle"))

		self.editable_params.append(self.ang_inp)
		self.td(self.ang_inp)
		self.td(gui.Spacer(1,1), colspan = 2)
		self.td(gui.Label("Mask Y"))

		self.slider_params.append(self.m_y_slider)
		self.td(self.m_y_slider, colspan = 4)

		for w in self.editable_params:
			w.connect(gui.ACTIVATE, self.try_recalc, w)
		for w in self.slider_params:
			w.connect(gui.BLUR, self.try_recalc, w)

	def try_recalc(self, widget):
		if self.validate():
			# really should only recalc changed items
			self.mask.radial = self.type_switch.value
			self.floor = float(self.floor_inp.value)
			self.floor_ceil_diff = float(self.ceil_inp.value) - self.floor
			self.ang_rads = degs_to_rads(float(self.other_inp.value))
			self.mask.refresh()
			self.mask.repaint()
			self.parent.recalculate(self)

	def validate(self):
		try:
			for w in self.editable_params:
				f = float(w.value)
			for w in self.slider_params:
				f = float(w.value)
			return True
		except ValueError:
			print("Couldn't form float")
		except OverflowError:
			print("Value out of bounds for float")
		return False

def degs_to_rads(f):
	return f * math.pi / 180

class Rotation(Transformation):
	def __init__(self, parent):
		super().__init__(parent)
		self.other_label.value = "Angle"

	def transform(self, coords_to_transform, grid_x, grid_y):
		ang = float(self.ang_rads) * (self.floor + (self.mask.array[grid_y][grid_x] * self.floor_ceil_diff))

		origin = (int(self.t_x_slider.value), int(self.t_y_slider.value))
		x = coords_to_transform[0] - origin[0]
		y = coords_to_transform[1] - origin[1]

		c = math.cos(ang)
		s = math.sin(ang)
		rot_x = x * c - y * s
		rot_y = x * s + y * c
		rot_x = rot_x + origin[0]
		rot_y = rot_y + origin[1]

		return numpy.array([rot_x, rot_y])

class Scale(Transformation):
	def __init__(self, parent):
		print("T")
		super().__init__(parent)
		self.other_label.value = "Intensity"
		self.other_inp.value = 1
		self.floor_inp.value = 0
		self.t_x_slider.disabled = True
		self.t_y_slider.disabled = True

	def transform(self, coords_to_transform, grid_x, grid_y):
		scale = 1 + (float(self.other_inp.value) - 1) * (self.floor + (self.mask.array[grid_y][grid_x] * self.floor_ceil_diff)) + 0.0000001

		origin = (int(self.t_x_slider.value), int(self.t_y_slider.value))
		x = coords_to_transform[0] - origin[0]
		y = coords_to_transform[1] - origin[1]

		x = x / scale + origin[0]
		y = y / scale + origin[1]

		return numpy.array([x, y])

class StretchFixedAngle(Transformation):
	def __init__(self, parent):
		super().__init__(parent)

		self.other_inp.value = 1
		self.floor_inp.value = 0
		self.other_label.value = "Angle"

	def transform(self, coords_to_transform, grid_x, grid_y):
		ang = float(self.ang_rads)
		intensity = (self.floor + (self.mask.array[grid_y][grid_x] * self.floor_ceil_diff)) + 0.00000001

		origin = (int(self.t_x_slider.value), int(self.t_y_slider.value))
		x = coords_to_transform[0] - origin[0]
		y = coords_to_transform[1] - origin[1]

		scale_x = math.cos(ang) * intensity * 320 # SHOULDN'T BE IN INNER LOOP!!! (Not the only offender)
		scale_y = math.sin(ang) * intensity * 320

		stretched_x = x - scale_x + origin[0]
		stretched_y = y - scale_y + origin[1]

		return numpy.array([stretched_x, stretched_y])

def dist(x1, y1, x2, y2):
	return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


class Mask(gui.Image):
	def __init__(self, parent):
		self.radial = False
		self.parent = parent
		self.array = numpy.zeros((320, 320))
		super().__init__(pygame.Surface((100, 100), depth = 8))
		for i in range(255):
			self.value.set_palette_at(i, Color(i, i, i))
	def refresh(self):
		thickness = int(self.parent.thick_inp.value)
		if self.radial:
			centre = (int(self.parent.m_x_slider.value), int(self.parent.m_y_slider.value))
			for row in range(len(self.array)):
				for col in range(len(self.array[0])):
					d = dist(centre[0], centre[1], col, row)
					if d > thickness:
						d = thickness
					self.array[row][col] = ((thickness - d) + 0.00001) / thickness
		else:
			centre = (int(self.parent.m_x_slider.value), int(self.parent.m_y_slider.value))
			ang_rads = degs_to_rads(float(self.parent.ang_inp.value))
			s = math.sin(ang_rads)
			c = math.cos(ang_rads)
			slope = s / c
			left_wall_intersect_y = slope * (0 - centre[0]) + centre[1]
			for row in range(len(self.array)):
				for col in range(len(self.array[0])):
					slope_from_left_wall_intersect = (row - left_wall_intersect_y) / (col + 0.000001)
					if slope_from_left_wall_intersect > slope:
						self.array[row][col] = ang_rads < math.pi
					else:
						d = (c * (centre[1] - row)) - (s * (centre[0] - col))
						if d > thickness:
							d = thickness
						self.array[row][col] = ((thickness - d) + 0.00001) / thickness
		mask_pixels = pygame.surfarray.pixels2d(self.value)
		for row in range(len(mask_pixels)):
			for col in range(len(mask_pixels[0])):
				mask_pixels[col][row] = int(self.array[int(row * 320 / 100)][int(col * 320 / 100)] * 255)

class DistortedTiling(pygame.Surface):
	def __init__(self, size, src):
		self.grid = range_2d()

		self.src = src
		self.tile_size = src.get_size()
		super().__init__(size)
		self.apply()
	def set_grid(self, g):
		self.grid = g

		self.apply()
	def apply(self):
		src_pixels = pygame.surfarray.pixels3d(self.src)
		targ_pixels = pygame.surfarray.pixels3d(self)
		for row in range(len(targ_pixels)):
			for col in range(len(targ_pixels[0])):
				final_x = int(self.grid[row][col][0]) % self.tile_size[0]
				final_y = int(self.grid[row][col][1]) % self.tile_size[1]
				targ_pixels[row][col] = src_pixels[final_x][final_y]
#
#		buff.write(bytes(ba))

def range_2d():
	global size
	width = size[0]
	height = size[1]
	grid = numpy.zeros((height, width, 2))
	for row in range(height):
		for col in range(width):
			grid[row][col][0] = row
			grid[row][col][1] = col

	return grid








top_con.tr()




src = pygame.image.load("lzr02_2.png")
tiling = DistortedTiling((320,320), src)

class DistortedTilingImage(gui.Image):
	def refresh(self, transform_list):
		grid = transform_list.get_final_grid()
		self.value.set_grid(grid)
		self.repaint()

img = DistortedTilingImage(tiling)






transformation_list = TransformationList()
transformation_list.subject = img
img_pane = gui.Table(width = 320, height = 400)
img_con = gui.Container(width = 320, height = 320)
img_con.add(img, 0, 0)
img_pane.tr()
img_pane.td(img_con, colspan = 2)
img_pane.tr()


def action_saveas(value):
	save_d.close()
	fname = save_d.value['fname'].value
	pygame.image.save(tiling, fname)
    
def action_open(value):
	global src, tiling, img, transformation_list
	open_d.close()
	fname = open_d.value['fname'].value
	src = pygame.image.load(fname)
	tiling = DistortedTiling((320,320), src)
	img_con.remove(img)
	img = DistortedTilingImage(tiling)
	

	img_con.add(img, 0, 0)
	transformation_list.subject = img

	img.repaint()

open_d = OpenDialog()
open_d.connect(gui.CHANGE, action_open, None)

save_d = SaveDialog()

save_d.connect(gui.CHANGE, action_saveas, None)



open_btn = gui.Button("Open")
open_btn.connect(gui.CLICK, open_d.open, None)
img_pane.td(open_btn)

save_btn = gui.Button("Save")
save_btn.connect(gui.CLICK, save_d.open, None)

img_pane.td(save_btn)

top_con.td(img_pane, colspan = 2, valign = -1)

top_con.td(transformation_list, rowspan = 2)
top_con.tr()
add_btn = gui.Button("Add transformation")
combo = gui.Select(value = "rotation")
add_btn.connect(gui.CLICK, transformation_list.add, combo)
top_con.td(add_btn, valign = -1)

combo.add("Rotation", "rotation")
combo.add("Scale", "scale")
combo.add("Stretch", "stretch")
top_con.td(combo, valign = -1)
top_con.tr()

class KeyUpInput(gui.Input):
	def event(self,e):
		used = None
		if e.type == KEYDOWN:
			if e.key == K_BACKSPACE:
				if self.pos:
					self._setvalue(self.value[:self.pos-1] + self.value[self.pos:])
					self.pos -= 1
			elif e.key == K_DELETE:
				if len(self.value) > self.pos:
					self._setvalue(self.value[:self.pos] + self.value[self.pos+1:])
			elif e.key == K_HOME:
				self.pos = 0
			elif e.key == K_END:
				self.pos = len(self.value)
			elif e.key == K_LEFT:
				if self.pos > 0: self.pos -= 1
				used = True
			elif e.key == K_RIGHT:
				if self.pos < len(self.value): self.pos += 1
				used = True
			elif e.key == K_RETURN:
				pass # don't want a weird character going into the box
			elif e.key == K_TAB:
				pass
			else:
				if (type(e.unicode) == str):
					c = e.unicode
				else:
					c = (e.unicode).encode('latin-1')

				try:
					if c:
						self._setvalue(self.value[:self.pos] + c + self.value[self.pos:])
						self.pos += 1
				except (TypeError, ValueError): #ignore weird characters
					pass
			self.repaint()
		elif e.type == gui.const.FOCUS:
			self.repaint()
		elif e.type == gui.const.BLUR:
			self.repaint()
		elif e.type == KEYUP:
			if e.key == K_RETURN:
				self.send(gui.const.ACTIVATE)

		self.pcls = ""
		if self.container.myfocus is self: self.pcls = "focus"

		return used

app.run(top_con)

