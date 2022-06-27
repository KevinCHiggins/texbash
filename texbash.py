# Import tkinter
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import math
import sys

bashed_img_width = 320
bashed_img_height = 320
bashed_img_size = (bashed_img_width, bashed_img_height)

scaled_pi = math.pi / 4
changed_generators = {}
sin_cache = {}
cos_cache = {}

def test_inv():
	print(matrix_multiply(rot_matrix(3), rot_matrix(-3)))
	print(matrix_multiply(scale_matrix(.2, 5), scale_matrix(5, .2)))
	print(matrix_multiply(translate_matrix(100, 200), translate_matrix(-100, -200)))

def bash(source, source_size, bashed_size):
	k = changed_generators.keys()

	data = bytearray(bashed_size[0] * bashed_size[1])

	h_w = int(bashed_img_width / 2)
	h_h = int(bashed_img_height / 2)
	static_trans_matrix = changed_generators["trans_stat"].generate()
	static_rot_matrix = changed_generators["rot_stat"].generate()
	static_scale_matrix = changed_generators["scale_stat"].generate()
	static_matrices = matrix_multiply_list([static_trans_matrix, static_rot_matrix, static_scale_matrix])

	for y in range(bashed_img_height):
		y_proportion = (y / h_h)
		scaled_y_proportion = y_proportion * .01
		scaled_shifted_y_proportion = scaled_y_proportion + 0.5
		for x in range(bashed_img_width):
			x_proportion = (x / h_w)
			scaled_x_proportion = x_proportion * .01

			r_matrix = rot_matrix_inv(scaled_pi * x_proportion * y_proportion)
			s_matrix = scale_matrix_inv(scaled_x_proportion + 0.5, scaled_shifted_y_proportion)

			final_matrix = matrix_multiply_list([static_matrices, r_matrix, s_matrix])


			#final_matrix = matrix_multiply(final_matrix, static_scale_matrix)


			bashed_pos = transform_pos((x, y), final_matrix)

			b_x = int(bashed_pos[0])
			b_y = int(bashed_pos[1])
			output_index = y * bashed_size[0] + x
			source_index = (b_y % source_size[1] * source_size[0] + b_x % source_size[0])
			data[output_index] = source[source_index]
	return bytes(data)

def get_img_size(img):
	return (img.width, img.height)

def translate_matrix(x, y):
	return (
		(1, 0, 0),
		(0, 1, 0),
		(x, y, 1))

def cached_sin(angle):
	if angle in sin_cache:
		return sin_cache[angle]
	else:
		s = math.sin(angle)
		sin_cache[angle] = s
		return s

def cached_cos(angle):
	if angle in cos_cache:
		return cos_cache[angle]
	else:
		s = math.cos(angle)
		cos_cache[angle] = s
		return s

def rot_matrix(angle):
	s = cached_sin(angle)
	c = cached_cos(angle)
	return (
		(c, 0 - s, 0),
		(s, c, 0),
		(0, 0, 1))

def scale_matrix(sx, sy):
	return (
		(sx, 0, 0),
		(0, sy, 0),
		(0, 0, 1))

def translate_matrix_inv(x, y):
	x = 0 - x
	y = 0 - y
	return translate_matrix(x, y)

def rot_matrix_inv(angle):
	agle = 0 - angle
	return rot_matrix(angle)

def scale_matrix_inv(sx, sy):
	# doesn't guard against divide by zero
	sx = 1 / sx
	sy = 1 / sy
	return scale_matrix(sx, sy)

def matrix_multiply_list(l):
	length = len(l)
	if length < 2:
		print("No multiplication performed")
		return l[0]
	a = l[0]
	for i in range(1, length):
		b = l[i]
		a = matrix_multiply(a, b)
	return a


def matrix_multiply(a, b):
	return (
		(dot_product(get_col(a, 0), b[0]), dot_product(get_col(a, 1), b[0]), dot_product(get_col(a, 2), b[0])),
		(dot_product(get_col(a, 0), b[1]), dot_product(get_col(a, 1), b[1]), dot_product(get_col(a, 2), b[1])),
		(dot_product(get_col(a, 0), b[2]), dot_product(get_col(a, 1), b[2]), dot_product(get_col(a, 2), b[2])))

def dot_product(vec_a, vec_b):
	return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2]

def get_col(mat, col):
	return (mat[0][col], mat[1][col], mat[2][col])


def transform_pos(pos, mat):
	return (pos[0] * mat[0][0] + pos[1] * mat[1][0] + mat[2][0],
		pos[0] * mat[0][1] + pos[1] * mat[1][1] + mat[2][1])

test_inv()

root = Tk()

tb_frame = ttk.Frame(root, padding = (10, 10, 10, 10))

tb_frame.pack(expand = True)



class EditableTranslationMatrixGenerator:

	def __init__(self, container, name, on_change_callback):
		self.frame = ttk.Frame(container, padding = (5, 5, 5, 5))

		self.frame.pack()
		self.name_label = ttk.Label(self.frame, text = name)
		self.name_label.pack(side = "left")
	
		self.x_label = ttk.Label(self.frame, text = "X")
		self.x_label.pack(side = "right")
		self.x_var = IntVar(container, 0)
		self.x_var.trace_add("write", on_change_callback)
		self.x_entry = ttk.Entry(self.frame, width = 3,
			textvariable = self.x_var)
		self.x_entry.pack(side = "right")
		self.y_label = ttk.Label(self.frame, text = "Y")
		self.y_label.pack(side = "right")
		self.y_var = IntVar(container, 0)
		self.y_var.trace_add("write", on_change_callback)
		self.y_entry = ttk.Entry(self.frame, width = 3,
			textvariable = self.y_var)
		self.y_entry.pack(side = "right")
	def generate(self):
		return translate_matrix_inv(self.x_var.get(), self.y_var.get())

class EditableScaleMatrixGenerator:
	def __init__(self, container, name, on_change_callback):
		self.frame = ttk.Frame(container, padding = (5, 5, 5, 5))

		self.frame.pack()
		self.name_label = ttk.Label(self.frame, text = name)
		self.name_label.pack(side = "left")
	
		self.x_label = ttk.Label(self.frame, text = "X")
		self.x_label.pack(side = "right")
		self.x_var = DoubleVar(container, 1)
		self.x_var.trace_add("write", on_change_callback)
		self.x_entry = ttk.Entry(self.frame, width = 3,
			textvariable = self.x_var)
		self.x_entry.pack(side = "right")
		self.y_label = ttk.Label(self.frame, text = "Y")
		self.y_label.pack(side = "right")
		self.y_var = DoubleVar(container, 1)
		self.y_var.trace_add("write", on_change_callback)
		self.y_entry = ttk.Entry(self.frame, width = 3,
			textvariable = self.y_var)
		self.y_entry.pack(side = "right")
	def generate(self):
		return scale_matrix_inv(self.x_var.get(), self.y_var.get())

class EditableRotationMatrixGenerator:
	def __init__(self, container, name, on_change_callback):
		self.frame = ttk.Frame(container, padding = (5, 5, 5, 5))

		self.frame.pack()
		self.name_label = ttk.Label(self.frame, text = name)
		self.name_label.pack(side = "left")
	
		self.angle_label = ttk.Label(self.frame, text = "Angle")
		self.angle_label.pack(side = "right")
		self.angle_var = IntVar(container, 0)
		self.angle_var.trace_add("write", on_change_callback)
		self.angle_entry = ttk.Entry(self.frame, width = 3,
			textvariable = self.angle_var)
		self.angle_entry.pack(side = "right")
	def generate(self):
		return rot_matrix_inv(self.angle_var.get())





#vars_register.append(add_matrix(tb_frame, "Translate (parametric)", ["X Low","X High","Y Low", "Y High"]))
#vars_register.append(add_matrix(tb_frame, "Rotate (parametric)", ["Angle Low", "Angle High"]))
#vars_register.append(add_matrix(tb_frame, "Scale (parametric)", ["X Low", "X High", "Y Low", "Y High"]))


source_img = Image.open("lzr02_2.bmp")
source_palette = source_img.getpalette()
source_img_bytes = source_img.tobytes()

bashed_bytes = bytes(bashed_img_width * bashed_img_height)

bashed_img = Image.frombuffer("P", bashed_img_size, bashed_bytes, "raw")
bashed_img.putpalette(source_palette)

bashed_photo_img = ImageTk.PhotoImage(bashed_img)

bashed_display = Label(tb_frame, image = bashed_photo_img)

bashed_display.configure(image = bashed_photo_img)
bashed_display.image = bashed_photo_img


bashed_display.pack()

sep = ttk.Separator(tb_frame, orient="horizontal")
sep.pack()

def refresh(*args):
	print("Refresh")
	bashed_bytes = bash(source_img_bytes, get_img_size(source_img), bashed_img_size)
	bashed_img = Image.frombuffer("P", bashed_img_size, bashed_bytes, "raw")
	bashed_img.putpalette(source_palette)

	bashed_photo_img = ImageTk.PhotoImage(bashed_img)

	bashed_display.configure(image = bashed_photo_img)
	bashed_display.image = bashed_photo_img
	print("Done")

changed_generators["trans_stat"] = EditableTranslationMatrixGenerator(tb_frame, "Translate (static):", refresh)
changed_generators["rot_stat"] = EditableRotationMatrixGenerator(tb_frame, "Rotate (static):", refresh)
changed_generators["scale_stat"] = EditableScaleMatrixGenerator(tb_frame, "Scale (static):", refresh)



refresh()


root.mainloop()


