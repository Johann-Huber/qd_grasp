# Created by Giuseppe Paolo 
# Date: 11/09/2020

import pygame as pg
import numpy  as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)


# The goal of this class is to just collect all the screen variables at one place,
# that all other classes can refer to and update.
class Basic:
  def __init__(self, frameside=400, frame_grid=(1, 1), title='', id_gui=-1, buttons=()):
    pg.init()
    screen = pg.display.set_mode((frame_grid[0] * frameside, frame_grid[1] * frameside))
    screen.fill(WHITE)

    self.n_frames = frame_grid[0] * frame_grid[1]
    self.has_canvas = np.zeros(self.n_frames)

    self.screen = screen  # A shared screen everyone can update
    self.frameside = frameside  # The length of the side of a square frame
    self.frame_grid = frame_grid  #

    self.canvas_prop = 0.9013  # Canvas side size relative to frame side
    self.circ_thick = 5  # How thick line to draw around matrixes

    if len(buttons) != 0:
      self.buttons = buttons
      self.n_buttons = len(self.buttons)
      self.w_button = 0.5 * self.frameside
      self.h_button = 0.5 * self.frameside * 0.9 / self.n_buttons
      self.b_offset = (0.5, 0.25)
      self.draw_gui()

    pg.display.set_caption(title)
    pg.display.update()

  def get_displacement(self, frame_id):

    i_x = frame_id % self.frame_grid[0]
    i_y = frame_id / self.frame_grid[0]

    return (i_x * self.frameside, i_y * self.frameside)

  def get_canvas_displacement(self, frame_id):

    frame_displace = np.array(self.get_displacement(frame_id))
    canvas_displace = np.array([(1. - self.canvas_prop) * self.frameside / 2, self.circ_thick])

    return np.int32(frame_displace + canvas_displace + 0.5)

  def get_x_from_frame(self, frame_pos):
    frame_pos = np.array(frame_pos)
    canvas_displace = np.array([(1. - self.canvas_prop) * self.frameside / 2, self.circ_thick])

    x = 1. * (frame_pos - canvas_displace) / (self.canvas_prop * self.frameside)

    return x

  def clean_frame(self, frame_id):
    # Constants
    circ_thick = self.circ_thick
    canvas_side = 1. * self.canvas_prop * self.frameside

    # Erase old
    frame_displace = self.get_displacement(frame_id)
    canvas_displace = np.array([(1. - self.canvas_prop) * self.frameside / 2, circ_thick])

    # Clear up old
    framerect = np.append(frame_displace, np.array([self.frameside, self.frameside]))
    pg.draw.rect(self.screen, WHITE, framerect)

    # Draw background
    circumfence = np.append(frame_displace + canvas_displace - circ_thick, np.array([canvas_side, canvas_side]) + circ_thick * 2)
    inside = np.append(frame_displace + canvas_displace, np.array([canvas_side, canvas_side]))

    pg.draw.rect(self.screen, BLACK, circumfence)
    pg.draw.rect(self.screen, WHITE, inside)

  #

  # Draw a 2D image - Ad 3D capability later
  def draw_matrix(self, matrix, frame_id, v_min=1337, v_max=1337, matrix_text=''):

    frame_displace = self.get_displacement(frame_id)

    # Clear up old
    self.clean_frame(frame_id)

    # Draw matrix
    i_max = matrix.shape[0]
    j_max = matrix.shape[1]

    if v_min == 1337:
      min_value = matrix.min()
    else:
      min_value = v_min

    if v_max == 1337:
      max_value = matrix.max()
    else:
      max_value = v_max

    if min_value == max_value:
      max_value += 1

    # compute the size of a pixel
    circ_thick = self.circ_thick
    canvas_side = 1. * self.canvas_prop * self.frameside
    pixel_side = 1. * self.canvas_prop * self.frameside / max(i_max, j_max)
    canvas_displace = np.array([(1. - self.canvas_prop) * self.frameside / 2, circ_thick])

    rectangle = np.array([0, 0, pixel_side, pixel_side])

    for i in range(i_max):
      for j in range(j_max):
        grid_displace = np.array([i * pixel_side, j * pixel_side])

        # Sum upp all displacements
        displace = frame_displace + canvas_displace + grid_displace

        # Change to rectangle format
        rec_disp = np.append(displace, np.array([0, 0]))

        # The shape to draw: [x_start, y_start, width, height]
        rec_xywh = np.int32(rec_disp + rectangle + 0.5)

        # Color of shape
        rel_value = 1. * (matrix[i, j] - min_value) / (max_value - min_value)
        rec_color = np.uint8(rel_value * np.array([255, 255, 255]) + 0.5)

        # Draw it!
        pg.draw.rect(self.screen, rec_color, rec_xywh)

    # Plot titles
    text_center = np.array([self.frameside / 2, (2 * self.frameside + canvas_side) / 3])
    text_center = np.int32(frame_displace + text_center)

    self.display_message(matrix_text, text_center)

  # pg.display.update()

  # state_coords: array with 2D x-coord of every state-center
  # state_color:	array with gray scale value of corresponding state
  def draw_free_states(self, frame_id, state_coord, state_color, size=4):

    new_color = 1. * state_color - np.min(state_color)
    max_value = np.max(new_color)

    if max_value > 0:
      new_color /= max_value

    for i in range(len(state_coord)):
      color = np.uint8(new_color[i] * np.array(WHITE))
      self.draw_x_coord(frame_id, state_coord[i], color, size)

  def draw_x_states(self, frame_id, x_now=[], x_goal=[], color_goal=RED):

    if len(x_now) > 0:
      self.draw_x_coord(frame_id, x_now, BLUE, 8)

    if len(x_goal) > 0:
      self.draw_x_coord(frame_id, x_goal, color_goal, 4)

    pg.display.update()

  def draw_x_coord(self, frame_id, x, color, size=4):

    frame_displace = self.get_displacement(frame_id)
    canvas_displace = np.array([(1. - self.canvas_prop) * self.frameside / 2, self.circ_thick])

    pos_now_relative = x * self.frameside * self.canvas_prop
    pos_now = frame_displace + canvas_displace + pos_now_relative
    pos_now = np.int32(pos_now + 0.5)  # round

    pg.draw.circle(self.screen, color, pos_now, size)

  def draw_gui(self):
    n_buttons = self.n_buttons
    w_button = self.w_button
    h_button = self.h_button

    x_c = self.b_offset[0] * self.frameside
    y_cs = np.linspace(self.b_offset[1], 1. - self.b_offset[1], n_buttons) * self.frameside

    for i in range(n_buttons):
      y_c = y_cs[i]

      # Plot box
      x = x_c - 0.5 * w_button
      y = y_c - 0.5 * h_button

      pg.draw.rect(self.screen, GRAY, (x, y, w_button, h_button))

      # Plot text
      text_center = np.int32((x_c, y_c))
      text_size = np.uint8(min(0.8 * h_button, 0.1 * self.frameside))

      self.display_message(self.buttons[i], text_center, size=text_size)

  # Some methods for handling text
  def text_objects(self, text, font):
    text_surface = font.render(text, True, BLACK)
    return text_surface, text_surface.get_rect()

  def display_message(self, text, text_center, size=30):
    font = pg.font.SysFont(None, size)

    text_surface, text_rectangle = self.text_objects(text, font)
    text_rectangle.center = text_center

    self.screen.blit(text_surface, text_rectangle)

  def get_frame_and_pos(self, window_target):

    frame_position = np.array(window_target) / self.frameside
    frame_id = frame_position[1] * self.frame_grid[0] + frame_position[0]

    displace = np.int32(self.get_displacement(frame_id))
    g_coord = np.int32(window_target) - displace

    return frame_id, g_coord

  def get_button(self, pos):

    n_buttons = self.n_buttons
    w_button = self.w_button
    h_button = self.h_button

    x_c = self.b_offset[0] * self.frameside
    x_min = x_c - w_button / 2
    x_max = x_c + w_button / 2

    if pos[0] < x_min or x_max < pos[0]:
      return -1

    y_cs = np.linspace(self.b_offset[1], 1. - self.b_offset[1], n_buttons) * self.frameside

    for i in range(n_buttons):
      y_min = y_cs[i] - h_button / 2
      y_max = y_cs[i] + h_button / 2

      if y_min < pos[1] and pos[1] < y_max:  # Found the button!
        return i

    return -1

