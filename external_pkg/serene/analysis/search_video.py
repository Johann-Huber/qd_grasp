# Created by Giuseppe Paolo 
# Date: 15/02/2021

import os, shutil
import parameters
import pickle as pkl
import numpy as np
import pygame as pg
import pygame.freetype  # Import the freetype module.
import time
import ffmpeg
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
from external_pkg.serene.environments.environments import registered_envs

class Renderer(object):
  """
  This class plots the archive/population/offs generation distributions
  """
  def __init__(self, bs_size, bs_min, surface_size=(300, 300), canvas_size=(1300, 400), save_video=False, history=False, path=None, frame_rate=10):
    self.history = history
    self.colors = cm.get_cmap('gist_rainbow')
    self.save_video = save_video
    self.path = path

    if self.save_video and not os.path.exists(os.path.join(self.path, 'tmp')):
      os.mkdir(os.path.join(self.path, 'tmp'))

    pg.init()
    self.canvas_size = np.array(canvas_size)
    self.surface_size = np.array(surface_size)
    self.bs_size = bs_size
    self.bs_min = bs_min # The coordinate min of the bs wrt the pixel coord. Needed to transform the bs coord to pixel
    self.canvas_center = self.canvas_size/2.
    self.surface_center = self.surface_size/2.
    self.TARGET_FPS = frame_rate

    if not self.save_video:
      self.screen = pg.display.set_mode(self.canvas_size, 0, 32)
    else:
      self.screen = pg.Surface(self.canvas_size)
    self.screen.fill((255, 245, 200))  ## Draw white background

    self.clock = pg.time.Clock()

    self.font = pg.font.SysFont('Arial', 18)

    space = 20 + self.surface_size[0]
    y_pose = self.canvas_size[1] - self.surface_size[1] - 10
    # Create surfaces
    # ---------------------------------------------------------------
    # Fader
    title_surface = self.font.render('Generation: {}'.format(10000), True, (0, 0, 0))
    self.title_bkg = pg.Surface(title_surface.get_size())
    self.title_bkg.fill((255, 245, 200))
    # Background
    self.bkg = pg.Surface(self.surface_size)
    self.bkg.fill(pg.color.THECOLORS["white"])  ## Draw white background
    # Pop
    self.pop_surface = pg.Surface(self.surface_size)
    self.pop_surface.fill(pg.color.THECOLORS["white"])  ## Draw white background
    self.pop_surface_zero = [20 + 0 * space, y_pose]
    text_surface = self.font.render('Population', True, (0, 0, 0))
    self.screen.blit(text_surface, (self.pop_surface_zero[0] + self.surface_size[0]/2 - text_surface.get_width()/2,
                                     y_pose-30))
    # Off
    self.off_surface = pg.Surface(self.surface_size)
    self.off_surface.fill(pg.color.THECOLORS["white"])  ## Draw white background
    self.off_surface_zero = [20 + 1 * space, y_pose]
    text_surface = self.font.render('Offsprings', True, (0, 0, 0))
    self.screen.blit(text_surface,
                     (self.off_surface_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
                      y_pose - 30))
    # Arch
    self.arch_surface = pg.Surface(self.surface_size, pygame.SRCALPHA)
    self.arch_surface.fill(pg.color.THECOLORS["white"])  ## Draw white background
    self.arch_surface_zero = [20 + 2 * space, y_pose]
    text_surface = self.font.render('Archive', True, (0, 0, 0))
    self.screen.blit(text_surface,
                     (self.arch_surface_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
                      y_pose - 30))
    # Rew Arch
    self.rew_arch_surface = pg.Surface(self.surface_size)
    self.rew_arch_surface.fill(pg.color.THECOLORS["white"])  ## Draw white background
    self.rew_arch_surface_zero = [20 + 3 * space, y_pose]
    text_surface = self.font.render('Rew Archive', True, (0, 0, 0))
    self.screen.blit(text_surface,
                     (self.rew_arch_surface_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
                      y_pose - 30))
    # ---------------------------------------------------------------

  def render(self, data):
    """
    This function renders all the generations.
    :param data:
    :return:
    """
    self.max_gen = max(list(data['population'].keys()))
    arch_points = 0
    arch_old_points = 0
    arch_new_points = []
    rew_arch_points = 0
    rew_arch_old_points = 0
    rew_arch_new_points = []

    for gen in range(1, self.max_gen):
      color = self.colors(gen/self.max_gen, bytes=True)

      self.screen.blit(self.title_bkg, (20, 20))
      title_surface = self.font.render('Generation: {}'.format(gen), True, (0, 0, 0))
      self.screen.blit(title_surface, (20, 20))

      # Draw Pop and Off
      # -----------------------------------------
      if not self.history:
        self.pop_surface = self.bkg.copy()
        self.off_surface = self.bkg.copy()

      self.draw(np.stack(data['population'][gen]), self.pop_surface, self.pop_surface_zero, color)
      self.draw(np.stack(data['offsprings'][gen]), self.off_surface, self.off_surface_zero, color)
      # -----------------------------------------

      # Draw Arch
      # -----------------------------------------
      self.arch_surface = self.bkg.copy() # Reset background
      if gen > 1:
        arch_old_points = arch_points.copy()
      arch_points = np.stack(data['archive'][gen])

      if gen > 1:
        # Find replaced points and hide them (Needed for ME archive)
        replaced_points = arch_old_points[~((arch_old_points[:, None, :] == arch_points).all(-1)).any(1)]
        if len(replaced_points) > 0:
          self.draw(replaced_points, self.arch_surface, self.arch_surface_zero, pg.color.THECOLORS["white"])

          # Delete replaced points from history of points
          for k in range(len(arch_new_points)):
            arch_new_points[k] = arch_new_points[k][~((arch_new_points[k][:, None, :] == replaced_points).all(-1)).any(1)]

        # Add last new points
        arch_new_points.append(arch_points[~((arch_points[:, None, :] == arch_old_points).all(-1)).any(1)])
        # Draw history of new points
        for k in range(len(arch_new_points)):
          self.draw(arch_new_points[k], self.arch_surface, self.arch_surface_zero, self.colors(k/self.max_gen, bytes=True))
      else:
        self.draw(arch_points, self.arch_surface, self.arch_surface_zero, color)
      # -----------------------------------------

      # Reward Archive
      # -----------------------------------------
      if gen in data['rew archive'] and len(data['rew archive'][gen]) > 0:
        self.rew_arch_surface = self.bkg.copy()  # Reset background
        if gen > 1 and not isinstance(rew_arch_points, int):
          rew_arch_old_points = rew_arch_points.copy()
        rew_arch_points = np.stack(data['rew archive'][gen])

        if gen > 1 and not isinstance(rew_arch_old_points, int):
          # Find replaced points and hide them (Needed for ME archive)
          replaced_points = rew_arch_old_points[~((rew_arch_old_points[:, None, :] == rew_arch_points).all(-1)).any(1)]
          if len(replaced_points) > 0:
            self.draw(replaced_points, self.rew_arch_surface, self.rew_arch_surface_zero, pg.color.THECOLORS["white"])

            # Delete replaced points from history of points
            for k in range(len(rew_arch_new_points)):
              rew_arch_new_points[k] = rew_arch_new_points[k][~((rew_arch_new_points[k][:, None, :] == replaced_points).all(-1)).any(1)]

          # Add last new points
          rew_arch_new_points.append(rew_arch_points[~((rew_arch_points[:, None, :] == rew_arch_old_points).all(-1)).any(1)])
          # Draw history of new points
          for k in range(len(rew_arch_new_points)):
            self.draw(rew_arch_new_points[k], self.rew_arch_surface, self.rew_arch_surface_zero, self.colors(k / self.max_gen, bytes=True))
        else:
          self.draw(rew_arch_points, self.rew_arch_surface, self.rew_arch_surface_zero, color)
      else:
        self.draw(np.array([[0, 0]]), self.rew_arch_surface, self.rew_arch_surface_zero, pg.color.THECOLORS["white"])
      # -----------------------------------------

      if not self.save_video:
        pg.display.flip() ## Need to flip cause of drawing reasons
        self.clock.tick(self.TARGET_FPS)
      else:
        imgdata = pg.surfarray.array3d(self.screen)
        plt.imsave(os.path.join(self.path, 'tmp', f'{gen:10}.jpg'), imgdata.swapaxes(0, 1))
    return self.screen

  def draw(self, points, surface, zero, color):
    """
    This function draws the surface
    :param data:
    :param surface:
    :param zero:
    :return:
    """
    pixels = self.bs2pixel(points)
    [pg.draw.circle(surface, color, [int(p[0]), int(p[1])], int(2)) for p in pixels]

    # Draw Borders
    border_thickness = 2
    pg.draw.rect(surface, pg.color.THECOLORS['black'], [0, 0, surface.get_width(), border_thickness])
    pg.draw.rect(surface, pg.color.THECOLORS['black'], [0, 0, border_thickness, surface.get_height()])
    pg.draw.rect(surface, pg.color.THECOLORS['black'], [surface.get_width()-border_thickness, 0, border_thickness, surface.get_height()])
    pg.draw.rect(surface, pg.color.THECOLORS['black'], [0, surface.get_height()-border_thickness, surface.get_width(), border_thickness])
    self.screen.blit(surface, zero)

  def bs2pixel(self, points):
    """
    This function transforms from bs to pixels
    :param points:
    :return:
    """
    return np.array(self.surface_size) * (points - self.bs_min)/self.bs_size

if __name__ == '__main__':
  parser = argparse.ArgumentParser('Run archive eval script')
  parser.add_argument('-p', '--path', help='Path of experiment')
  parser.add_argument('-s', '--save', help="Save video", action='store_true')
  parser.add_argument('-hist', '--history', help="Shows population and offsprings history", action='store_true')
  parser.add_argument('-fr', '--frame_rate', help='Frame rate', default=10)

  args = parser.parse_args()

  path = args.path
  save_video = args.save
  history = args.history
  frame_rate = args.frame_rate
  # history = True
  # save_video = True

  # path = '/home/giuseppe/src/cmans/experiment_data/HardMaze_ME_std/2021_02_12_16:31_605524/'
  # path = '/home/giuseppe/src/cmans/experiment_data/HardMaze_NS_std/2021_02_12_11:31_879638/'
  # path = '/home/giuseppe/src/cmans/experiment_data/HardMaze_FitNS_std/2021_02_12_17:14_977596/'
  # path = '/home/giuseppe/src/cmans/experiment_data/Curling_NS_std/2021_02_16_12:20_182118'

  params = parameters.Params()
  params.load(os.path.join(path, '_params.json'))
  env = registered_envs[params.env_name]

  bs_size = np.array(env['grid']['max_coord']) - np.array(env['grid']['min_coord'])

  renderer = Renderer(bs_size=bs_size,
                      bs_min=np.array(env['grid']['min_coord']),
                      save_video=save_video,
                      path=os.path.join(path, 'analyzed_data'),
                      history=history,
                      frame_rate=frame_rate)

  print("Loading data...")
  with open(os.path.join(path, 'analyzed_data/gt_bd.pkl'), 'rb') as f:
    data = pkl.load(f)
  del data['population']['final']

  print("Rendering...")
  renderer.render(data)

  if save_video:
    print("Generating video...")
    stream = ffmpeg.input(os.path.join(path, 'analyzed_data/tmp/*.jpg'), pattern_type='glob', framerate=renderer.TARGET_FPS)
    stream = ffmpeg.output(stream, os.path.join(path, 'analyzed_data/movie.mp4'))
    ffmpeg.run(stream, overwrite_output=True)

    print("Deleting tmp files...")
    os.system('rm -r {}'.format(os.path.join(path, 'analyzed_data/tmp')))
    print("Done.")
