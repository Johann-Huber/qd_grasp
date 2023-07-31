# Created by giuseppe
# Date: 04/06/19

import numpy as np
import gym
import os

def generate_random_states(filepath, filename, samples=50000):
  """
  This function generates random images of possible environment states.
  :param filepath: Path were to save the images
  :param filename: Name of the images collection
  :param samples: Number of images to generate
  :return:
  """
  assert os.path.exists(filepath), "Path {} does not exist.".format(filepath)

  images = []
  env = gym.make("Billiard-v0")
  env.params.RANDOM_BALL_INIT_POSE = True
  print("Generating...")
  for k in range(samples):
    if k%100 == 0 and k>0:
      print("Done {}".format(k))
    env.reset()
    images.append(env.render("rgb_array"))
  print("Done.")
  print("Saving...")
  images = np.stack(images)
  with open(os.path.join(filepath, filename), "wb") as f:
    np.save(f, images)
  print("Done.")
  return images
