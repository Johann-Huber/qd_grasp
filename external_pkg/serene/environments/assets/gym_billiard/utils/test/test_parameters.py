from gym_billiard.utils import physics
import Box2D as b2

def test_create_table():
  physics_eng = physics.PhysicsSim()
  assert len(physics_eng.walls) == 4, 'Not all the walls have been created'
  for wall in physics_eng.walls:
    assert wall.active == True, 'Wall {} not active'.format(wall.userData['name'])
    assert wall.angle == 0.0, 'Wall {} angle not 0.0'.format(wall.userData['name'])
    assert wall.awake == True, 'Wall {} not awake'.format(wall.userData['name'])
    assert wall.inertia == 0.0, 'Wall {} inertia not 0.0. This means wall is Dynamic.'.format(wall.userData['name'])
  del physics_eng

def test_create_ball():
  physics_eng = physics.PhysicsSim(balls_pose=[[1, 2]])
  assert len(physics_eng.balls) == 1, 'Not enough balls'
  assert physics_eng.balls[0].position + physics_eng.wt_transform == b2.b2Vec2([1, 2]), 'Wrong ball pose'


