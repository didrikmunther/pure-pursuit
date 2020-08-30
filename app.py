#!./env/bin/python3

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

class Car:
	wheelbase = 1 
	velocity = 3
	dt = 0.1
	state = np.array([0, 0, np.pi / 2])  # (x, y, yaw)

	steering_angle = 0

	def __init__(self, pos):
		self.state[0] = pos[0]
		self.state[1] = pos[1]

	def next(self):
		u = np.array(
			[self.velocity * self.dt * np.cos(self.state[2]),
			self.velocity * self.dt * np.sin(self.state[2]),
			self.velocity * self.dt * np.tan(self.steering_angle) / self.wheelbase]
		)

		# print('New yaw: ', u[2] / 2 / np.pi * 360, self.steering_angle)

		self.state = self.state + u

	def pursuit(self, coord):
		print('New coord received: ', coord)

		x = coord[0] - self.state[0]
		y = coord[1] - self.state[1]

		# print('Relative coords: ', [xg, yg])

		xr = x * np.cos(np.pi / 2 - self.state[2]) - y * np.sin(np.pi / 2 - self.state[2])
		yr = y * np.cos(np.pi / 2 - self.state[2]) + x * np.sin(np.pi / 2 - self.state[2])

		# print('Rotated: ', [x, y])

		# alpha = (np.arctan2(y, x) - self.state[2])
		# self.steering_angle = np.arctan2(2*self.wheelbase*np.sin(alpha), np.linalg.norm([x, y]))

		l = np.sqrt( xr**2 + yr**2 )
		r = l**2 / (2 * xr)
		h = self.wheelbase

		self.steering_angle = -np.arctan2(h, r)

		print('R, L:', r, l)
		print('Steering:', self.steering_angle / 2 / np.pi * 360, self.steering_angle)
		print('Yaw:', self.state[2] / 2 / np.pi * 360, self.state[2])

	def relative(self, coord):
		xg = coord[0] - self.state[0]
		yg = coord[1] - self.state[1]
		return [xg, yg]

def line_intersections(cx, cy, r, point1, point2):
	dx = point2[0] - point1[0]
	dy = point2[1] - point1[1]

	A = dx * dx + dy * dy
	B = 2 * ( dx * ( point1[0] - cx ) + dy * ( point1[1] - cy ) )
	C = (point1[0] - cx) ** 2 + (point1[1] - cy) ** 2 - r ** 2

	det = B ** 2 - 4 * A * C
	if A <= .000001 or det < 0:
		return np.array([])
	elif det == 0:
		t = -B / (2 * A)
		return np.array([point1[0] + t * dx, point1[1] + t * dy])
	else:
		t = (-B + np.sqrt(det)) / (2 * A)
		a = [point1[0] + t * dx, point1[1] + t * dy]
		t = (-B - np.sqrt(det)) / (2 * A)
		b = [point1[0] + t * dx, point1[1] + t * dy]
		return np.array([a, b])

with open('path.json', 'r') as f:
    path = json.load(f)

x = [x[0] for x in path]
y = [x[1] for x in path]

fig, ax = plt.subplots()
track, = ax.plot(x, y)
line, = ax.plot([], [], color='k')

circle = Circle((0, 0), 0, fill=False)
ax.add_artist(circle)

horizon = ax.scatter(0, 0)

xa = []
ya = []

car = Car(path[0])

lookahead = 4

def update_pursuit(coord):
	horizon.set_offsets(coord)
	car.pursuit(coord)

prev_pointi = 1

while True:
	xa.append(car.state[0])
	ya.append(car.state[1])
	car.next()

	car_pos = np.array([car.state[0], car.state[1]])

	collided_points = []
	possible_coords = []
	for pointi in range(prev_pointi, len(path)):
		prev = np.array(path[pointi-1])
		current = np.array(path[pointi])
		coord_dist = np.linalg.norm(prev - current)

		# all points inside circle
		collision = sum([ (1 if x <= lookahead else 0) for x in [np.linalg.norm( x - car_pos ) for x in [prev, current]]])
		if collision >= 2:
			collided_points.append(pointi)
			continue

		positions = line_intersections(car_pos[0], car_pos[1], lookahead, prev, current)

		if len(positions) >= 2:
			collided_points.append(pointi)
			real_positions = []

			# only allow positions on the line segment
			for position in positions:
				if np.isclose([np.linalg.norm(prev - position) + np.linalg.norm(current - position)], [coord_dist], rtol=.001)[0]:
					real_positions.append(position)

			# pick the point closest to the current coord
			if len(real_positions) >= 2:
				possible_coords.append(sorted(real_positions, key=lambda x: np.linalg.norm(x - current))[0])
				
			elif len(real_positions) >= 1:
				possible_coords.append(real_positions[0])

		elif len(positions) >= 1:
			collided_points.append(pointi)
			possible_coords.append(positions[0])
		elif len(possible_coords) > 0:
			break
			
	prev_pointi = collided_points[0] if len(collided_points) > 0 else 1

	plane = np.array([np.cos(car.state[2]), np.sin(car.state[2])])
	best_coords = sorted(possible_coords, key=lambda x: -plane.dot(np.array(x) - plane))
	
	if len(best_coords) >= 1:
		update_pursuit(best_coords[0])

	print('-----------------------')

	plt.pause(0.0000000001)
	line.set_data(xa, ya)

	circle.center = car.state[0], car.state[1]
	circle.set_radius(lookahead)

	# newpath = list(map(lambda x: [
	# 	(x[0] - car.state[0]) * np.cos(np.pi / 2 - car.state[2]) - (x[1] - car.state[1]) * np.sin(np.pi / 2 - car.state[2]) - car.state[0],
	# 	(x[1] - car.state[1]) * np.cos(np.pi / 2 - car.state[2]) + (x[0] - car.state[0]) * np.sin(np.pi / 2 - car.state[2]) - car.state[1]
	# ], path))
	# track.set_data([x[0] for x in newpath], [x[1] for x in newpath])

plt.show()