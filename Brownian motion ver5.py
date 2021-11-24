import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
import sympy

plt.rcParams[
    'animation.ffmpeg_path'] = \
    r'C:\Users\okamoto_k\ffmpeg-20200628-4cfcfb3-win64-static\bin/ffmpeg'


class Particle:
    """A class representing a two-dimensional particle."""

    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):
        """Initialize the particle's position, velocity, and radius.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor.

        """

        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius

        self.styles = styles
        if not self.styles:
            # Default circle styles
            self.styles = {'edgecolor': 'b', 'fill': False}

    # For convenience, map the components of the particle's position and
    # velocity vector onto the attributes x, y, vx and vy.
    @property
    def x(self):
        return self.r[0]

    @x.setter
    def x(self, value):
        self.r[0] = value

    @property
    def y(self):
        return self.r[1]

    @y.setter
    def y(self, value):
        self.r[1] = value

    @property
    def vx(self):
        return self.v[0]

    @vx.setter
    def vx(self, value):
        self.v[0] = value

    @property
    def vy(self):
        return self.v[1]

    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlaps(self, other):
        """Does the circle of this Particle overlap that of other?"""

        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def draw(self, ax0):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=tuple(self.r), radius=self.radius, **self.styles)
        ax0.add_patch(circle)
        return circle

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""

        self.r += self.v * dt

        # Make the Particles bounce off the walls
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        if self.x + self.radius > 1:
            self.x = 1 - self.radius
            self.vx = -self.vx
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        if self.y + self.radius > 1:
            self.y = 1 - self.radius
            self.vy = -self.vy


class Simulation:
    """A class for a simple hard-circle molecular dynamics simulation.

    The simulation is carried out on a square domain: 0 <= x < 1, 0 <= y < 1.

    """

    def __init__(self, n, radius=0.01, styles=None):
        """Initialize the simulation with n Particles with radii radius.

        radius can be a single value or a sequence with n values.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor when drawing
        the Particles.c

        """
        # this comment is necessary to suppress an unnecessary PyCharm warning
        # noinspection PyTypeChecker
        self.init_particles(n, radius, styles)

    def init_particles(self, n, radius, styles=None):
        """Initialize the n Particles of the simulation.

        Positions and velocities are chosen randomly; radius can be a single
        value or a sequence with n values.

        """

        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            # r isn't iterable: turn it into a generator that returns the
            # same value n times.
            def r_gen(n, radius):
                for i in range(n):
                    yield radius

            radius = r_gen(n, radius)

        self.n = n
        self.tim = []
        self.num = []
        self.i = 50
        self.particles = []
        for i, rad in enumerate(radius):
            # Try to find a random initial position for this particle.
            while True:
                # Choose x, y so that the Particle is entirely inside the
                # domain of the simulation.
                x, y = rad + (1 - 2 * rad) * np.random.random(2)
                # Choose a random velocity (within sdome reasonable range of
                # values) for the Particle.
                vr = v_mean * np.random.normal(loc=1, scale=0.30)
                vphi = 2 * np.pi * np.random.random()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                particle = Particle(x, y, vx, vy, rad, styles)
                # Check that the Particle doesn't overlap one that's already
                # been placed.
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    self.particles.append(particle)
                    break

    def handle_collisions(self):
        """Detect and handle any collisions between the Particles.

        When two Particles collide, they do so elastically: their velocities
        change such that both energy and momentum are conserved.

        """

        def change_velocities(p1, p2):
            """
            Particles p1 and p2 have collided elastically: update their
            velocities.

            """

            m1, m2 = p1.radius ** 2, p2.radius ** 2
            m = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2) ** 2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2 * m2 / m * np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
            u2 = v2 - 2 * m1 / m * np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
            p1.v = (u1 + u2) / 2
            p2.v = (u1 + u2) / 2

        # We're going to need a sequence of all of the pairs of particles when
        # we are detecting collisions. combinations generates pairs of indexes
        # into the self.particles list of Particles on the fly.
        # self.particles.pop(0)
        # self.n = self.n-1
        pairs = combinations(range(self.n), 2)
        comb = []
        for i, j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                comb.append(j)
                change_velocities(self.particles[i], self.particles[j])
        for k in sorted(list(set(comb)), reverse=True):
            self.particles.pop(k)
            self.circles.pop(k)
            self.ax0.patches.pop(k)
            self.n = self.n - 1

    def advance_animation(self, dt):
        """Advance the animation by dt, returning the updated Circles list."""

        for i, p in enumerate(self.particles):
            p.advance(dt)
            self.circles[i].center = p.r
        self.handle_collisions()
        return self.circles, self.n

    def advance(self, dt):
        """Advance the animation by dt."""
        for i, p in enumerate(self.particles):
            p.advance(dt)
        self.handle_collisions()

    def n_model(self):
        x = sympy.Symbol('x')
        y = sympy.Symbol('y')
        a = 4 * np.round(np.sqrt(2), decimals=1) * v_mean * radii
        self.expr1 = nparticles * (1 / 2) ** (a * y * x) - y
        self.expr2 = sympy.solve(self.expr1, y)
        self.modline = [], []
        for k in range(n_frames):
            if k == 0:
                self.modline[1].append(nparticles)
            else:
                self.modline[1].append(self.expr2[0].subs([(x, k)]))
            self.modline[0].append(k)
        return self.expr1, self.expr2, self.modline

    def init(self):
        """Initialize the Matplotlib animation."""

        self.circles = []
        for particle in self.particles:
            self.circles.append(particle.draw(self.ax0))
        self.numtim_text.set_text('')
        self.line.set_data([], [])
        return self.circles, self.numtim_text, self.line

    def animate(self, i):
        """The function passed to Matplotlib's FuncAnimation routine."""
        if i == 0:
            self.advance_animation(0)
        else:
            self.advance_animation(1)

        self.numtim_text.set_text(
            'frame = {:d}'.format(i) + '   N = {:d}'.format(self.n))
        self.tim.append(i)
        self.num.append(self.n)
        self.line.set_data(self.tim, self.num)
        if i > 50:
            self.i = i
        self.ax1.set_xlim(0, self.i)
        return self.circles, self.numtim_text, self.line, self.i

    def do_animation(self, save=False):
        """Set up and carry out the animation of the molecular dynamics.

        To save the animation as a MP4 movie, set save=True.
        """

        fig, (self.ax0, self.ax1) = plt.subplots(nrows=2, figsize=(8, 8))
        fig.suptitle('TD Simulation _ coalescence of dislocations',
                     fontsize=20)
        fig.text(0.7, 0.85, 'Parameters\n'
                 + 'initial N [1/mm2] = {:d}\n'.format(nparticles)
                 + 'radius [um] = {:.0f}\n'.format(1000 * radii)
                 + 'v_mean [um/frame] = {:.0f}'.format(1000 * v_mean),
                 ha='left', va='top')

        for s in ['top', 'bottom', 'left', 'right']:
            self.ax0.spines[s].set_linewidth(2)
        self.ax0.set_aspect('equal', 'box')
        self.ax0.set_xlim(0, 1)
        self.ax0.set_ylim(0, 1)
        self.ax0.xaxis.set_ticks([])
        self.ax0.yaxis.set_ticks([])
        self.ax0.set_xlabel('1mm')
        self.ax0.set_ylabel('1mm')
        self.numtim_text = self.ax0.text(0.02, 0.95, '',
                                         transform=self.ax0.transAxes)
        self.ax0.set_title('moving dislocations and coalescence')

        self.ax1.set_aspect("auto")
        self.ax1.set_xlim(0, self.i)
        self.ax1.set_ylim(0, nparticles + 10)
        self.ax1.set_xlabel('frame')
        self.ax1.set_ylabel('N')
        self.ax1.grid(True, linestyle=':')
        self.line, = self.ax1.plot([], [], lw=2, label='Simulation')
        self.n_model()
        self.ax1.plot(self.modline[0], self.modline[1],
                      label='Model [ '
                            + r'$N = N_0*(\frac{1}{2})^{4\sqrt{2}vrNx}$'
                            + ' -->\n'
                            + '       N = {} ]'.format(self.expr2[0]))
        self.ax1.set_title('N of dislocations vs frame')
        self.ax1.legend(shadow=True, fancybox=True)

        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init,
                                       frames=n_frames, interval=5, blit=False)
        if save:
            FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec',
                                                                  'libx264'])
            anim.save(r'C:\Users\okamoto_k\Videos\Brownian motion ver5.mp4',
                      writer=FFwriter)
        else:
            plt.show()


if __name__ == '__main__':
    nparticles = 1000
    radii = 0.01
    v_mean = 0.005
    n_frames = 500
    styles = {'edgecolor': 'C0', 'linewidth': 0, 'fill': 'C0'}
    sim = Simulation(nparticles, radii, styles)
    sim.do_animation(save=False)
