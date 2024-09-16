#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simulation of heat conduction in a SUS304 wire

Usage:
    simulation.py [-D=<D>] [-L=<L>] [-v=<v>] [-U=<U>] [--Nx=<Nx>] [--dt=<dt>] [--Tf=<Tf>]
    simulation.py -h | --help

Options:
    -D, --diameter <D>  Diameter of the wire [default: 1.2] (mm)
    -L, --length <L>    Length of the wire [default: 6.0] (mm)
    -v, --velocity <v>  Velocity of the wire [default: 30.0] (mm/s)
    -U, --voltage <U>   Voltage applied to the wire [default: 1.45] (V)

    --Nx <Nx>    Number of nodes [default: 1000]
    --dt <dt>    Time step size [default: 0.001] (s)
    --Tf <Tf>    Final time [default: 1.0] (s)

    -h, --help    Show this screen
"""


import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0

from docopt import docopt

# Material properties of SUS304
rho = 7.93e3    # Density of steel [kg/m^3]
Cp = 550.0      # Specific heat capacity of steel [J/(kg K)]
lamb = 16.2     # Thermal conductivity of steel [W/(m K)]
sigma = lambda T: 1.0 / (7.16e-7 + 6.23e-10*T)  # Electrical conductivity of steel [S/m]
# Environmental conditions
T0 = 20.0       # Initial temperature [C]

# source term of the heat equation
def source_term(U, L, T):
    return (U/L)**2 / (rho*Cp) * sigma(T)

def simulation(D, L, v, U, Nx, dt, Tf):
    sec_area = np.pi * (D/2)**2
    # Build mesh
    dx = L / (Nx - 1)
    mesh = np.linspace(0, L, Nx)

    # Create a new sparse PETSc matrix, fill it then assemble it
    A = PETSc.Mat().createAIJ([Nx, Nx])
    A.setUp()

    kappa = lamb / (rho*Cp)
    diag_entries = 1.0 + 2.0 * kappa * dt / dx**2
    nofd_entries = -kappa * dt / dx**2 - v * dt / dx / 2.0
    pofd_entries = -kappa * dt / dx**2 + v * dt / dx / 2.0

    A.setValues(0, 0, 1.0)
    A.setValues(Nx-1, Nx-1, 1.0)
    for i in range(1, Nx-1):
        A.setValues(i, i-1, nofd_entries)
        A.setValues(i, i  , diag_entries)
        A.setValues(i, i+1, pofd_entries)

    A.assemble()

    # Define the initial condition
    init_cond = np.zeros(Nx) + T0

    # Create a new vector, fill it with the initial condition
    x = PETSc.Vec().createSeq(Nx)
    b = PETSc.Vec().createSeq(Nx)
    b.setArray(init_cond)

    # Instantiate the KSP solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    print("Solving the system with the %s solver" % ksp.getType())

    # Time-stepping loop
    fig, ax = plt.subplots(figsize=(8, 6))
    set_plot(ax, L=int(L*1e3))
    plt.title('Temperature distribution in a SUS304 wire', fontsize=12)
    ax.plot(init_cond + 273.15, mesh * 1e3, 'k-', lw=2, label='Initial condition')
    for t in range(int(Tf/dt)):
        # Solve the linear system
        ksp.solve(b, x)

        # Update the solution
        current_sol = x.getArray()
        source = source_term(U, L, current_sol)
        b.setArray(current_sol + source * dt)
        b.setValue(0, T0)
        b.setValue(Nx-1, T0)

        # Plot the solution
        ax.plot(current_sol + 273.15, mesh * 1e3, 'r-', lw=2, alpha=0.5)

    plt.tight_layout()
    plt.show()

def set_plot(ax, L=6):
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(which='major', length=6, width=1, direction='out', top=False, right=False)
    ax.tick_params(which='minor', length=4, width=1, direction='out', top=False, right=False)
    ax.set_xlim(273.15, 2400)
    ax.set_ylim(0, L)
    ax.set_xlabel('Temperature(K)')
    ax.set_ylabel('Position(mm)', rotation='vertical')

def main():
    # Parse the command-line arguments
    args = docopt(__doc__)
    D = float(args['--diameter']) * 1e-3
    L = float(args['--length']) * 1e-3
    v = float(args['--velocity']) * -1e-3
    U = float(args['--voltage'])
    Nx = int(args['--Nx'])
    dt = float(args['--dt'])
    Tf = float(args['--Tf'])

    # Run the simulation
    simulation(D, L, v, U, Nx, dt, Tf)


if __name__ == '__main__':
    main()

