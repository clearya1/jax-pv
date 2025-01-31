# jax-pv: Point Vortices in JAX

Authors: Andrew Cleary, Jacob Page

jax-pv is a fully differentable point vortex solver, for both unbounded and rotating disc-bounded domains, implemented in [JAX](https://github.com/google/jax).

For full details about jax-pv, read our paper [Exploring the free energy landscape of a rotating superfluid](https://pubs.aip.org/aip/cha/article/33/10/103123/2916703/Exploring-the-free-energy-landscape-of-a-rotating).

## Getting Started

There are currently 3 demo scripts in this repository, each of which outlines how to use jax-pv to answer a different research question:
- [Search for relative equilibria in an unbounded domain]()
- [Search for energy-minimising pathways between relative equilibria]()
- [Search for homoclinic connections between relative equilibira]()

## Organization

As well as the above demo scripts, jax-pv contains the following:

- `data`: folder containing the relative equilibria we computed for various numbers of vortices.
- `arnoldi.py` : Code to implement the arnoldi iteration in the Newton-GMRES-hookstep.
- `dneb.py` : Code to implement the (Doubly) Nudged Elastic Band method.
- `hungarian.py` : Code to implement the hungarian algorithm.
- `loss_functions.py` : A script with useful loss functions to find relative equilibria and connections.
- `newton.py` : Code to implement the Newton-GMRES-hookstep method to find relative equilibria.
- `newton_upo.py` : Code to implement the Newton-GMRES-hookstep method to find relative periodic orbits.
- `stability.py` : Code to compute the linear stability of relative equilibria.
- `stability_rpo.py` : Code to compute the linear stability of relative periodic orbits.
- `utils.py` : Generic, useful functions for manipulating systems of point vortices.
- `velocity_transforms.py` : Code to convert from Cartesian to angular velocities.
- `VorticesMotion.py` : Contains the Vortices class, the fully differentiable point vortex solver in the unbounded domain.
- `VorticesMotionDisc.py` : Contains the VorticesDisc class, the fully differentiable point vortex solver in the rotating, disc-bounded domain.
- `VorticesMotionPeriodic.py` : Contains the PeriodicVortices class, the fully differentiable point vortex solver in the doubly periodic domain.

## Citation

If you use jax-pv, please cite: Cleary, Andrew & Page, Jacob 2023 Exploring the free-energy landscape of a rotating superfluid. Chaos: An Interdisciplinary Journal of Nonlinear Science 33 (10), 103123
