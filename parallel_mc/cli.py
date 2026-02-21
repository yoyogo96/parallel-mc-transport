"""
CLI entry point for parallel Monte Carlo neutron transport.

Usage:
    python -m parallel_mc --quick                    # Quick test (10k particles, 100 batches)
    python -m parallel_mc --production               # Production run (50k particles, 300 batches)
    python -m parallel_mc --backend cpu --particles 20000 --batches 200
    python -m parallel_mc --list-backends            # Show available backends
    python -m parallel_mc --validate                 # Compare against OpenMC reference
"""
import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Monte Carlo Neutron Transport for 100 MWth Marine MCFR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m parallel_mc --quick                    Quick test run
  python -m parallel_mc --production               Full production run
  python -m parallel_mc --backend metal --particles 50000
  python -m parallel_mc --list-backends            Show available backends
  python -m parallel_mc --validate                 Compare with OpenMC reference
        """,
    )

    parser.add_argument('--quick', action='store_true', help='Quick test: 10k particles, 100 batches')
    parser.add_argument('--production', action='store_true', help='Production: 50k particles, 300 batches')
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'metal', 'auto'], default='auto',
                        help='Backend selection (default: auto-detect)')
    parser.add_argument('--particles', '-n', type=int, default=None, help='Particles per batch')
    parser.add_argument('--batches', '-b', type=int, default=None, help='Total batches')
    parser.add_argument('--inactive', type=int, default=None, help='Inactive batches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON file')
    parser.add_argument('--list-backends', action='store_true', help='List available backends')
    parser.add_argument('--validate', action='store_true', help='Run validation against OpenMC')
    parser.add_argument('--reflector', choices=['beo', 'ss316h'], default='beo',
                        help='Reflector material (default: beo for OpenMC validation)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # List backends
    if args.list_backends:
        from .backends import list_backends
        print("Available backends:")
        print(f"  {'Name':<10} {'Description':<40} {'Available'}")
        print(f"  {'-'*10} {'-'*40} {'-'*9}")
        for name, desc, avail in list_backends():
            status = "YES" if avail else "NO"
            print(f"  {name:<10} {desc:<40} {status}")
        return 0

    # Validate mode
    if args.validate:
        from .validation.compare import run_validation
        return run_validation(
            backend_name=args.backend,
            n_particles=args.particles or 20000,
            n_batches=args.batches or 150,
            seed=args.seed,
        )

    # Determine run parameters
    if args.quick:
        n_particles = args.particles or 10000
        n_batches = args.batches or 100
        n_inactive = args.inactive or 20
    elif args.production:
        n_particles = args.particles or 50000
        n_batches = args.batches or 300
        n_inactive = args.inactive or 50
    else:
        n_particles = args.particles or 20000
        n_batches = args.batches or 150
        n_inactive = args.inactive or 30

    # Select backend
    from .backends import auto_select_backend, get_backend
    if args.backend == 'auto':
        backend = auto_select_backend()
    else:
        backend = get_backend(args.backend)

    # Build materials
    from .materials import build_fuel_salt, build_reflector_beo, build_reflector_ss316h
    from .geometry import MCFRGeometry

    fuel = build_fuel_salt()
    if args.reflector == 'beo':
        refl = build_reflector_beo()
    else:
        refl = build_reflector_ss316h()
    materials = {0: fuel, 1: refl}
    geometry = MCFRGeometry()

    # Run
    from .eigenvalue import PowerIteration

    solver = PowerIteration(
        backend=backend,
        geometry=geometry,
        materials=materials,
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=n_inactive,
        seed=args.seed,
    )

    result = solver.solve(verbose=not args.quiet)

    # Save output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Default output path
        import os
        output_path = os.path.join(os.getcwd(), 'results', 'parallel_mc_results.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        if not args.quiet:
            print(f"\nResults saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
