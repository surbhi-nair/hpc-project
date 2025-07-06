# src/utils/run_animations.py
import argparse
from pathlib import Path
from animation_core import create_animation, plot_velocity_decay_with_steady_state

'''
python src/utils/run_animations.py --milestone 2
python src/utils/run_animations.py --milestone 3 --test 1
python src/utils/run_animations.py --milestone 3 --test 2
python src/utils/run_animations.py --milestone 3  # both test1 and test2
'''

def run_milestone2_gifs(project_root, duration):
    m2_dir = project_root / "plots" / "m2"
    create_animation(m2_dir, "m2_density_step_", "m2_density.gif", duration)
    create_animation(m2_dir, "m2_velocity_step_", "m2_velocity.gif", duration)

def run_milestone3_gifs(project_root, duration, test=None):
    if test == 1 or test is None:
        test1_dir = project_root / "plots" / "m3_test1"
        create_animation(test1_dir, "density_step_", "m3_test1_density.gif", duration)
        create_animation(test1_dir, "velocity_step_", "m3_test1_velocity.gif", duration)
        plot_velocity_decay_with_steady_state("test1")
    if test == 2 or test is None:
        test2_dir = project_root / "plots" / "m3_test2"
        create_animation(test2_dir, "density_step_", "m3_test2_density.gif", duration)
        create_animation(test2_dir, "velocity_step_", "m3_test2_velocity.gif", duration)
        plot_velocity_decay_with_steady_state("test2")

def run_milestone4_gifs(project_root, duration):
    m4_dir = project_root / "plots" / "m4"
    for omega in [1.00, 1.20, 1.40, 1.60]:
        omega_tag = f"omega_{omega:.2f}"
        omega_dir = m4_dir / omega_tag
        omega_dir.mkdir(parents=True, exist_ok=True)

        create_animation(m4_dir, f"{omega_tag}_density_", f"{omega_tag}/m4_density.gif", duration)
        create_animation(m4_dir, f"{omega_tag}_velocity_", f"{omega_tag}/m4_velocity.gif", duration)

def run_milestone5_gifs(project_root, duration):
    m5_dir = project_root / "plots" / "m5_lid_driven_300x300"
    # create_animation(m5_dir, "velocity_step_", "m5_velocity.gif", duration)
    m5streamline_dir = m5_dir / "streamplots"
    create_animation(m5streamline_dir, "sliding_lid_velocity_field_", "m5_streamlines.gif", duration)

def main():
    parser = argparse.ArgumentParser(description="Generate GIFs for LBM milestone visualizations.")
    parser.add_argument("--milestone", type=int, choices=[2, 3, 4, 5], required=True,
                        help="Specify which milestone's plots to animate (e.g., 2, 3, 4, or 5)")
    parser.add_argument("--test", type=int, choices=[1, 2],
                        help="(Milestone 3 only) Specify which test's plots to animate (1 or 2). Omit to generate both.")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Frame duration in seconds (default: 2.0)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    if args.milestone == 2:
        run_milestone2_gifs(project_root, args.duration)
    elif args.milestone == 3:
        run_milestone3_gifs(project_root, args.duration, args.test)
    elif args.milestone == 4:
        run_milestone4_gifs(project_root, args.duration)
    elif args.milestone == 5:
        run_milestone5_gifs(project_root, args.duration)

if __name__ == "__main__":
    main()