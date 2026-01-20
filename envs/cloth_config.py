"""Cloth configuration and XML generation for MuJoCo flexcomp cloth simulation."""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple


@dataclass
class ClothConfig:
    """Configuration for MuJoCo flexcomp cloth simulation.

    Attributes:
        count_x: Number of vertices along X axis
        count_y: Number of vertices along Y axis
        spacing: Distance between vertices in meters
        mass: Total cloth mass in kg
        radius: Collision radius per vertex in meters
        young: Young's modulus in Pa (0 to disable elasticity plugin)
        poisson: Poisson's ratio (0-0.5)
        thickness: Cloth thickness in meters
        edge_equality: Whether to use equality constraints on edges
        edge_damping: Edge damping coefficient
        self_collide: Self-collision mode ("none", "auto", "bvh", "sap", "narrow")
        contact_condim: Contact constraint dimensionality
        contact_solref: Solver reference parameters (timeconst, dampratio)
        rgba: Color as (r, g, b, a) tuple
        timestep: Simulation timestep in seconds
        iterations: Solver iterations
    """

    # Mesh resolution
    count_x: int = 15
    count_y: int = 15
    spacing: float = 0.025

    # Physical properties
    mass: float = 0.15
    radius: float = 0.002

    # Elasticity (shell plugin) - set young=0 to disable
    young: float = 5e4
    poisson: float = 0.2
    thickness: float = 2e-3

    # Edge constraints
    edge_equality: bool = True
    edge_damping: float = 0.05

    # Contact
    self_collide: str = "none"
    contact_condim: int = 3
    contact_solref: Tuple[float, float] = (0.005, 1.0)

    # Visual
    rgba: Tuple[float, float, float, float] = (0.9, 0.3, 0.3, 1.0)

    # Solver
    timestep: float = 0.002
    iterations: int = 50

    @property
    def cloth_size(self) -> Tuple[float, float]:
        """Returns (width, height) in meters."""
        return ((self.count_x - 1) * self.spacing, (self.count_y - 1) * self.spacing)

    @property
    def vertex_count(self) -> int:
        """Total number of vertices."""
        return self.count_x * self.count_y


# Preset configurations
CLOTH_PRESETS: Dict[str, ClothConfig] = {
    "fast": ClothConfig(
        count_x=9,
        count_y=9,
        spacing=0.0375,
        mass=0.2,
        radius=0.002,
        young=3e4,
        poisson=0.15,
        thickness=2e-3,
        edge_equality=True,
        edge_damping=0.1,
        self_collide="none",
        iterations=30,
    ),
    "medium": ClothConfig(
        count_x=15,
        count_y=15,
        spacing=0.025,
        mass=0.15,
        radius=0.001,  # Reduced from 0.002 to minimize phantom collisions
        young=2e4,           # Reduced from 5e4 - softer cloth, less spring-back
        poisson=0.2,
        thickness=2e-3,
        edge_equality=True,
        edge_damping=0.15,   # Increased from 0.05 - more damping for stability
        self_collide="none",
        iterations=75,       # Increased from 50 - better constraint satisfaction
    ),
    "realistic": ClothConfig(
        count_x=21,
        count_y=21,
        spacing=0.018,
        mass=0.12,
        radius=0.0015,
        young=8e4,
        poisson=0.25,
        thickness=1.5e-3,
        edge_equality=True,
        edge_damping=0.03,
        self_collide="auto",  # Enable self-collision with auto mode selection
        iterations=100,
    ),
    "legacy": ClothConfig(
        # Matches original cloth_flex.xml exactly (no elasticity)
        count_x=9,
        count_y=9,
        spacing=0.0375,
        mass=0.2,
        radius=0.001,
        young=0,  # No elasticity plugin
        poisson=0,
        thickness=0,
        edge_equality=False,
        edge_damping=0,
        self_collide="none",
        contact_condim=3,
        contact_solref=(0.005, 1.0),
        iterations=50,
    ),
    "piper": ClothConfig(
        # Narrower cloth for Piper robot with limited reach
        # Robot bases at Y=±0.18, can reach ~0.04m inward → Y=±0.14
        # Cloth width = (count_y-1) * spacing = 12 * 0.025 = 0.30m → corners at Y=±0.15
        count_x=11,  # Shorter in X (folding direction)
        count_y=13,  # Narrower for Piper reachability
        spacing=0.025,  # Same density as medium
        mass=0.12,
        radius=0.001,
        young=2e4,
        poisson=0.2,
        thickness=2e-3,
        edge_equality=True,
        edge_damping=0.15,
        self_collide="none",
        iterations=75,
    ),
}


def generate_cloth_xml(
    config: ClothConfig,
    base_pos: Tuple[float, float, float] = (0, 0, 0.81),
) -> str:
    """Generate cloth XML string from configuration.

    Args:
        config: ClothConfig instance with cloth parameters
        base_pos: Initial position (x, y, z) for the cloth center

    Returns:
        Valid MuJoCo XML string for the cloth model

    Note:
        As of MuJoCo 3.3.3, the shell/membrane plugins have been integrated into
        the engine. The elasticity is now controlled via the <elasticity> element
        and the elastic2d attribute on flexcomp.
    """
    use_elasticity = config.young > 0

    lines = [
        '<mujoco model="cloth_flex">',
        '    <compiler angle="radian"/>',
        f'    <option timestep="{config.timestep}" integrator="implicitfast" '
        f'solver="CG" iterations="{config.iterations}" tolerance="1e-6"/>',
    ]

    # Asset section
    rgba_str = f"{config.rgba[0]} {config.rgba[1]} {config.rgba[2]} {config.rgba[3]}"
    lines.extend([
        '',
        '    <asset>',
        '        <texture type="2d" name="cloth_tex" builtin="flat" '
        'rgb1="0.8 0.2 0.2" width="64" height="64"/>',
        f'        <material name="cloth_mat" texture="cloth_tex" rgba="{rgba_str}" '
        'specular="0.1"/>',
        '    </asset>',
    ])

    # Worldbody section
    pos_str = f"{base_pos[0]} {base_pos[1]} {base_pos[2]}"
    count_str = f"{config.count_x} {config.count_y} 1"
    spacing_str = f"{config.spacing} {config.spacing} 0.01"

    # Build flexcomp attributes - add elastic2d if using elasticity
    flexcomp_attrs = [
        f'name="cloth" type="grid" dim="2" count="{count_str}"',
        f'spacing="{spacing_str}" pos="{pos_str}"',
        f'material="cloth_mat" rgba="{rgba_str}"',
        f'radius="{config.radius}" mass="{config.mass}"',
    ]

    lines.extend([
        '',
        '    <worldbody>',
        '        <body name="cloth_root" pos="0 0 0">',
        f'            <flexcomp {flexcomp_attrs[0]}',
        f'                      {flexcomp_attrs[1]}',
        f'                      {flexcomp_attrs[2]}',
        f'                      {flexcomp_attrs[3]}>',
    ])

    # Contact element
    solref_str = f"{config.contact_solref[0]} {config.contact_solref[1]}"
    if config.self_collide != "none":
        lines.append(
            f'                <contact selfcollide="{config.self_collide}" '
            f'condim="{config.contact_condim}" solref="{solref_str}" '
            'contype="1" conaffinity="1"/>'
        )
    else:
        lines.append(
            f'                <contact condim="{config.contact_condim}" '
            f'solref="{solref_str}" contype="1" conaffinity="1"/>'
        )

    # Edge element with damping (stiffness only available for dim=1, not 2D cloth)
    edge_parts = []
    if config.edge_equality:
        edge_parts.append('equality="true"')
    if config.edge_damping > 0:
        edge_parts.append(f'damping="{config.edge_damping}"')
    if edge_parts:
        lines.append(f'                <edge {" ".join(edge_parts)}/>')

    # Elasticity element (replaces shell plugin in MuJoCo >= 3.2.4)
    if use_elasticity:
        lines.append(
            f'                <elasticity young="{config.young}" '
            f'poisson="{config.poisson}" thickness="{config.thickness}"/>'
        )

    # Close elements
    lines.extend([
        '            </flexcomp>',
        '        </body>',
        '    </worldbody>',
        '</mujoco>',
    ])

    return '\n'.join(lines)


def get_cloth_config(preset: str = "medium", **overrides) -> ClothConfig:
    """Get a cloth configuration by preset name with optional overrides.

    Args:
        preset: Preset name ("fast", "medium", "realistic", "legacy")
        **overrides: Individual parameters to override from the preset

    Returns:
        ClothConfig instance

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset not in CLOTH_PRESETS:
        available = ", ".join(CLOTH_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")

    config = CLOTH_PRESETS[preset]

    if overrides:
        config = replace(config, **overrides)

    return config
