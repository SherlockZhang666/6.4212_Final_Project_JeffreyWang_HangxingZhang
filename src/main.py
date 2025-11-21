# src/main.py

import numpy as np

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SpatialInertia,
    UnitInertia,
    CoulombFriction,
    Box,
    MultibodyPlant,
    StartMeshcat,
    MeshcatVisualizer,
    Simulator,
)

from pydrake.geometry import (
    ProximityProperties,
    AddRigidHydroelasticProperties,
    AddContactMaterial,
)

# 6.4212 helper that configures the parser's package map, etc.
from manipulation.utils import ConfigureParser


def add_table(plant: MultibodyPlant):
    """
    Adds a rigid table in front of the robot.
    """
    table_height = 0.75        # meters
    table_thickness = 0.05
    table_size = [0.9, 1.2, table_thickness]  # (x, y, z)

    box = Box(*table_size)

    # Pose of the table frame in world coordinates
    X_WTable = RigidTransform(
        RotationMatrix.Identity(),
        [0.7, 0.0, table_height - table_thickness / 2.0],
    )

    # Rigid body for the table
    table_body = plant.AddRigidBody(
        "table_body",
        SpatialInertia(
            mass=1.0,
            p_PScm_E=[0.0, 0.0, 0.0],
            G_SP_E=UnitInertia.SolidBox(*table_size),
        ),
    )

    # Contact properties (rigid hydroelastic is fine for the table)
    table_prox = ProximityProperties()
    AddContactMaterial(
        0.9,  # elastic modulus scale
        0.4,  # dissipation
        CoulombFriction(0.9, 0.4),
        table_prox,
    )

    # Collision + visual geometry
    plant.RegisterCollisionGeometry(
        table_body,
        RigidTransform.Identity(),  # frame of the body
        box,
        "table_collision",
        table_prox,
    )

    # Brownish color RGBA
    table_color = [0.6, 0.4, 0.2, 1.0]
    plant.RegisterVisualGeometry(
        table_body,
        RigidTransform.Identity(),
        box,
        "table_visual",
        table_color,
    )

    # Weld to the world at X_WTable
    plant.WeldFrames(plant.world_frame(), table_body.body_frame(), X_WTable)


def add_shirt_proxy(plant: MultibodyPlant):
    """
    Adds a thin hydroelastic box to represent a T-shirt laid on the table.
    For now it's a rigid hydroelastic body with a small tilt ("wrinkle")
    to make grasping easier, as recommended in the feedback.
    """
    shirt_length = 0.6   # x-size
    shirt_width = 0.5    # y-size
    shirt_thickness = 0.015  # z-size (thin)

    box = Box(shirt_length, shirt_width, shirt_thickness)

    # Slightly above the table with a tiny tilt to avoid perfect flat contact
    X_WShirt = RigidTransform(
        RollPitchYaw(0.05, 0.0, 0.0),  # small roll to create “wrinkle”
        [0.7, 0.0, 0.75 + shirt_thickness / 2.0 + 0.001],  # table_height = 0.75
    )

    shirt_body = plant.AddRigidBody(
        "shirt_body",
        SpatialInertia(
            mass=0.1,
            p_PScm_E=[0.0, 0.0, 0.0],
            G_SP_E=UnitInertia.SolidBox(
                shirt_length, shirt_width, shirt_thickness
            ),
        ),
    )

    # Hydroelastic contact properties – using rigid hydroelastic
    shirt_prox = ProximityProperties()
    hydro_modulus = 1e5
    resolution_hint = 0.01
    AddRigidHydroelasticProperties(hydro_modulus, resolution_hint, shirt_prox)

    AddContactMaterial(
        0.8,  # elastic modulus scale
        0.3,  # dissipation
        CoulombFriction(0.8, 0.3),
        shirt_prox,
    )

    # Register collision + visual geometry
    plant.RegisterCollisionGeometry(
        shirt_body,
        RigidTransform.Identity(),
        box,
        "shirt_collision",
        shirt_prox,
    )

    shirt_color = [0.2, 0.6, 0.9, 1.0]  # bluish
    plant.RegisterVisualGeometry(
        shirt_body,
        RigidTransform.Identity(),
        box,
        "shirt_visual",
        shirt_color,
    )

    # Free body – not welded to world
    plant.SetDefaultFreeBodyPose(shirt_body, X_WShirt)


def add_iiwa_with_robotiq(plant: MultibodyPlant, parser: Parser):
    """
    Adds an IIWA arm and a Robotiq gripper using Drake's older parsing API:
    AddModelsFromUrl instead of AddModelFromFile.
    """

    # Correct package URLs for the 6.4212 environment
    iiwa_url = (
        "package://manipulation/iiwa_description/urdf/iiwa14_no_collision.urdf"
    )
    robotiq_url = (
        "package://manipulation/robotiq_description/urdf/robotiq_85.urdf"
    )

    # Load IIWA
    iiwa_models = parser.AddModelsFromUrl(iiwa_url)
    iiwa_model = iiwa_models[0]

    # Load Robotiq gripper
    robotiq_models = parser.AddModelsFromUrl(robotiq_url)
    robotiq_model = robotiq_models[0]

    # Weld IIWA base to world
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("iiwa_link_0", iiwa_model),
        RigidTransform.Identity(),
    )

    # Weld Robotiq to IIWA link_7
    plant.WeldFrames(
        plant.GetFrameByName("iiwa_link_7", iiwa_model),
        plant.GetFrameByName("robotiq_85_base_link", robotiq_model),
        RigidTransform.Identity(),
    )

    return iiwa_model, robotiq_model



def make_cloth_station_diagram(time_step: float = 0.002):
    """
    Builds the full diagram: plant + scene graph + iiwa + robotiq + table + shirt,
    plus Meshcat visualization.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)

    parser = Parser(plant)
    ConfigureParser(parser)  # 6.4212 helper to add package paths

    # Add robot + gripper
    iiwa_model, robotiq_model = add_iiwa_with_robotiq(plant, parser)

    # Add table and shirt proxy
    add_table(plant)
    add_shirt_proxy(plant)

    # Finalize the plant
    plant.Finalize()

    # Start Meshcat and visualize
    meshcat = StartMeshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    return diagram, context, meshcat, plant, iiwa_model, robotiq_model


def main():
    diagram, context, meshcat, plant, iiwa_model, robotiq_model = (
        make_cloth_station_diagram()
    )

    # For now, just run the simulation for a few seconds so you can see the stationary scene.
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.AdvanceTo(1.0)  # 1 second is enough just to see it in Meshcat

    print(
        "Cloth station built. Open the Meshcat URL printed above to inspect the IIWA, table, and shirt proxy."
    )


if __name__ == "__main__":
    main()
