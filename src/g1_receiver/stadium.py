"""A regulation field and small school stadium for the Aero spectator replay."""

from __future__ import annotations

import math
from random import Random
import struct
import xml.etree.ElementTree as ET
import zlib

import mujoco

YARD_M = .9144
FIELD_HALF_WIDTH = 80 * YARD_M / 3
# The replay starts at the home 30, on the far hash.
FIELD_CENTER = (20 * YARD_M, -FIELD_HALF_WIDTH / 3)
PAINT = ".92 .94 .91 1"
ALUMINUM = ".59 .65 .69 1"
STEEL = ".32 .39 .44 1"
NAVY = ".075 .17 .27 1"
GOLD = "1 .70 .14 1"
TURF_LIGHT = ".244 .421 .245 1"
TURF_DARK = ".201 .371 .217 1"
TURF_ENDZONE = ".135 .250 .323 1"
STUDIO_BACKGROUND = ".035 .050 .075"

NUMERALS = {
    "0": (((.15, 0), (.85, 0), (1, .15), (1, .85), (.85, 1), (.15, 1), (0, .85), (0, .15), (.15, 0)),),
    "1": (((.13, .76), (.53, 1), (.53, 0)), ((.04, 0), (.97, 0))),
    "2": (((0, .82), (.17, 1), (.83, 1), (1, .83), (.13, .15), (0, 0), (1, 0)),),
    "3": (((0, 1), (.81, 1), (1, .83), (1, .64), (.78, .52),
           (1, .39), (1, .17), (.82, 0), (0, 0)), ((.40, .52), (.78, .52))),
    "4": (((.85, 0), (.85, 1), (0, .30), (1, .30)),),
    "5": (((1, 1), (0, 1), (0, .52), (.83, .52), (1, .36), (1, .17), (.83, 0), (0, 0)),),
}
LETTERS = {
    "A": (((0, 0), (0, .72), (.26, 1), (.74, 1), (1, .72), (1, 0)), ((0, .4), (1, .4))),
    "C": (((1, .94), (.84, 1), (.14, 1), (0, .85), (0, .15), (.14, 0), (.84, 0), (1, .06)),),
    "F": (((0, 0), (0, 1), (1, 1)), ((0, .55), (.78, .55))),
    "L": (((0, 1), (0, 0), (1, 0)),),
    "N": (((0, 0), (0, 1), (1, 0), (1, 1)),),
    "O": NUMERALS["0"],
    "S": (((1, 1), (.17, 1), (0, .83), (0, .66), (.17, .51),
           (.83, .51), (1, .34), (1, .17), (.83, 0), (0, 0)),),
}


def _numbers(values):
    return " ".join(f"{value:.7g}" for value in values)


def _yaw(angle):
    return _numbers((math.cos(angle / 2), 0, 0, math.sin(angle / 2)))


def _geom(parent, shape, rgba, **attributes):
    return ET.SubElement(parent, "geom", type=shape, rgba=rgba, contype="0", conaffinity="0",
                         mass="0", **attributes)


def _box(parent, pos, size, rgba, yaw=0, **attributes):
    _geom(parent, "box", rgba, pos=_numbers(pos), size=_numbers(size), quat=_yaw(yaw), **attributes)


def _pole(parent, start, finish, radius, rgba):
    _geom(parent, "cylinder", rgba, size=str(radius), fromto=_numbers((*start, *finish)))


def _stadium_assets(assets):
    """Create a deterministic fine turf texture without requiring image assets on disk."""
    random = Random(18)
    size = 128
    rows = []
    for _ in range(size):
        row = bytearray([0])  # PNG row filter
        for _ in range(size):
            value = random.randrange(195, 255)
            row.extend((value, min(255, value + 3), max(0, value - 6)))
        rows.append(row)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    texture = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    ET.SubElement(assets, "texture", name="stadium_turf", type="2d", file="stadium-turf.png")
    for name, color in (
        ("light", TURF_LIGHT),
        ("dark", TURF_DARK),
        ("endzone", TURF_ENDZONE),
    ):
        ET.SubElement(assets, "material", name=f"stadium_turf_{name}", texture="stadium_turf",
                      rgba=color, texrepeat="28 150", specular=".025", shininess=".02")
    ET.SubElement(assets, "texture", name="stadium_sky", type="skybox", builtin="gradient",
                  rgb1=".30 .58 .83", rgb2=".82 .90 .94", width="64", height="256")
    return {"stadium-turf.png": texture}


def _field(parent, *, goalposts=True):
    cx, cy = FIELD_CENTER
    half = FIELD_HALF_WIDTH
    goal = 50 * YARD_M
    for segment in range(20):
        x = cx + (-47.5 + 5 * segment) * YARD_M
        shade, color = ("light", TURF_LIGHT) if segment % 2 else ("dark", TURF_DARK)
        _box(parent, (x, cy, -.013), (2.5 * YARD_M, half + .85, .010), color,
             material=f"stadium_turf_{shade}")
    for side in (-1, 1):
        x = cx + side * 55 * YARD_M
        _box(parent, (x, cy, -.013), (5 * YARD_M, half + .85, .010), TURF_ENDZONE,
             material="stadium_turf_endzone")
        _box(parent, (cx, cy + side * half, .001), (60 * YARD_M, .065, .002), PAINT)
        _box(parent, (cx + side * 60 * YARD_M, cy, .001), (.065, half, .002), PAINT)
        _box(parent, (cx + side * goal, cy, .001), (.065, half, .002), PAINT)
        if goalposts:
            _goal(parent, cx + side * 60 * YARD_M, cy, side)

    for yard in range(-45, 50):
        x = cx + yard * YARD_M
        if yard % 5 == 0:
            _box(parent, (x, cy, .001), (.051, half, .002), PAINT)
        else:
            for lateral in (-half / 3, half / 3, -half + .58, half - .58):
                _box(parent, (x, cy + lateral, .001), (.051, .31, .002), PAINT)
        if yard % 10 == 0:
            for sign in (-1, 1):
                y = cy + sign * (half - 6.77)
                _yard_number(parent, str(50 - abs(yard)), x, y, sign)


def _yard_number(parent, text, x, y, sign):
    width = .79
    height = 1.72
    for digit, letter in enumerate(text):
        origin = 1.10 * digit - .96
        for path in NUMERALS[letter]:
            for (a, b), (c, d) in zip(path, path[1:]):
                u, v = (width * c - width * a) * sign, (d - b) * height * sign
                center = (x + sign * (origin + (a + c) * width / 2),
                          y + sign * (b + d - 1) * height / 2, .001)
                _box(parent, center, (math.hypot(u, v) / 2 + .045, .065, .002), PAINT,
                     math.atan2(v, u))


def _goal(parent, x, y, side):
    pole = x + side * 1.65
    height = 10 / 3 * YARD_M
    width = 23 + 1 / 3
    left, right = y - width * YARD_M / 6, y + width * YARD_M / 6
    _pole(parent, (pole, y, .13), (pole, y, height - .40), .092, GOLD)
    _pole(parent, (pole, y, height - .40), (x, y, height), .078, GOLD)
    _pole(parent, (x, left, height), (x, right, height), .056, GOLD)
    for offset in (left, right):
        _pole(parent, (x, offset, height), (x, offset, 8.9), .047, GOLD)
    _pole(parent, (pole, y, .06), (pole, y, 1.73), .18, NAVY)
    _pole(parent, (pole, y, 1.53), (pole, y, 1.66), .184, GOLD)


def _track_edge(assets, parent, name, inside, outside, z, color):
    """A closed stadium-shaped ribbon, including semicircular turns."""
    cx, cy = FIELD_CENTER
    straight_half = 48.
    vertices = []
    for side in (1, -1):
        for segment in range(49):
            theta = -math.pi / 2 + segment * math.pi / 48 + (math.pi if side < 0 else 0)
            for radius in (inside, outside):
                x = cx + side * straight_half + radius * math.cos(theta)
                y = cy + radius * math.sin(theta)
                vertices.extend(((x, y, z), (x, y, z - .025)))
    faces = []
    for i in range(98):
        a, b, c, d = range(i * 4, i * 4 + 4)
        e, f, g, h = range((i + 1) % 98 * 4, (i + 1) % 98 * 4 + 4)
        faces.extend(((a, c, g), (a, g, e), (b, h, d), (b, f, h),
                      (a, e, f), (a, f, b), (c, h, g), (c, d, h)))
    ET.SubElement(assets, "mesh", name=name,
                  vertex=_numbers(v for vertex in vertices for v in vertex),
                  face=" ".join(str(i) for face in faces for i in face))
    _geom(parent, "mesh", color, mesh=name)


def _track(assets, parent):
    width = 1.12
    radius = 28.55
    _track_edge(assets, parent, "stadium_track", radius, radius + 6 * width, .00, ".43 .22 .20 1")
    for lane in range(7):
        position = radius + lane * width
        _track_edge(assets, parent, f"stadium_lane_{lane}", position - .023, position + .023,
                    .003, ".84 .80 .69 1")


def _fence(parent, y, xmin, xmax):
    for i in range(round((xmax - xmin) / 2.8) + 1):
        x = min(xmax, xmin + 2.8 * i)
        _pole(parent, (x, y, 0), (x, y, 1.08), .032, STEEL)
    for z in (.15, 1.07):
        _pole(parent, (xmin, y, z), (xmax, y, z), .024, STEEL)
    _box(parent, ((xmin + xmax) / 2, y, .60), ((xmax - xmin) / 2, .012, .44), ".37 .47 .49 .19")


def _bleachers(parent, x, y, direction, span, rows):
    for block_x, length in ((x - span / 4 - .9, span / 2 - 1.8), (x + span / 4 + .9, span / 2 - 1.8)):
        for row in range(rows):
            row_y, z = y + direction * row * .74, .36 + row * .31
            _box(parent, (block_x, row_y, z), (length / 2, .21, .049), ALUMINUM)
            _box(parent, (block_x, row_y - direction * .29, z - .22), (length / 2, .30, .032), STEEL)
            _box(parent, (block_x, row_y + direction * .18, z + .081), (length / 2, .035, .047), STEEL)
        for station in (-.41, -.15, .15, .41):
            sx = block_x + length * station
            end_y = y + direction * ((rows - 1) * .74 + .28)
            end_z = .37 + (rows - 1) * .31
            _pole(parent, (sx, y - direction * .50, .12), (sx, end_y, end_z), .045, STEEL)
            _pole(parent, (sx, end_y, .05), (sx, end_y, end_z + 1), .035, STEEL)
        for height in (.46, .84):
            _pole(parent, (block_x - length / 2, end_y, end_z + height),
                  (block_x + length / 2, end_y, end_z + height), .028, ALUMINUM)
    for row in range(rows * 2):
        _box(parent, (x, y + direction * (.30 + row * .37), .07 + row * .155), (.58, .21, .045), STEEL)
    for side in (-1, 1):
        _pole(parent, (x + side * .66, y - direction * .25, .92),
              (x + side * .66, y + direction * (rows - 1) * .74, .92 + (rows - 1) * .31),
              .025, ALUMINUM)


def _press_box(parent, x, y):
    _box(parent, (x, y, 3.85), (6.1, 1.08, .73), NAVY)
    _box(parent, (x, y, 4.63), (6.27, 1.24, .12), STEEL)
    _box(parent, (x, y - 1.09, 3.19), (6.1, .031, .045), GOLD)
    for offset in (-4.5, -2.71, -.92, .92, 2.71, 4.5):
        _box(parent, (x + offset, y - 1.10, 4.13), (.81, .028, .26), ".17 .34 .44 1")
        _box(parent, (x + offset, y - 1.135, 4.18), (.80, .009, .018), ".36 .51 .61 1")
    for index, letter in enumerate("FALCONS"):
        origin = x + .76 * (index - 3) - .23
        for path in LETTERS[letter]:
            for (a, b), (c, d) in zip(path, path[1:]):
                _pole(parent, (origin + .46 * a, y - 1.10, 3.37 + .38 * b),
                      (origin + .46 * c, y - 1.10, 3.37 + .38 * d), .021, PAINT)
    for offset in (-5.61, 5.61):
        _pole(parent, (x + offset, y, 0), (x + offset, y, 3.35), .09, STEEL)


def _grounds(parent):
    cx, cy = FIELD_CENTER
    _bleachers(parent, cx - 3, cy + 37.2, 1, 40, 8)
    _press_box(parent, cx - 3, cy + 43.6)
    _bleachers(parent, cx + 2, cy - 37.7, -1, 27, 5)
    _fence(parent, cy + 36.01, cx - 42, cx + 42)
    _fence(parent, cy - 36.01, cx - 42, cx + 42)
    for direction in (-1, 1):
        for x in (cx - 32, cx + 32):
            y = cy + direction * 44.
            _pole(parent, (x, y, .0), (x, y, 15.), .12, STEEL)
            _pole(parent, (x - 1.45, y, 14.68), (x + 1.45, y, 14.68), .061, STEEL)
            for lamp in (-1.2, -.6, 0, .6, 1.2):
                _box(parent, (x + lamp, y, 14.92), (.23, .14, .20), STEEL)
                _box(parent, (x + lamp, y - direction * .146, 14.92), (.19, .010, .16), PAINT)
    # Sideline benches stay behind the white boundary and out of the route.
    for direction in (-1, 1):
        y = cy + direction * 26.1
        for offset in (-14., 5, 22):
            x = cx + offset
            _box(parent, (x, y, .45), (1.9, .20, .055), ALUMINUM)
            _box(parent, (x, y + direction * .18, .69), (1.9, .034, .12), ALUMINUM)
            for leg in (-1.5, 1.5):
                _pole(parent, (x + leg, y, 0), (x + leg, y, .62), .032, STEEL)
    # The tree line breaks up the skyline past the stadium perimeter.
    rng = Random(108)
    for direction in (-1, 1):
        for i in range(25):
            x = cx - 76 + 6.3 * i + rng.uniform(-2.7, 2.7)
            y = cy + direction * rng.uniform(48., 64.)
            height = rng.uniform(2.8, 4.7)
            tint = rng.uniform(.86, 1.16)
            green = _numbers((.178 * tint, .28 * tint, .185 * tint, 1))
            _pole(parent, (x, y, 0), (x, y, height), .18, ".27 .24 .19 1")
            for branch_x, branch_y, factor in ((0., 0., 1.), (-1.1, .6, .74), (1., -.4, .80)):
                _geom(parent, "ellipsoid", green, pos=_numbers((x + branch_x, y + branch_y, height * factor)),
                      size=_numbers((1.57 * factor, 1.46 * factor, height * .49 * factor)))


def _passing_machine(world, origin, heading):
    """Static visual launcher; the archived physics still owns the football."""
    machine = ET.SubElement(world, "body", name="studio_passing_machine",
                            pos=_numbers((origin[0], origin[1], 0)), quat=_yaw(heading))
    _box(machine, (-.44, 0, .15), (.47, .38, .12), ".075 .10 .14 1")
    for x in (-.76, -.12):
        for y in (-.38, .38):
            _geom(machine, "cylinder", ".035 .045 .06 1", pos=_numbers((x,y,.13)),
                  size=".13 .06", quat=".7071068 .7071068 0 0")
    _box(machine, (-.43, 0, .67), (.12, .14, .43), ".23 .30 .38 1")
    _box(machine, (-.43, 0, 1.10), (.32, .26, .13), ".11 .16 .23 1")
    _box(machine, (-.67, -.268, 1.10), (.10, .008, .055), ".12 .72 .91 1")
    # Twin wheels and an open center channel end at the measured release point.
    for y in (-.205, .205):
        _geom(machine, "cylinder", ".065 .085 .115 1", pos=_numbers((-.22,y,origin[2])),
              size=".235 .075", quat=".7071068 .7071068 0 0")
        _geom(machine, "cylinder", ".42 .53 .65 1", pos=_numbers((-.22,y*1.39,origin[2])),
              size=".115 .012", quat=".7071068 .7071068 0 0")
    _box(machine, (-.48, 0, origin[2]-.17), (.38, .12, .025), ".30 .39 .49 1")


def _stadium_scene(scene_xml, height, width, field_heading_rad, scrimmage_xy_m,
                   quarterback_origin, quarterback_heading_rad, studio=False):
    scene = ET.fromstring(scene_xml)
    world = scene.find("worldbody")
    assets = scene.find("asset")
    backdrop = world.find("geom[@name='backdrop']")
    if backdrop is not None:
        world.remove(backdrop)
    pelvis = world.find("body[@name='pelvis']")
    for marker in pelvis.findall("geom"):
        if marker.get("name", "").startswith(("localization_", "rear_marker_")):
            pelvis.remove(marker)
    floor = world.find("geom[@name='floor']")
    floor.set("rgba", "0 0 0 0")
    _geom(world, "plane", STUDIO_BACKGROUND + " 1" if studio else ".206 .29 .18 1",
          name="stadium_grounds", pos="0 0 -.06", size="180 180 .05")
    visual = scene.find("visual")
    visual.find("global").attrib.update(offwidth=str(width), offheight=str(height), fovy="43")
    visual.find("headlight").attrib.update(ambient=".20 .22 .25", diffuse=".20 .21 .23",
                                          specular=".015 .015 .015")
    ET.SubElement(visual, "quality", offsamples="4", shadowsize="4096")
    ET.SubElement(visual, "map", znear=".0025", zfar="3", fogstart=".90", fogend="2.9",
                  shadowclip=".26", shadowscale=".40")
    ET.SubElement(visual, "rgba", fog=".72 .82 .88 1")
    ET.SubElement(scene, "statistic", extent="115")
    for light in world.findall("light"):
        world.remove(light)
    ET.SubElement(world, "light", name="stadium_sun", type="directional", pos="-5 -7 11",
                  dir=".52 .45 -.85", ambient=".25 .25 .21", diffuse=".75 .72 .64",
                  specular=".24 .24 .21", castshadow="true")
    stadium = ET.SubElement(world, "body", name="stadium",
                            pos=_numbers((*scrimmage_xy_m, 0)), quat=_yaw(field_heading_rad))
    files = _stadium_assets(assets)
    if studio:
        for texture in list(assets.findall("texture")):
            if texture.get("type") == "skybox":
                assets.remove(texture)
        ET.SubElement(assets, "texture", name="studio_background", type="skybox", builtin="flat",
                      rgb1=STUDIO_BACKGROUND, rgb2=STUDIO_BACKGROUND, width="32", height="32")
        visual.find("rgba").set("fog", STUDIO_BACKGROUND + " 1")
        light = world.find("light[@name='stadium_sun']")
        light.attrib.update(name="studio_softbox", ambient=".40 .43 .49", diffuse=".65 .68 .73",
                            specular=".12 .14 .17", castshadow="false")
        _field(stadium, goalposts=False)
    else:
        _field(stadium)
        _track(assets, stadium)
        _grounds(stadium)
    if quarterback_origin is not None:
        from .stadium_quarterback import add_quarterback

        if studio:
            _passing_machine(world, quarterback_origin, quarterback_heading_rad)
        else:
            add_quarterback(world, quarterback_origin, quarterback_heading_rad)
    return ET.tostring(scene, encoding="unicode"), files


class StadiumRenderer:
    """Replay the plant's poses in a separate MuJoCo model with spectator set dressing."""

    def __init__(self, plant, *, height=448, width=800, field_heading_rad=0.,
                 scrimmage_xy_m=(0., 0.), quarterback_origin=None, release_time=None,
                 quarterback_heading_rad=None, studio=False, receiver_split_m=None,
                 midfield=False):
        if quarterback_origin is not None and (release_time is None or not math.isfinite(release_time)):
            raise ValueError("quarterback presentation requires a finite release time")
        if quarterback_heading_rad is None:
            quarterback_heading_rad = field_heading_rad
        field_origin = scrimmage_xy_m
        if receiver_split_m is not None:
            if not math.isfinite(receiver_split_m) or not 0 <= receiver_split_m <= FIELD_HALF_WIDTH:
                raise ValueError("receiver split must lie within the field half width")
            shift = -FIELD_CENTER[1] - receiver_split_m
            field_origin = (scrimmage_xy_m[0] - math.sin(field_heading_rad) * shift,
                            scrimmage_xy_m[1] + math.cos(field_heading_rad) * shift)
        if midfield:
            if quarterback_origin is None:
                raise ValueError('midfield presentation requires the measured passer origin')
            c, s = math.cos(field_heading_rad), math.sin(field_heading_rad)
            field_origin = (quarterback_origin[0] - c * FIELD_CENTER[0] + s * FIELD_CENTER[1],
                            quarterback_origin[1] - s * FIELD_CENTER[0] - c * FIELD_CENTER[1])
        xml, assets = _stadium_scene(plant.scene_xml, height, width, field_heading_rad, field_origin,
                                     quarterback_origin, quarterback_heading_rad, studio)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.quarterback = None
        if quarterback_origin is not None and not studio:
            from .stadium_quarterback import QuarterbackPose

            self.quarterback = QuarterbackPose(self.model, quarterback_origin, quarterback_heading_rad)
        self.release_time = release_time
        self.renderer = mujoco.Renderer(self.model, height=height, width=width, max_geom=4096)
        self.scene_option = mujoco.MjvOption()
        # Kinematic replay draws the hand meshes without the solver's tendon overlays.
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True

    def update_scene(self, data, *, camera):
        self.data.time = data.time
        self.data.qpos[:] = data.qpos
        if self.quarterback is not None:
            self.quarterback.update(self.data, self.release_time)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.scene_option)

    def render(self):
        return self.renderer.render()

    def close(self):
        self.renderer.close()
