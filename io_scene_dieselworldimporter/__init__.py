bl_info = {
    "name": "PAYDAY 2 Level Importer (JSON)",
    "author": "Columbus",
    "version": (5, 12, 2),
    "blender": (4, 0, 0),
    "location": "File > Import > PAYDAY 2 Level (.json)",
    "description": "Imports PAYDAY 2 level .json files, converting .model files with PD2ModelParser (parallel), instancing repeats, and rebuilding textured materials from material_configs",
    "warning": "Requires io_scene_dieselmodeltoolwrapper (or _master) / PD2ModelParser.exe. Needs Blender 4.0+ (node group interface API)",
    "category": "Import-Export",
}

# ---------------------------------------------------------------------------
# 5.9.0 — bug-fix pass. Behaviour changes worth knowing about:
#
#   * "Only Import g_ Meshes" no longer drops lights inside instances, no
#     longer turns off shadow-projection shadows, and no longer mangles the
#     placement of lights whose parent mesh it deletes.
#   * Fresnel can no longer manufacture opacity out of nothing (5.12.2).
#     It is gated by the surface's own alpha, so fully transparent stays
#     fully transparent — 5.12.1 turned the cut-out areas of decals and
#     broken windows into solid black.
#   * Glass gets fresnel (5.12.1): it turns opaque at grazing angles
#     instead of reading as an evenly transparent grey film. The curve is
#     the Schlick form the material_config's fresnel_settings vector3
#     describes; see FRESNEL_SETTINGS_ORDER for the component mapping.
#   * A plain "generic" template carrying an opacity texture now wires that
#     texture straight to the BSDF Alpha, bypassing the alpha-mode step,
#     the GSMA multiply and fresnel entirely.
#   * Light cone geometry is shaded as FAKE volumetrics by default
#     (5.11.2): an additive surface whose brightness follows apparent
#     view depth, so it reads as fog while keeping the texture's real UVs
#     and costing almost nothing to render. True volume shading, hiding,
#     and leaving it alone are all still available.
#   * Spot lights get a soft cone edge and an emitter size that scales with
#     range, instead of a 0.05 point source with Blender's hard 0.15 blend.
#   * material_config resolution order (5.10.3): <unit>.material_config
#     first, then the .unit's own <material_config file="..."/>, then the
#     .object's <diesel materials="..."/>, then any materials="..." found
#     anywhere in the .object. A dangling or absent reference now falls
#     back to a same-named config beside the unit/object/model instead of
#     leaving the model untextured.
#   * Opacity textures feed the BSDF alpha directly (5.10.1) instead of
#     being split and read from the red channel, and now force a blended
#     alpha mode when the render template didn't already ask for one.
#   * Shadow-projection lights get their shadows back (5.10.0): the unit
#     path is now part of the decision, and JSON lights actually pair up
#     with the model's light nodes instead of always being rebuilt at the
#     unit root with default settings.
#   * Light intensity presets replaced with the game's light_intensity_db
#     values. They are far smaller than the old guesses, so raise "Light
#     Power Scale" if the level comes in dark.
#   * Normal maps always take X from the alpha channel (5.9.2). Auto-
#     detecting an RGB layout was unreliable — compression noise in the red
#     or blue channel tipped maps into the wrong branch — so only the Y
#     source (green, or red for red-only maps) is detected now.
#   * brute_force_zero_transform() now resets scale/delta_scale too.
#   * Minimum Blender raised to 4.0 — the shader node group has always used
#     the 4.0 interface API, the old bl_info just claimed 2.93.
#   * Material.blend_method / use_screen_refraction are now feature-checked,
#     so materials still build on Blender 4.3+.
# ---------------------------------------------------------------------------

import bpy
import json
import os
import re
import math
import time
import shutil
import tempfile
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from bpy.props import (StringProperty, BoolProperty, EnumProperty,
                       FloatProperty, IntProperty)
from bpy.types import AddonPreferences, Operator
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector, Matrix

# ----------------------------------------------------------------------------
# Constants / logging
# ----------------------------------------------------------------------------

LOG_PREFIX = "[PD2 Importer]"

# Set from the operator's "Verbose Console Log" option; gates chatty
# per-unit/per-light logging in helper functions (console printing itself
# is a measurable slowdown on big levels).
VERBOSE = False


def vlog(msg):
    if VERBOSE:
        print(f"{LOG_PREFIX} {msg}")


# Fake-volumetric cone shaping. A Layer Weight blend above 0.5 widens the
# band the fade happens over, and a bias above 1.0 lifts the whole curve so
# the centre of the shaft doesn't wash out before the rim has faded.
CONE_LAYER_BLEND = 0.8
CONE_FADE_BIAS = 1.1

# Spot cone edge softness. Blender defaults to 0.15, which reads as a hard
# rim; Diesel spots fade out long before the cone boundary.
# Glass fresnel master strength, overridden per-import from the options.
GLASS_FRESNEL = 1.0

# Component order of the material_config's fresnel_settings vector3, e.g.
#   <variable name="fresnel_settings" type="vector3" value="2 1 0.6"/>
# read as power=2, scale=1, bias=0.6 and evaluated as the Schlick form
#   fresnel = bias + scale * facing ** power
# An exponent of 2 with a unit scale are the textbook values, and a
# fractional third component reads as a bias/minimum, which is why the
# order is this way round. If your assets disagree, this tuple is the only
# thing that needs changing.
FRESNEL_SETTINGS_ORDER = ("power", "scale", "bias")

SPOT_BLEND_DEFAULT = 0.5
SPOT_BLEND_MIN = 0.25

TINY_THRESHOLD = 1e-4
HUGE_THRESHOLD = 1e7

# Module-name fragments that identify the Diesel Model Tool Wrapper addon
WRAPPER_NAME_FRAGMENT = "dieselmodeltoolwrapper"

PARSER_TIMEOUT_SECONDS = 180

# Blender's duplicate suffix is normally three digits, but it keeps counting
# past .999 on large levels (g_g.1000, light_omni.1247 ...), so accept 3+.
_dup_suffix_re = re.compile(r"^(.*)\.(\d{3,})$")

# Internal object markers. Both are stripped from the finished level by
# clear_internal_markers() so they never end up saved in the .blend.
KEEP_TRANSFORM_PROP = "_pd2_keep_transform"
SHADOW_FLAG_PROP = "_pd2_shadow_projection"


def log(msg):
    print(f"{LOG_PREFIX} {msg}")


def log_error(msg):
    print(f"{LOG_PREFIX} [ERROR] {msg}")


# ----------------------------------------------------------------------------
# Value parsing / sanitizing
# ----------------------------------------------------------------------------

def sanitize_value(v):
    """Force garbage floats (tiny sci-notation noise or absurdly huge values) to 0."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if abs(v) < TINY_THRESHOLD:
        return 0.0
    if abs(v) > HUGE_THRESHOLD:
        return 0.0
    return v


def parse_triplet(s, kind):
    """Parse 'Vector3(x, y, z)' or 'Rotation(a, b, c)' into a sanitized 3-tuple."""
    if not s:
        return (0.0, 0.0, 0.0)
    try:
        inner = s[s.index("(") + 1:s.rindex(")")]
        parts = [p.strip() for p in inner.split(",")]
        vals = [sanitize_value(p) for p in parts[:3]]
        while len(vals) < 3:
            vals.append(0.0)
        return tuple(vals)
    except Exception:
        log_error(f"Could not parse {kind} string: {s!r} -> defaulting to (0,0,0)")
        return (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------
# Diesel binary scriptdata parser (pure Python) — for .continent / .mission /
# .world files. Format per the community-documented spec (kythyria's
# payday2-tools): sectioned pools of floats/strings/vectors/quaternions/
# idstrings/tables, with 32-bit tagged value refs.
# ----------------------------------------------------------------------------

import struct

_SD_X64_MAGIC = 568494624

# ---- Diesel Idstring hash (Bob Jenkins lookup8, 64-bit) --------------------
# Verified against known PAYDAY 2 idstring test vectors.

_M64 = (1 << 64) - 1


def _mix64(a, b, c):
    a = (a - b - c) & _M64; a ^= c >> 43
    b = (b - c - a) & _M64; b ^= (a << 9) & _M64
    c = (c - a - b) & _M64; c ^= b >> 8
    a = (a - b - c) & _M64; a ^= c >> 38
    b = (b - c - a) & _M64; b ^= (a << 23) & _M64
    c = (c - a - b) & _M64; c ^= b >> 5
    a = (a - b - c) & _M64; a ^= c >> 35
    b = (b - c - a) & _M64; b ^= (a << 49) & _M64
    c = (c - a - b) & _M64; c ^= b >> 11
    a = (a - b - c) & _M64; a ^= c >> 12
    b = (b - c - a) & _M64; b ^= (a << 18) & _M64
    c = (c - a - b) & _M64; c ^= b >> 22
    return a, b, c


def diesel_hash(data, level=0):
    """64-bit Diesel Idstring hash of a path string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    a = b = level & _M64
    c = 0x9e3779b97f4a7c13
    length = len(data)
    i = 0
    ln = length
    while ln >= 24:
        a = (a + int.from_bytes(data[i:i + 8], "little")) & _M64
        b = (b + int.from_bytes(data[i + 8:i + 16], "little")) & _M64
        c = (c + int.from_bytes(data[i + 16:i + 24], "little")) & _M64
        a, b, c = _mix64(a, b, c)
        i += 24
        ln -= 24
    c = (c + length) & _M64
    k = data[i:]
    if ln >= 17:
        c = (c + (int.from_bytes(k[16:ln], "little") << 8)) & _M64
    if ln >= 9:
        b = (b + int.from_bytes(k[8:min(ln, 16)], "little")) & _M64
    if ln >= 1:
        a = (a + int.from_bytes(k[0:min(ln, 8)], "little")) & _M64
    a, b, c = _mix64(a, b, c)
    return c


# ---- Hashlist (borrowed from the model tool's directory) -------------------

HASHLIST_FILENAMES = ("hashes.txt", "hashlist", "hashlist.txt", "hashes")
_hashlist_cache = None  # dict: u64 hash (both byte orders) -> path string


def find_hashlist_file(parser_exe):
    """Look for the hashlist that ships with PD2ModelParser / the model tool,
    in the exe's directory and common subfolders."""
    if not parser_exe:
        return None
    exe_dir = os.path.dirname(parser_exe)
    search_dirs = [exe_dir,
                   os.path.join(exe_dir, "Data"),
                   os.path.join(exe_dir, "lib"),
                   os.path.dirname(exe_dir)]
    for d in search_dirs:
        for name in HASHLIST_FILENAMES:
            cand = os.path.join(d, name)
            if os.path.isfile(cand) and os.path.getsize(cand) > 1024:
                return cand
    return None


def load_hashlist(parser_exe):
    """Build (or load a cached) hash -> path lookup from the model tool's
    hashlist. Hashing ~500k paths in Python takes a little while the first
    time, so the built table is pickled next to the temp dir and reused
    while the hashlist file is unchanged."""
    global _hashlist_cache
    if _hashlist_cache is not None:
        return _hashlist_cache

    hl_path = find_hashlist_file(parser_exe)
    if hl_path is None:
        log("No hashlist found near the model tool; idstrings will stay "
            "as @ID...@ placeholders")
        _hashlist_cache = {}
        return _hashlist_cache

    import pickle
    st = os.stat(hl_path)
    cache_file = os.path.join(
        tempfile.gettempdir(),
        f"pd2_hashlist_{diesel_hash(hl_path):016x}_{st.st_mtime_ns}_{st.st_size}.pickle")

    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "rb") as f:
                _hashlist_cache = pickle.load(f)
            log(f"Loaded cached hashlist table "
                f"({len(_hashlist_cache)} entries) from {hl_path}")
            return _hashlist_cache
        except Exception:
            pass

    log(f"Building hash table from {hl_path} (one-time, may take a moment)...")
    t0 = time.time()
    table = {}
    dh = diesel_hash
    with open(hl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                table[dh(s)] = s
    _hashlist_cache = table
    log(f"Hashed {len(table)} paths in {time.time() - t0:.1f}s")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return table


def resolve_idstring(h):
    """Resolve a 64-bit idstring to a path, trying both byte orders
    (some files store the byte-swapped form)."""
    if _hashlist_cache:
        s = _hashlist_cache.get(h)
        if s is not None:
            return s
        s = _hashlist_cache.get(
            int.from_bytes(h.to_bytes(8, "little"), "big"))
        if s is not None:
            return s
    return f"@ID{h:016x}@"


class _ScriptdataReader:
    def __init__(self, buf):
        self.buf = buf
        self.is_x64 = (len(buf) >= 4 and
                       struct.unpack_from("<I", buf, 0)[0] == _SD_X64_MAGIC)
        self.osize = 8 if self.is_x64 else 4
        pad = 16 if self.is_x64 else 12
        step = pad + self.osize
        self.float_off = self._off(pad + step * 0)
        self.string_off = self._off(pad + step * 1)
        self.vector_off = self._off(pad + step * 2)
        self.quat_off = self._off(pad + step * 3)
        self.idstring_off = self._off(pad + step * 4)
        self.table_off = self._off(pad + step * 5)
        self.root_off = 152 if self.is_x64 else 100
        self.seen_tables = {}

    def _off(self, at):
        fmt = "<Q" if self.is_x64 else "<I"
        return struct.unpack_from(fmt, self.buf, at)[0]

    def _string(self, index):
        rec = self.string_off + self.osize + index * (16 if self.is_x64 else 8)
        s_off = self._off(rec)
        end = self.buf.find(b"\x00", s_off)
        if end < 0:
            end = len(self.buf)
        return self.buf[s_off:end].decode("utf-8", errors="replace")

    def value(self, offset):
        item = struct.unpack_from("<I", self.buf, offset)[0]
        tag = (item >> 24) & 0xFF
        v = item & 0xFFFFFF
        if tag == 0:
            return None
        if tag == 1:
            return False
        if tag == 2:
            return True
        if tag == 3:
            return struct.unpack_from("<f", self.buf, self.float_off + v * 4)[0]
        if tag == 4:
            return self._string(v)
        if tag == 5:
            x, y, z = struct.unpack_from("<3f", self.buf, self.vector_off + v * 12)
            return ("__vector__", x, y, z)
        if tag == 6:
            x, y, z, w = struct.unpack_from("<4f", self.buf, self.quat_off + v * 16)
            return ("__quaternion__", x, y, z, w)
        if tag == 7:
            h = struct.unpack_from("<Q", self.buf, self.idstring_off + v * 8)[0]
            return resolve_idstring(h)
        if tag == 8:
            if v in self.seen_tables:
                return self.seen_tables[v]
            rec = self.table_off + v * (32 if self.is_x64 else 20)
            if self.is_x64:
                meta_i = struct.unpack_from("<q", self.buf, rec)[0]
                count = struct.unpack_from("<I", self.buf, rec + 8)[0]
                items_off = struct.unpack_from("<Q", self.buf, rec + 16)[0]
            else:
                meta_i = struct.unpack_from("<i", self.buf, rec)[0]
                count = struct.unpack_from("<I", self.buf, rec + 4)[0]
                items_off = struct.unpack_from("<I", self.buf, rec + 12)[0]
            table = {}
            self.seen_tables[v] = table
            if meta_i >= 0:
                table["_meta"] = self._string(meta_i)
            for i in range(count):
                io_ = items_off + i * 8
                k = self.value(io_)
                val = self.value(io_ + 4)
                table[k if isinstance(k, str) else str(k)] = val
            return table
        raise ValueError(f"Unrecognised scriptdata tag {tag} at 0x{offset:x}")


def _sd_jsonify(v):
    """Convert parsed scriptdata into JSON-friendly structures, formatting
    vectors/rotations in the same string style as the level JSON."""
    if isinstance(v, tuple):
        if v[0] == "__vector__":
            return f"Vector3({v[1]:g}, {v[2]:g}, {v[3]:g})"
        if v[0] == "__quaternion__":
            # Quaternion -> Diesel-style yaw/pitch/roll degrees string
            x, y, z, w = v[1], v[2], v[3], v[4]
            try:
                from mathutils import Quaternion as MQuat
                e = MQuat((w, x, y, z)).to_euler('ZXY')
                yaw, pitch, roll = (math.degrees(e.z), math.degrees(e.x),
                                    math.degrees(e.y))
                return (f"Rotation({yaw:g}, {pitch:g}, {roll:g}) "
                        f"[quat {x:g}, {y:g}, {z:g}, {w:g}]")
            except Exception:
                return f"Quaternion({x:g}, {y:g}, {z:g}, {w:g})"
    if isinstance(v, float):
        # trim float noise
        return round(v, 6)
    if isinstance(v, dict):
        return {k: _sd_jsonify(val) for k, val in v.items()}
    return v


def parse_diesel_scriptdata(filepath):
    """Parse a Diesel binary scriptdata file (.continent, .mission, .world,
    .world_sounds ...). If the file is already textual XML/JSON it is
    returned as-is under a '_raw_text' key."""
    with open(filepath, "rb") as f:
        buf = f.read()
    head = buf.lstrip()[:1]
    if head in (b"<", b"{"):
        return {"_raw_text": buf.decode("utf-8", errors="replace")}
    reader = _ScriptdataReader(buf)
    return _sd_jsonify(reader.value(reader.root_off))


def convert_instances_to_json(instances, assets_dir, out_dir):
    """For every instance entry, find its continent scriptdata file in the
    extracted assets and write a converted .json next to the level JSON.
    Tries '<folder>.continent' plus common sibling files."""
    os.makedirs(out_dir, exist_ok=True)
    n_ok = n_missing = 0
    for inst in instances:
        folder = inst.get("folder", "").strip("/")
        if not folder:
            continue
        rel = folder.replace("/", os.sep)
        candidates = [
            os.path.join(assets_dir, rel + ".continent"),
            os.path.join(assets_dir, rel, "world.continent"),
            os.path.join(assets_dir, os.path.dirname(rel),
                         os.path.basename(rel) + ".continent"),
        ]
        src = next((c for c in candidates if os.path.isfile(c)), None)
        if src is None:
            log_error(f"  instance continent not found: {folder}")
            n_missing += 1
            continue
        out_name = folder.replace("/", "_") + ".continent.json"
        out_path = os.path.join(out_dir, out_name)
        try:
            data = parse_diesel_scriptdata(src)
            payload = {
                "_instance": inst,
                "_source_file": src,
                "continent": data,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            log(f"  instance converted: {os.path.basename(src)} -> {out_name}")
            n_ok += 1
        except Exception as e:
            log_error(f"  failed to convert {src}: {e}")
            traceback.print_exc()
            n_missing += 1
    return n_ok, n_missing


def collect_instance_units(instances, assets_dir):
    """Parse each instance's continent scriptdata (raw, not jsonified) and
    extract its statics as importable unit dicts. Returns a list of
    {instance, units:[{path, name_id, position, quat, lights}]} groups."""
    groups = []
    for inst in instances:
        folder = inst.get("folder", "").strip("/")
        if not folder:
            continue
        rel = folder.replace("/", os.sep)
        # Kept in step with convert_instances_to_json: the third candidate
        # used to be missing here, so some instances converted fine but were
        # silently skipped for placement.
        candidates = [
            os.path.join(assets_dir, rel + ".continent"),
            os.path.join(assets_dir, rel, "world.continent"),
            os.path.join(assets_dir, os.path.dirname(rel),
                         os.path.basename(rel) + ".continent"),
        ]
        src = next((c for c in candidates if os.path.isfile(c)), None)
        if src is None:
            log_error(f"  instance continent not found: {folder}")
            continue
        try:
            with open(src, "rb") as f:
                buf = f.read()
            if buf.lstrip()[:1] in (b"<", b"{"):
                log_error(f"  instance file is textual, skipping placement: {src}")
                continue
            reader = _ScriptdataReader(buf)
            root = reader.value(reader.root_off)
        except Exception as e:
            log_error(f"  failed to parse {src}: {e}")
            traceback.print_exc()
            continue

        statics = root.get("statics") if isinstance(root, dict) else None
        if not isinstance(statics, dict):
            # some files nest one level deeper or ARE the statics table
            statics = root if isinstance(root, dict) else {}

        inst_units = []
        for sid, static in statics.items():
            if not isinstance(static, dict):
                continue
            ud = static.get("unit_data")
            if not isinstance(ud, dict):
                continue
            path = ud.get("name", "")
            if not isinstance(path, str) or not path or path.startswith("@ID"):
                if isinstance(path, str) and path.startswith("@ID"):
                    log_error(f"  unresolved unit idstring in {folder}: {path}")
                continue
            pos = ud.get("position")
            if isinstance(pos, tuple) and pos[0] == "__vector__":
                position = (sanitize_value(pos[1]), sanitize_value(pos[2]),
                            sanitize_value(pos[3]))
            else:
                position = (0.0, 0.0, 0.0)
            rot = ud.get("rotation")
            if isinstance(rot, tuple) and rot[0] == "__quaternion__":
                quat = (rot[1], rot[2], rot[3], rot[4])  # x, y, z, w
            else:
                quat = (0.0, 0.0, 0.0, 1.0)
            lights = ud.get("lights")
            lights = _sd_jsonify(lights) if isinstance(lights, dict) else {}
            name_id = ud.get("name_id")
            if not isinstance(name_id, str) or not name_id:
                name_id = os.path.basename(path)
            inst_units.append({
                "path": path,
                "name_id": name_id,
                "unit_id": ud.get("unit_id", 0),
                "position": position,
                "quat": quat,
                "lights": lights,
            })
        if inst_units:
            groups.append({"instance": inst, "units": inst_units})
            log(f"  instance '{inst['name']}': {len(inst_units)} units from "
                f"{os.path.basename(src)}")
        else:
            log_error(f"  no statics found in {src}")
    return groups


def apply_instance_unit_transform(root, position, quat):
    """Local transform for a unit inside an instance: continent position/100
    and the continent quaternion, composed with the -90 X model-upright fix
    (applied first, exactly like the euler path for regular units)."""
    from mathutils import Quaternion as MQuat
    x, y, z, w = quat
    q = MQuat((w, x, y, z))
    upright = MQuat((1.0, 0.0, 0.0), math.radians(-90.0))
    root.rotation_mode = 'QUATERNION'
    root.rotation_quaternion = q @ upright
    root.location = Vector(tuple(p / 100.0 for p in position))


# ----------------------------------------------------------------------------
# .massunit parsing (mass-placed scatter: foliage, rocks, clutter)
# ----------------------------------------------------------------------------
#
# Binary layout (little-endian), reverse-engineered and verified against
# real files (contiguous offsets, exact end-of-data, all quaternions unit
# length):
#
#   header  16 bytes : u32 entry_count, u32 (unknown), u32 header_size,
#                      u32 (0)
#   entry   32 bytes : u64 unit_idstring_hash, u32 count_a, u32 count_b,
#                      u32 count_c, u32 data_offset, u32 0, u32 0
#   instance 28 bytes: float3 position (cm), float4 quaternion (x, y, z, w)
#
# count_b/count_c are the allocated instance count (count_a is occasionally
# slightly smaller — an enabled subset); count_b is used so nothing is lost.

MASSUNIT_INSTANCE_SIZE = 28
MASSUNIT_ENTRY_SIZE = 32


def find_massunit_file(json_path):
    """A level JSON at <level>/world/world.json has its massunit file at
    <level>/massunit.massunit (one directory up). Returns None when there
    isn't one — massunits are entirely optional."""
    world_dir = os.path.dirname(os.path.abspath(json_path))
    parent = os.path.dirname(world_dir)
    candidates = [os.path.join(parent, "massunit.massunit"),
                  os.path.join(world_dir, "massunit.massunit")]
    for d in (parent, world_dir):
        try:
            for fn in os.listdir(d):
                if fn.lower().endswith(".massunit"):
                    candidates.append(os.path.join(d, fn))
        except OSError:
            pass
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) >= 16:
            return c
    return None


def parse_massunit_file(path):
    """Return [{path, position, quat}] for every mass-placed instance.
    Unresolvable idstrings (no hashlist entry) are skipped with a log."""
    try:
        with open(path, "rb") as f:
            buf = f.read()
    except OSError as e:
        log_error(f"  cannot read massunit file: {e}")
        return []
    if len(buf) < 16:
        return []

    n_entries, _unk, hdr_size, _pad = struct.unpack_from("<4I", buf, 0)
    if hdr_size < 16 or hdr_size > len(buf):
        hdr_size = 16
    table_end = hdr_size + n_entries * MASSUNIT_ENTRY_SIZE
    if n_entries <= 0 or table_end > len(buf):
        log_error(f"  massunit header looks wrong (entries={n_entries})")
        return []

    out = []
    n_unresolved = 0
    for i in range(n_entries):
        rec = hdr_size + i * MASSUNIT_ENTRY_SIZE
        h, _ca, count, _cc, data_off, _e, _f = struct.unpack_from(
            "<Q6I", buf, rec)
        if count <= 0:
            continue
        end = data_off + count * MASSUNIT_INSTANCE_SIZE
        if data_off < table_end or end > len(buf):
            log_error(f"  massunit entry {i} out of range, skipped")
            continue
        unit_path = resolve_idstring(h)
        if not unit_path or unit_path.startswith("@ID"):
            n_unresolved += 1
            continue
        for j in range(count):
            o = data_off + j * MASSUNIT_INSTANCE_SIZE
            px, py, pz, qx, qy, qz, qw = struct.unpack_from("<7f", buf, o)
            out.append({
                "path": unit_path,
                "position": (sanitize_value(px), sanitize_value(py),
                             sanitize_value(pz)),
                "quat": (qx, qy, qz, qw),
            })
    if n_unresolved:
        log_error(f"  {n_unresolved} massunit idstring(s) could not be "
                  f"resolved (hashlist incomplete) — those units skipped")
    log(f"Massunits: {len(out)} instances across "
        f"{n_entries - n_unresolved} unit type(s) from "
        f"{os.path.basename(path)}")
    return out


# ----------------------------------------------------------------------------
# PD2ModelParser.exe discovery
# ----------------------------------------------------------------------------

def _scan_prefs_for_exe(prefs_obj):
    """Look through an addon's preference string properties for an existing
    path to PD2ModelParser.exe (or any existing .exe path)."""
    if prefs_obj is None:
        return None
    best = None
    try:
        for prop in prefs_obj.bl_rna.properties:
            if prop.type != 'STRING':
                continue
            try:
                val = getattr(prefs_obj, prop.identifier, "")
            except Exception:
                continue
            if not isinstance(val, str) or not val:
                continue
            val_abs = bpy.path.abspath(val)
            if not os.path.isfile(val_abs):
                # Maybe it's a directory containing the exe
                if os.path.isdir(val_abs):
                    cand = os.path.join(val_abs, "PD2ModelParser.exe")
                    if os.path.isfile(cand):
                        return cand
                continue
            base = os.path.basename(val_abs).lower()
            if "pd2modelparser" in base:
                return val_abs
            if base.endswith(".exe") and best is None:
                best = val_abs
    except Exception:
        pass
    return best


def find_parser_exe(own_prefs):
    """Locate PD2ModelParser.exe. Priority:
    1. Our own addon preference (manual override)
    2. The Diesel Model Tool Wrapper addon's preferences (auto-detect)
    Returns (exe_path or None, wrapper_found: bool)."""
    # 1. Manual override in our own preferences
    manual = bpy.path.abspath(own_prefs.parser_exe) if own_prefs.parser_exe else ""
    if manual and os.path.isfile(manual):
        log(f"Using manually set PD2ModelParser: {manual}")
        return manual, True

    # 2. Scan the wrapper addon's preferences
    wrapper_found = False
    for addon_name, addon in bpy.context.preferences.addons.items():
        if WRAPPER_NAME_FRAGMENT in addon_name.lower():
            wrapper_found = True
            log(f"Found wrapper addon: {addon_name}")
            exe = _scan_prefs_for_exe(getattr(addon, "preferences", None))
            if exe:
                log(f"Auto-detected PD2ModelParser from wrapper prefs: {exe}")
                return exe, True

    if manual:
        log_error(f"Manually set parser path does not exist: {manual}")
    return None, wrapper_found


# ----------------------------------------------------------------------------
# Import-speed helpers
# ----------------------------------------------------------------------------

def find_layer_collection(layer_col, target_col):
    """Recursively find the LayerCollection wrapping target_col."""
    if layer_col.collection == target_col:
        return layer_col
    for child in layer_col.children:
        found = find_layer_collection(child, target_col)
        if found is not None:
            return found
    return None


def set_collection_excluded(context, col, excluded):
    """Exclude/include a collection from the active view layer. While a
    collection is excluded, its objects are skipped by every depsgraph
    re-evaluation that bpy.ops calls trigger — which is what makes bulk
    imports slow down as the scene fills up."""
    lc = find_layer_collection(context.view_layer.layer_collection, col)
    if lc is not None:
        lc.exclude = excluded
        return True
    return False


# ----------------------------------------------------------------------------
# Transform helpers
# ----------------------------------------------------------------------------

def brute_force_zero_transform(obj):
    """TOP PRIORITY: obliterate any transform data on the object.
    Location, rotation (all modes), scale, delta transforms and the parent
    inverse are all forced to zero/identity. Scale and delta_scale used to be
    left alone, which let a junk root scale from the glTF survive the reset
    the rest of this function performs."""
    obj.matrix_parent_inverse.identity()
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    obj.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    obj.delta_location = (0.0, 0.0, 0.0)
    obj.delta_rotation_euler = (0.0, 0.0, 0.0)
    obj.delta_rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    if tuple(round(s, 6) for s in obj.scale) != (1.0, 1.0, 1.0):
        vlog(f"  zeroing non-identity scale {tuple(obj.scale)} on {obj.name}")
    obj.scale = (1.0, 1.0, 1.0)
    obj.delta_scale = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Depsgraph-independent world matrices
#
# Prototypes live in a collection that is never linked into the scene, and
# with Fast Import the level collection is excluded from the view layer. In
# both cases the depsgraph never evaluates those objects, so Object.
# matrix_world holds whatever was cached last (often identity) rather than
# the object's real placement. Re-parenting code therefore computes world
# matrices from local data instead of reading matrix_world, and writes the
# result to matrix_basis instead of assigning matrix_world.
# ---------------------------------------------------------------------------

_MAX_PARENT_DEPTH = 64


def local_world_matrix(obj, cache=None, _depth=0):
    """Object's world matrix derived purely from matrix_basis / parent
    inverse up the parent chain. Falls back to matrix_world for bone or
    vertex parenting, which cannot be reconstructed this way."""
    if obj is None:
        return Matrix.Identity(4)
    if cache is not None:
        hit = cache.get(obj.name)
        if hit is not None:
            return hit
    if _depth >= _MAX_PARENT_DEPTH:
        return obj.matrix_world.copy()
    if getattr(obj, "parent_type", 'OBJECT') != 'OBJECT':
        m = obj.matrix_world.copy()
    else:
        m = obj.matrix_basis.copy()
        if obj.parent is not None:
            m = (local_world_matrix(obj.parent, cache, _depth + 1)
                 @ obj.matrix_parent_inverse @ m)
    if cache is not None:
        cache[obj.name] = m
    return m


def reparent_keep_world(obj, new_parent, world):
    """Re-parent obj under new_parent so that its world matrix stays `world`,
    without relying on a depsgraph evaluation having happened."""
    obj.parent = new_parent
    obj.matrix_parent_inverse.identity()
    if new_parent is None:
        obj.matrix_basis = world
    else:
        obj.matrix_basis = local_world_matrix(new_parent).inverted_safe() @ world


def clear_internal_markers(collection):
    """Remove the importer's private custom properties from the finished
    level so they are not saved into the .blend."""
    n_cleared = 0
    for o in collection.all_objects:
        for prop in (KEEP_TRANSFORM_PROP, SHADOW_FLAG_PROP):
            if prop in o:
                del o[prop]
                n_cleared += 1
    if n_cleared:
        vlog(f"  cleared {n_cleared} internal marker propertie(s)")


def apply_unit_transform(root, position, rotation, rotation_order='XYZ',
                         flip_rot=(False, False, False),
                         flip_pos=(False, False, False), upright=True):
    """Apply -90 X base rotation + the unit's JSON rotation, position / 100.
    JSON mapping: Rotation(A, B, C) -> Z(yaw)=A, X=B, Y=C.
    Default order XYZ applies the -90 X upright FIRST and the yaw about the
    world's vertical axis LAST. Pass upright=False for container empties
    (like instance roots) whose children already carry their own -90 X."""
    rz, rx, ry = rotation
    if flip_rot[0]:
        rx = -rx
    if flip_rot[1]:
        ry = -ry
    if flip_rot[2]:
        rz = -rz

    base_x = -90.0 if upright else 0.0
    root.rotation_mode = rotation_order
    root.rotation_euler = (
        math.radians(base_x + rx),
        math.radians(ry),
        math.radians(rz),
    )

    px, py, pz = (p / 100.0 for p in position)
    if flip_pos[0]:
        px = -px
    if flip_pos[1]:
        py = -py
    if flip_pos[2]:
        pz = -pz
    root.location = Vector((px, py, pz))


# ----------------------------------------------------------------------------
# Synchronous model pipeline: PD2ModelParser.exe -> .glb -> glTF import
# ----------------------------------------------------------------------------

_unit_object_re = re.compile(
    r'<object\b[^>]*\bfile\s*=\s*"([^"]+)"', re.IGNORECASE)

_text_cache = {}       # abs path -> file text (or None if unreadable)
_unit_object_cache = {}  # (unit_path, assets_dir) -> object ref or None
_matconfig_for_unit = {}  # (unit_path, assets_dir) -> mc path or None


def _read_text_tolerant(full):
    """Read a small text asset, memoised. The .unit/.object files are read
    several times per model (model lookup, then the material chain), and
    they're tiny, so caching them removes almost all of that I/O."""
    if full in _text_cache:
        return _text_cache[full]
    try:
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        text = None
    _text_cache[full] = text
    return text




def _object_path_from_unit(unit_path, assets_dir):
    """If <unit_path>.unit exists, return the path from its
    <object file="..."/> attribute (Diesel-style forward-slash path,
    no extension). Returns None if the .unit is absent/unreadable or
    has no object reference."""
    ck = (unit_path, assets_dir)
    if ck in _unit_object_cache:
        return _unit_object_cache[ck]
    ref = None
    rel = unit_path.replace("/", os.sep).replace("\\", os.sep) + ".unit"
    full = os.path.normpath(os.path.join(assets_dir, rel))
    if os.path.isfile(full):
        text = _read_text_tolerant(full)
        m = _unit_object_re.search(text) if text else None
        if m:
            ref = m.group(1).strip().replace("\\", "/")
            # Strip an extension if the unit file happens to include one
            if ref.lower().endswith((".model", ".object", ".unit")):
                ref = ref.rsplit(".", 1)[0]
            ref = ref or None
    _unit_object_cache[ck] = ref
    return ref


def find_model_file(unit_path, assets_dir):
    """Locate <unit_path>.model under assets_dir. If it's missing, fall
    back to reading <unit_path>.unit and following its <object file="..."/>
    reference (which may itself chain through further .unit files) until
    a .model is found. Loop-protected."""
    if not unit_path:
        return None
    seen = set()
    path = unit_path
    while path and path not in seen:
        seen.add(path)
        rel = path.replace("/", os.sep).replace("\\", os.sep) + ".model"
        full = os.path.normpath(os.path.join(assets_dir, rel))
        if os.path.exists(full):
            if path != unit_path:
                vlog(f"  .model for {unit_path} resolved via .unit chain "
                     f"-> {path}")
            return full
        # No .model here — try the .unit file's object reference
        next_path = _object_path_from_unit(path, assets_dir)
        if next_path is None or next_path == path:
            return None
        path = next_path
    return None


def deselect_all():
    for o in bpy.context.selected_objects:
        o.select_set(False)


# Resolved once: `nice` is how POSIX priority is dropped here, because the
# obvious preexec_fn=os.nice route is NOT thread-safe (CPython documents it as
# able to deadlock between fork and exec) and these conversions run inside a
# ThreadPoolExecutor. A deadlock there would not even hit the except clause
# below — it would just hang until PARSER_TIMEOUT_SECONDS.
_NICE_EXE = shutil.which("nice") if os.name != "nt" else None


def _low_priority_popen_kwargs():
    """Extra subprocess kwargs that start a process at below-normal OS
    priority, so parallel PD2ModelParser runs use spare CPU but yield
    immediately to Blender and other applications."""
    if os.name == "nt":
        # BELOW_NORMAL_PRIORITY_CLASS
        return {"creationflags": subprocess.CREATE_NO_WINDOW | 0x00004000}
    # Detach from Blender's process group so a stuck parser cannot take
    # signals meant for Blender. Priority itself is handled by _nice_cmd().
    return {"start_new_session": True}


def _nice_cmd(cmd, low_priority):
    """Wrap the command in `nice` on POSIX, which is the thread-safe way to
    start the child at a lower priority."""
    if low_priority and _NICE_EXE:
        return [_NICE_EXE, "-n", "10"] + cmd
    return cmd


def convert_model_to_glb(parser_exe, model_path, glb_path, low_priority=True):
    """Run PD2ModelParser.exe SYNCHRONOUSLY (blocking) to convert a .model
    into a .glb. Returns True on success."""
    cmd = _nice_cmd([parser_exe, f"--load={model_path}",
                     f"--export={glb_path}"], low_priority)
    vlog(f"  Converting: {os.path.basename(model_path)} -> glb (blocking)")
    extra = {}
    if low_priority:
        try:
            extra = _low_priority_popen_kwargs()
        except Exception:
            extra = {}
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PARSER_TIMEOUT_SECONDS,
            **extra,
        )
    except (ValueError, OSError):
        # Priority flags rejected on this platform — retry plainly
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=PARSER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            log_error(f"  PD2ModelParser timed out on {model_path}")
            return False
        except Exception as e:
            log_error(f"  Failed to run PD2ModelParser: {e}")
            return False
    except subprocess.TimeoutExpired:
        log_error(f"  PD2ModelParser timed out on {model_path}")
        return False
    except Exception as e:
        log_error(f"  Failed to run PD2ModelParser: {e}")
        return False

    if result.returncode != 0:
        log_error(f"  PD2ModelParser exited with code {result.returncode}")
        if result.stdout:
            log_error(f"  stdout: {result.stdout[-1000:]}")
        if result.stderr:
            log_error(f"  stderr: {result.stderr[-1000:]}")
        return False

    if not os.path.isfile(glb_path) or os.path.getsize(glb_path) == 0:
        log_error("  Parser reported success but no .glb was produced")
        return False
    return True


GLB_CACHE_DIRNAME = "pd2_glb_cache"
TEX_CACHE_DIRNAME = "pd2_tex_cache"


def _glb_cache_dir():
    d = os.path.join(tempfile.gettempdir(), GLB_CACHE_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def _dir_stats(path):
    """(file count, total bytes) for one flat directory."""
    n_files = total = 0
    if os.path.isdir(path):
        try:
            entries = os.scandir(path)
        except OSError:
            return 0, 0
        with entries:
            for e in entries:
                try:
                    if e.is_file():
                        total += e.stat().st_size
                        n_files += 1
                except OSError:
                    pass
    return n_files, total


def reset_module_caches():
    """Drop every module-level cache. The texture caches in particular store
    bpy image NAMES, so after a file reload (or an Orphan Data purge) they
    can resolve to nothing, or worse to an unrelated image that happened to
    take the name. unregister() and the Clear Cache operator both call this."""
    global _hashlist_cache
    _hashlist_cache = None
    for cache in (_text_cache, _unit_object_cache, _matconfig_for_unit,
                  _matconfig_cache, _texture_cache, _normal_mode_cache):
        cache.clear()
    _reported_dds_formats.clear()


def _glb_cache_path(model_path):
    """Cache key ties the converted .glb to the exact .model file contents:
    full path + size + mtime. Any change to the .model produces a new key,
    so stale conversions can never be served."""
    try:
        st = os.stat(model_path)
    except OSError:
        return None
    key = f"{os.path.normcase(model_path)}|{st.st_size}|{st.st_mtime_ns}"
    return os.path.join(_glb_cache_dir(),
                        f"{diesel_hash(key):016x}.glb")


def convert_all_models_parallel(parser_exe, unique_paths, assets_dir, tmp_dir,
                                max_workers, progress_cb=None,
                                low_priority=True, use_cache=True):
    """Pre-pass: convert every unique .model to .glb using a pool of
    PD2ModelParser processes running concurrently. The exe is an external
    process, so N of them can run at once without touching Blender data.

    With use_cache, each converted .glb is kept in a persistent cache keyed
    by the .model's path/size/mtime, so re-importing a level skips the
    conversion cost entirely for unchanged models.
    Returns (glb_map: unit_path -> glb_path, missing: set of unit paths)."""
    glb_map = {}
    missing = set()
    jobs = []  # (model_path, glb_path, cache_path, [unit_path, ...])
    # Several unit paths routinely resolve to ONE .model (find_model_file
    # chains through .unit files), so jobs are keyed by the resolved model
    # rather than by unit path. Converting the same model two or three times
    # was pure waste, and worse: those duplicate jobs shared a cache path and
    # therefore raced each other writing the same temp file.
    by_model = {}
    n_cached = 0

    for path in unique_paths:
        model_path = find_model_file(path, assets_dir)
        if model_path is None:
            log_error(f"  .model not found for: {path}")
            missing.add(path)
            continue
        cache_path = _glb_cache_path(model_path) if use_cache else None
        if (cache_path and os.path.isfile(cache_path)
                and os.path.getsize(cache_path) > 0):
            # Cache hit: reuse the previous conversion, no exe launch at all
            glb_map[path] = cache_path
            n_cached += 1
            continue
        key = os.path.normcase(model_path)
        if key in by_model:
            by_model[key][3].append(path)
            continue
        idx = len(by_model)
        job = (model_path, os.path.join(tmp_dir, f"model_{idx:05d}.glb"),
               cache_path, [path])
        by_model[key] = job
        jobs.append(job)

    n_shared = sum(len(j[3]) - 1 for j in jobs)
    if n_cached:
        log(f"Conversion cache: {n_cached}/{len(unique_paths)} models reused "
            f"from previous imports")
    if n_shared:
        log(f"{n_shared} unit path(s) share a .model with another — "
            f"converted once each")
    if not jobs:
        return glb_map, missing

    workers = max(1, min(max_workers, len(jobs)))
    log(f"Converting {len(jobs)} unique models with {workers} parallel workers...")

    def run_one(job):
        model_path, glb_path, cache_path, unit_paths = job
        ok = convert_model_to_glb(parser_exe, model_path, glb_path,
                                  low_priority=low_priority)
        if ok and cache_path:
            # Store for future imports. Write to a PRIVATE temp name in the
            # cache dir, then atomically replace. The old code derived the
            # temp name from the cache path alone, so two workers could
            # interleave their copies into the same file before either
            # replace() ran.
            tmp_cache = None
            try:
                fd, tmp_cache = tempfile.mkstemp(
                    dir=os.path.dirname(cache_path), suffix=".tmp")
                os.close(fd)
                shutil.copyfile(glb_path, tmp_cache)
                os.replace(tmp_cache, cache_path)
            except OSError:
                # caching is best-effort; the import still proceeds
                if tmp_cache and os.path.isfile(tmp_cache):
                    try:
                        os.remove(tmp_cache)
                    except OSError:
                        pass
        return unit_paths, glb_path if ok else None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                unit_paths, glb_path = fut.result()
            except Exception as e:
                log_error(f"  conversion worker crashed: {e}")
                traceback.print_exc()
                done += 1
                continue
            for up in unit_paths:
                if glb_path:
                    glb_map[up] = glb_path
                else:
                    missing.add(up)
            done += 1
            if progress_cb:
                progress_cb(done, len(jobs))

    return glb_map, missing


def duplicate_objects(proto_objects, collection):
    """Fast duplication of an already-imported object tree. Object data
    (meshes, materials) is SHARED (linked duplicates), which is dramatically
    faster and lighter than re-running the glTF importer, and produces zero
    duplicate '.001' materials."""
    mapping = {}
    copies = []
    for o in proto_objects:
        c = o.copy()  # shares o.data
        mapping[o] = c
        copies.append(c)
        collection.objects.link(c)
    # Re-wire parents into the copied tree
    for o, c in mapping.items():
        if o.parent in mapping:
            c.parent = mapping[o.parent]
            c.matrix_parent_inverse = o.matrix_parent_inverse.copy()
        # Re-target modifiers (Armature, Mirror, etc.) at the copied tree,
        # not the prototype that gets deleted at the end of the import.
        for m in c.modifiers:
            if hasattr(m, "object") and m.object in mapping:
                m.object = mapping[m.object]
        # Same for constraints (e.g. Child Of / Copy Transforms)
        for con in c.constraints:
            if getattr(con, "target", None) in mapping:
                con.target = mapping[con.target]
    return copies


def import_glb(glb_path, collection):
    """Import a .glb with Blender's built-in (synchronous) glTF importer.
    Returns the list of newly created objects."""
    deselect_all()
    before_names = {o.name for o in bpy.data.objects}
    try:
        bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as e:
        log_error(f"  glTF import failed: {e}")
        traceback.print_exc()
        return []

    new_objects = [o for o in bpy.data.objects if o.name not in before_names]
    for o in bpy.context.selected_objects:
        if o.name not in before_names and o not in new_objects:
            new_objects.append(o)

    # Link everything into the level collection
    for o in new_objects:
        for col in list(o.users_collection):
            if col is not collection:
                col.objects.unlink(o)
        if o.name not in collection.objects:
            collection.objects.link(o)
    return new_objects


def build_unit(new_objects, name_id, collection):
    """Create the unit's root empty at absolute zero, parent all top-level
    imported objects to it (zeroing their junk transforms — top priority),
    and return the root. Child meshes below the top level are untouched.

    The 'only g_ meshes' filter is NOT applied here: it runs once per unique
    model in _import_prototype, before any duplicates exist, so every placed
    copy inherits an already-filtered tree. The old only_g_meshes parameter
    was dead (every call site passed False) and could not have worked anyway
    — it ran the filter after the top-level loop below, so the marker the
    filter sets was written after the only code that reads it."""
    root = bpy.data.objects.new(name_id, None)
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = 0.25
    collection.objects.link(root)
    brute_force_zero_transform(root)

    new_set = set(new_objects)
    top_level = [o for o in new_objects if o.parent not in new_set]

    for o in top_level:
        if o.get(KEEP_TRANSFORM_PROP):
            # Orphaned by the g_ filter (e.g. a light whose parent mesh was
            # removed) — keep its real placement, just attach to the root.
            world = local_world_matrix(o)
            reparent_keep_world(o, root, world)
            continue
        brute_force_zero_transform(o)
        o.parent = root
        o.matrix_parent_inverse.identity()

    return root


def stamp_light_shadow_flags(objects):
    """Record the shadow-projection verdict on every light BEFORE anything
    can delete the ancestors it depends on.

    light_shadows_enabled() walks the parent chain looking for a node named
    '*shadow_projection*', because the glTF importer often hangs the light
    under a parent that carries the real name. The g_ filter deletes non-g_
    MESH parents and re-parents the light to the unit root, which destroys
    that chain — so shadow-projection omnis silently lost their shadows, but
    only when 'Only Import g_ Meshes' was enabled. Stamping the answer onto
    the light while the hierarchy is still intact makes the two options
    independent again. o.copy() carries the marker to every duplicate."""
    n_flagged = 0
    for o in objects:
        if o.type != 'LIGHT' or SHADOW_FLAG_PROP in o:
            continue
        on = light_shadows_enabled(o.name, o)
        o[SHADOW_FLAG_PROP] = 1 if on else 0
        n_flagged += 1 if on else 0
    if n_flagged:
        vlog(f"  -> flagged {n_flagged} shadow-projection light(s)")


def filter_g_meshes(objects, fallback_parent=None):
    """Delete every MESH object whose name (ignoring .001 suffixes) doesn't
    start with 'g_', re-parenting surviving children first. ONLY meshes are
    ever deleted — lights, empties, and every other object type are always
    kept, with their world transforms preserved even when their parent mesh
    is removed. Returns the list of surviving objects."""
    def base_name(o):
        m = _dup_suffix_re.match(o.name)
        return m.group(1) if m else o.name

    to_delete = [o for o in objects
                 if o.type == 'MESH' and not base_name(o).startswith("g_")]
    if not to_delete:
        return list(objects)

    # Everything below is driven by NAMES, not references: bpy.data.objects.
    # remove() can invalidate other Python handles (StructRNA removed), and
    # the old code iterated the live to_delete list while deleting from it.
    delete_set = set(to_delete)
    delete_names = [o.name for o in to_delete]
    survivor_names = [o.name for o in objects if o not in delete_set]

    # Snapshot every survivor's world matrix FIRST, computed from local data
    # so it is correct even though prototypes live in an unlinked collection
    # the depsgraph never evaluates. Re-parenting one object would otherwise
    # invalidate the cached matrix_world of everything below it.
    world_cache = {}
    survivors_live = [o for o in objects if o not in delete_set]
    worlds = {o.name: local_world_matrix(o, world_cache)
              for o in survivors_live}

    n_orphaned = 0
    for o in survivors_live:
        p = o.parent
        depth = 0
        while p in delete_set and depth < _MAX_PARENT_DEPTH:
            p = p.parent
            depth += 1
        if p is not o.parent:
            reparent_keep_world(o, p if p is not None else fallback_parent,
                                worlds[o.name])
            if p is None and fallback_parent is None:
                # Now top-level: mark so build_unit does NOT zero its
                # transform like a model root — its placement is real.
                o[KEEP_TRANSFORM_PROP] = True
                n_orphaned += 1

    n_removed = 0
    for name in delete_names:
        o = bpy.data.objects.get(name)
        if o is None:
            continue
        data = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        n_removed += 1
        if data is not None and data.users == 0:
            try:
                bpy.data.meshes.remove(data)
            except (ReferenceError, RuntimeError):
                pass

    survivors = [o for o in (bpy.data.objects.get(nm) for nm in survivor_names)
                 if o is not None]
    n_lights_kept = sum(1 for o in survivors if o.type == 'LIGHT')
    vlog(f"  -> only-g_ filter removed {n_removed} mesh(es), "
         f"kept {len(survivors)} object(s) "
         f"({n_lights_kept} light(s), {n_orphaned} re-rooted)")
    # Deleting objects can invalidate other Python references (StructRNA
    # removed), so survivors were re-fetched by name.
    return survivors


# ----------------------------------------------------------------------------
# Light importing
# ----------------------------------------------------------------------------

# PD2 uses named intensity presets for light multipliers. Approximate relative
# brightness values (tunable globally with the Light Power Scale option).
# Verbatim from the game's light_intensity_db. These are much smaller and
# much more tightly clustered than the values that used to be guessed here
# (streetlight was 8.0, now 1.2), so the overall level will come in dimmer —
# raise "Light Power Scale" to compensate rather than editing this table.
LIGHT_MULTIPLIER_PRESETS = {
    "none": 0.0,
    "identity": 1.0,
    "match": 0.4,
    "candle": 0.5,
    "desklight": 0.6,
    "neonsign": 0.7,
    "flashlight": 0.8,
    "monitor": 0.9,
    "dimlight": 1.0,
    "streetlight": 1.2,
    "searchlight": 1.4,
    "reddot": 2.5,
    "sun": 3.0,
    "inside of borg queen": 6.0,
    "megatron": 8.0,
}


def parse_light_multiplier(value):
    """Multiplier can be a named preset string ('match', 'streetlight'...)
    or a plain number."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return LIGHT_MULTIPLIER_PRESETS.get(value.strip().lower(), 1.0)
    return 1.0


def parse_color(s):
    """Parse 'Vector3(r, g, b)' 0-1 color without the tiny-value zeroing used
    for transforms (a color channel of 0.00005 should stay, not matter)."""
    try:
        inner = s[s.index("(") + 1:s.rindex(")")]
        vals = [float(p.strip()) for p in inner.split(",")[:3]]
        while len(vals) < 3:
            vals.append(1.0)
        return tuple(min(max(v, 0.0), 1.0) for v in vals)
    except Exception:
        return (1.0, 1.0, 1.0)


def light_shadows_enabled(name, obj=None, extra_names=()):
    """Shadow policy: shadows OFF for every light, except shadow-projection
    omnis (e.g. 'light_omni_shadow_projection', '..._01'), which have them
    ON. Matches 'shadow_projection' anywhere in the given name, any of
    extra_names, the object's own name, its light data name, or any
    ancestor's name — the glTF importer can decorate node names (orientation
    empties, suffixes) or hang the light under a parent node that carries the
    real name, and the JSON light name can differ from the node name.

    extra_names is how the UNIT PATH gets into the decision. In PD2 the
    shadow-projection marker usually lives on the unit
    (units/lights/light_omni_shadow_projection/...), not on the JSON light
    entry, and create_unit_lights only ever saw the latter — so a light built
    from JSON came out with shadows off no matter what unit it belonged to."""
    needle = "shadow_projection"
    for candidate in (name,) + tuple(extra_names):
        if candidate and needle in str(candidate).lower():
            return True
    if obj is not None:
        try:
            stamped = obj.get(SHADOW_FLAG_PROP)
            if stamped is not None:
                # Verdict recorded while the model hierarchy was still
                # intact (see stamp_light_shadow_flags) — the ancestor walk
                # below cannot be trusted once the g_ filter has run.
                return bool(stamped)
            if needle in obj.name.lower():
                return True
            if obj.data is not None and needle in obj.data.name.lower():
                return True
            p = obj.parent
            depth = 0
            while p is not None and depth < 16:
                if needle in p.name.lower():
                    return True
                p = p.parent
                depth += 1
        except ReferenceError:
            pass
    return False


# PD2 fakes volumetric light shafts with actual cone/card GEOMETRY that the
# engine draws additively — black pixels vanish, bright ones glow. Imported
# with an ordinary opaque material they read as solid grey cones blocking the
# scene. These name fragments identify that geometry.
LIGHT_CONE_NAME_HINTS = (
    "lightcone", "light_cone", "cone_light", "lightshaft", "light_shaft",
    "lightbeam", "light_beam", "godray", "god_ray", "glow", "flare",
    "volumetric",
)


def looks_like_light_cone(obj):
    """True when the object (or one of its materials) is named like PD2's
    additive light-shaft geometry."""
    def hit(text):
        low = text.lower()
        return any(h in low for h in LIGHT_CONE_NAME_HINTS)

    m = _dup_suffix_re.match(obj.name)
    if hit(m.group(1) if m else obj.name):
        return True
    for slot in getattr(obj, "material_slots", ()):
        if slot.material is not None and hit(slot.material.name):
            return True
    return False


def _first_image_in_material(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return node.image
    return None


def make_cone_fake_volume_material(mat, strength=0.4, falloff=1.6):
    """Return a FAKE-VOLUMETRIC variant of mat.

    Rather than shading an actual volume, this is the trick engines use:
    the cone stays a surface, drawn additively, with its brightness driven
    by how much apparent depth the viewer is looking through. Looking
    straight into the cone you see through its full thickness, so it is at
    its brightest; towards the silhouette the ray clips the edge and the
    shaft fades to nothing. Layer Weight's Facing output gives exactly that
    ratio, and inverting it produces the soft rim that makes flat geometry
    read as fog.

    Two practical advantages over a real volume shader: it samples the cone
    texture through its ORIGINAL UVs (volumes have no UVs and have to fall
    back to Generated coordinates), and it costs roughly nothing to render,
    which matters on levels carrying hundreds of shafts."""
    base = mat.name if mat else "light_cone"
    m = _dup_suffix_re.match(base)
    base = m.group(1) if m else base
    glow_name = f"{base}#cone_fake_vol"
    existing = bpy.data.materials.get(glow_name)
    if existing is not None:
        return existing

    img = _first_image_in_material(mat)
    glow = bpy.data.materials.new(glow_name)
    glow.use_nodes = True
    nt = glow.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (620, 0)
    add = nt.nodes.new('ShaderNodeAddShader')
    add.location = (430, 0)
    transp = nt.nodes.new('ShaderNodeBsdfTransparent')
    transp.location = (250, 120)
    emit = nt.nodes.new('ShaderNodeEmission')
    emit.location = (250, -100)

    # --- depth fade: 1 - facing, sharpened, scaled by strength ------------
    lw = nt.nodes.new('ShaderNodeLayerWeight')
    lw.location = (-460, -260)
    lw.inputs["Blend"].default_value = CONE_LAYER_BLEND
    inv = nt.nodes.new('ShaderNodeMath')
    inv.location = (-280, -260)
    inv.operation = 'SUBTRACT'
    inv.label = f"{CONE_FADE_BIAS:g} - facing"
    inv.inputs[0].default_value = CONE_FADE_BIAS
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])

    pw = nt.nodes.new('ShaderNodeMath')
    pw.location = (-100, -260)
    pw.operation = 'POWER'
    pw.label = "Edge falloff"
    pw.inputs[1].default_value = falloff
    nt.links.new(inv.outputs[0], pw.inputs[0])

    mul = nt.nodes.new('ShaderNodeMath')
    mul.location = (70, -260)
    mul.operation = 'MULTIPLY'
    mul.label = "Strength"
    mul.inputs[1].default_value = strength
    mul.use_clamp = False
    nt.links.new(pw.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], emit.inputs["Strength"])

    if img is not None:
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.location = (-120, 60)
        tex.image = img
        tex.label = "Light cone"
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    else:
        emit.inputs["Color"].default_value = (1.0, 0.95, 0.85, 1.0)

    nt.links.new(transp.outputs[0], add.inputs[0])
    nt.links.new(emit.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])

    if hasattr(glow, "blend_method"):
        glow.blend_method = 'BLEND'
    if hasattr(glow, "surface_render_method"):
        glow.surface_render_method = 'BLENDED'
    if hasattr(glow, "shadow_method"):
        glow.shadow_method = 'NONE'
    # Backfaces must render and accumulate: seeing the far wall of the cone
    # through the near one is what sells the illusion of thickness.
    glow.use_backface_culling = False
    if hasattr(glow, "show_transparent_back"):
        glow.show_transparent_back = True
    glow["pd2_cone_glow"] = True
    return glow


def make_cone_volume_material(mat, strength=2.0, density=1.0):
    """Return a VOLUMETRIC variant of mat: the cone mesh becomes a body of
    glowing fog rather than a surface, which is what the shaft is meant to
    represent.

    Note on the texture: volume shaders have no UV coordinates in Blender —
    UVs only exist on surfaces — so the cone texture is sampled from
    Generated coordinates (the mesh's own bounding box, 0..1 on each axis).
    For the gradient-along-the-axis textures PD2 uses on light shafts that
    reads correctly; it is not a substitute for the original UV layout if
    the texture happens to be doing something more elaborate.

    The texture drives both emission colour and density, so dark parts of
    the texture thin the fog out instead of glowing black."""
    base = mat.name if mat else "light_cone"
    m = _dup_suffix_re.match(base)
    base = m.group(1) if m else base
    vol_name = f"{base}#cone_volume"
    existing = bpy.data.materials.get(vol_name)
    if existing is not None:
        return existing

    img = _first_image_in_material(mat)
    vol = bpy.data.materials.new(vol_name)
    vol.use_nodes = True
    nt = vol.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (400, 0)
    pv = nt.nodes.new('ShaderNodeVolumePrincipled')
    pv.location = (150, 0)
    pv.inputs["Emission Strength"].default_value = strength
    # Surface is left unconnected on purpose: the cone should contribute
    # nothing as a surface, only as a volume.
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])

    if img is not None:
        tc = nt.nodes.new('ShaderNodeTexCoord')
        tc.location = (-620, -60)
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.location = (-420, -60)
        tex.image = img
        tex.label = "Light cone (Generated coords — volumes have no UVs)"
        tex.extension = 'EXTEND'
        nt.links.new(tc.outputs["Generated"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], pv.inputs["Emission Color"])
        nt.links.new(tex.outputs["Color"], pv.inputs["Color"])
        # Density follows the texture's brightness so the shaft thins out
        # where the texture is dark.
        bw = nt.nodes.new('ShaderNodeRGBToBW')
        bw.location = (-180, -260)
        nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
        dens = nt.nodes.new('ShaderNodeMath')
        dens.location = (-20, -260)
        dens.operation = 'MULTIPLY'
        dens.inputs[1].default_value = density
        dens.label = "Density"
        nt.links.new(bw.outputs["Val"], dens.inputs[0])
        nt.links.new(dens.outputs[0], pv.inputs["Density"])
    else:
        pv.inputs["Color"].default_value = (1.0, 0.95, 0.85, 1.0)
        pv.inputs["Emission Color"].default_value = (1.0, 0.95, 0.85, 1.0)
        pv.inputs["Density"].default_value = density

    if hasattr(vol, "shadow_method"):
        vol.shadow_method = 'NONE'
    vol.use_backface_culling = False
    vol["pd2_cone_glow"] = True
    return vol


def apply_light_cone_mode(objects, mode, strength=0.4, density=1.0,
                          falloff=1.6):
    """Handle PD2's light-shaft geometry. 'FAKE' shades it as an additive
    surface with a view-angle depth fade, 'VOLUME' turns it into real
    glowing fog, 'HIDE' removes it from the viewport and renders, 'KEEP'
    leaves it as imported."""
    if mode == 'KEEP':
        return 0
    n_done = 0
    for o in objects:
        if o.type != 'MESH' or not looks_like_light_cone(o):
            continue
        if mode == 'HIDE':
            o.hide_viewport = True
            o.hide_render = True
            n_done += 1
            continue
        for slot in o.material_slots:
            if slot.material is None or slot.material.get("pd2_cone_glow"):
                continue
            if mode == 'VOLUME':
                slot.material = make_cone_volume_material(
                    slot.material, strength, density)
            else:
                slot.material = make_cone_fake_volume_material(
                    slot.material, strength, falloff)
        # Cones are light, not geometry: they must not cast shadows, and
        # they are single-sided cards meant to be visible from both sides.
        o.visible_shadow = False
        n_done += 1
    if n_done:
        vlog(f"  -> light cone geometry: {mode.lower()} applied to "
             f"{n_done} object(s)")
    return n_done


def apply_model_light_defaults(objects, extra_names=()):
    """Configure light nodes that came from the .glb but have no JSON
    settings: KEEP them (their placement/rotation is real level lighting),
    just apply the shadow policy. Light data may be shared between
    duplicated units — fine, since the policy is name-based and identical
    for every copy."""
    n = 0
    n_shadow = 0
    for o in objects:
        if o.type == 'LIGHT':
            sh = light_shadows_enabled(o.name, o, extra_names)
            o.data.use_shadow = sh
            n += 1
            n_shadow += 1 if sh else 0
    if n:
        vlog(f"  -> kept {n} model light(s), shadow policy applied "
            f"({n_shadow} with shadows ON)")
    return n


def create_unit_lights(unit, root, collection, power_scale, unit_objects=None,
                       unit_path=""):
    """Apply the unit's JSON 'lights' settings. The model .glb usually already
    contains the light nodes (imported with default settings but CORRECT
    position/rotation from the model hierarchy). So: match each JSON light to
    an imported light object by name and configure it in place, keeping its
    transform. Only if no matching node exists is a new light created at the
    unit root. This avoids the old duplication of one default-settings light
    (from the model) plus one unrotated custom light (from the JSON)."""
    lights_data = unit.get("lights") or {}

    def base_name(n):
        m = _dup_suffix_re.match(n)
        return m.group(1) if m else n

    # (base name lowercased, object name). Object NAMES are stored, not
    # references: deleting any object below may invalidate handles.
    available = []
    for o in (unit_objects or []):
        if o.type == 'LIGHT':
            available.append((base_name(o.name).lower(), o.name))
    # Every model light node name, kept for the shadow decision even after
    # matching has consumed the entries.
    model_light_names = [obj_name for _, obj_name in available]

    def find_match(json_name):
        """Pair a JSON light with one of the model's light nodes.

        Exact base-name equality first. If that fails, fall back to an
        unambiguous case-insensitive substring match — the JSON name is
        routinely a short label ('omni') while the glTF node keeps the full
        Diesel name ('light_omni_shadow_projection_01'), so requiring exact
        equality meant almost nothing ever matched. Every JSON light then
        got built fresh at the unit root with default settings, which both
        duplicated the model's own light node and lost the shadow verdict
        stamped onto it."""
        want = base_name(json_name).lower()
        tiers = ((lambda bn: bn == want, "model node"),
                 (lambda bn: bool(want) and (want in bn or bn in want),
                  "model node (fuzzy)"))
        for predicate, kind in tiers:
            hits = [i for i, (bn, _) in enumerate(available) if predicate(bn)]
            if kind.endswith("(fuzzy)") and len(hits) != 1:
                continue  # ambiguous — safer to build a new light
            for i in hits:
                obj = bpy.data.objects.get(available[i][1])
                if obj is not None:
                    available.pop(i)
                    return obj, kind
            for i in reversed(hits):
                available.pop(i)   # all stale; drop and try the next tier
        return None, None

    created = 0
    for key, ld in lights_data.items():
        if not isinstance(ld, dict):
            continue
        name = ld.get("name", f"light_{key}")
        match, match_kind = find_match(name)

        if not ld.get("enabled", True):
            # Disabled in JSON: also remove the model's imported light node
            if match is not None:
                for child in list(match.children):
                    reparent_keep_world(child, match.parent,
                                        local_world_matrix(child))
                data = match.data
                bpy.data.objects.remove(match, do_unlink=True)
                if data and data.users == 0:
                    bpy.data.lights.remove(data)
                vlog(f"  -> light '{name}' disabled in JSON, removed model light")
            else:
                vlog(f"  -> light '{name}' disabled in JSON, skipping")
            continue

        color = parse_color(ld.get("color", ""))
        far_m = max(sanitize_value(ld.get("far_range", 0)) / 100.0, 0.01)
        mult = parse_light_multiplier(ld.get("multiplier", 1.0))

        spot_end = 0.0
        try:
            spot_end = float(ld.get("spot_angle_end", 0) or 0)
        except (TypeError, ValueError):
            pass
        is_spot = spot_end > 0.0

        if match is not None:
            obj = match
            # Duplicated units share light data — make it unique to this unit
            # before applying per-unit settings. Free the original if this
            # was its only user (otherwise it lingers as an orphan).
            old_data = obj.data
            obj.data = old_data.copy()
            if old_data.users == 0:
                try:
                    bpy.data.lights.remove(old_data)
                except Exception:
                    pass
            obj.data.type = 'SPOT' if is_spot else 'POINT'
            # Changing type swaps the datablock class — re-fetch the handle
            # so spot attributes are actually available.
            light = obj.data
            reused = True
        else:
            light = bpy.data.lights.new(name, 'SPOT' if is_spot else 'POINT')
            obj = bpy.data.objects.new(name, light)
            collection.objects.link(obj)
            obj.parent = root
            brute_force_zero_transform(obj)
            reused = False

        # Shadow policy inputs, widest first: the JSON light name, the unit
        # path, and — when no model node was matched — the names of the
        # model's own light nodes, since a unit that ships a
        # *_shadow_projection node casts shadows regardless of what the JSON
        # calls its lights.
        shadow_names = [unit_path, root.name if root else ""]
        if match is None:
            shadow_names.extend(model_light_names)

        light.color = color
        # far_range drives both reach and power; multiplier scales brightness.
        # Power grows with range squared so far-reaching lights actually light
        # up their whole area at a similar surface brightness.
        light.energy = mult * (far_m ** 2) * power_scale
        light.use_custom_distance = True
        light.cutoff_distance = far_m
        # Diesel's lights read soft; a 5cm emitter gives razor-sharp shadow
        # edges and a hard pool of light that looks nothing like the game.
        # Scale the emitter with the light's reach, within sane bounds.
        light.shadow_soft_size = min(0.35, max(0.06, far_m * 0.03))
        light.use_shadow = light_shadows_enabled(name, obj, shadow_names)

        if is_spot:
            light.spot_size = math.radians(min(max(spot_end, 1.0), 170.0))
            # Blender's 0.15 default leaves a hard-edged cone. The engine's
            # spots fade out well before the cone boundary, so never go
            # below SPOT_BLEND_MIN even when the JSON supplies a start angle.
            blend = SPOT_BLEND_DEFAULT
            try:
                spot_start = float(ld.get("spot_angle_start", 0) or 0)
                if 0 < spot_start < spot_end:
                    blend = 1.0 - (spot_start / spot_end)
            except (TypeError, ValueError):
                pass
            light.spot_blend = min(1.0, max(SPOT_BLEND_MIN, blend))

        created += 1
        kind = "SPOT" if is_spot else "POINT"
        src = match_kind if reused else "new at root"
        shadow_state = "shadows ON" if light.use_shadow else "shadows off"
        vlog(f"  -> light '{name}' [{kind}, {src}] "
            f"color=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f}) "
            f"range={far_m:.2f}m mult={mult:g} energy={light.energy:.0f}W {shadow_state}")

    leftovers = [o for o in (bpy.data.objects.get(obj_name)
                             for _, obj_name in available)
                 if o is not None]
    if leftovers:
        apply_model_light_defaults(leftovers, [unit_path])
        vlog(f"  -> {len(leftovers)} model light(s) without JSON settings "
            f"kept as imported: {', '.join(o.name for o in leftovers)}")
    return created


def strip_imported_lights(objects):
    """Remove light objects that came from the .glb (used when no JSON light
    settings apply, so no default-settings lights linger). Children of a
    removed light are re-parented to its parent with their world transform
    preserved. Returns fresh survivor references (deletion can invalidate
    existing ones)."""
    light_names = [o.name for o in objects if o.type == 'LIGHT']
    survivor_names = [o.name for o in objects if o.type != 'LIGHT']
    light_name_set = set(light_names)
    removed = 0
    for n in light_names:
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        # Rescue children before deleting their parent
        for child in list(o.children):
            new_parent = o.parent
            depth = 0
            while (new_parent is not None
                   and new_parent.name in light_name_set
                   and depth < _MAX_PARENT_DEPTH):
                new_parent = new_parent.parent
                depth += 1
            world = local_world_matrix(child)
            reparent_keep_world(child, new_parent, world)
            if new_parent is None:
                # Nothing left above it — its placement is real, so make
                # sure a later build_unit pass does not zero it out.
                child[KEEP_TRANSFORM_PROP] = True
        data = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if data and data.users == 0:
            bpy.data.lights.remove(data)
        removed += 1
    if removed:
        vlog(f"  -> stripped {removed} imported model light(s)")
        return [o for o in (bpy.data.objects.get(n) for n in survivor_names)
                if o is not None]
    return list(objects)


# ----------------------------------------------------------------------------
# Texture / material importing (unit -> object -> material_config chain)
# ----------------------------------------------------------------------------
#
# Pipeline (per imported .model):
#   <unit_path>.unit         ->  <object file="path/to/object"/>
#   path/to/object.object    ->  <diesel materials="path/to/material_config"/>
#   *.material_config        ->  <material name="..."> texture entries
#
# Only materials whose names exactly match the material slots of the model
# being imported are touched (anti model-mixing: multi-model configs never
# leak materials across models). Everything is rebuilt through one shared
# "PD2 Shader" node group, so textures just plug into the group.
#
# Render-template flags honored: ALPHA_MASKED / OPACITY / opacity: templates,
# SELF_ILLUMINATION (+BLOOM), GSMA (material_texture), NORMALMAP,
# CUBE_ENVIRONMENT_MAPPING. RL_*, CONTOUR, DEPTH_SCALING etc. are ignored.

PD2_NODE_GROUP_NAME = "PD2 Shader"

_object_in_unit_re = _unit_object_re  # same <object file="..."/> pattern
_diesel_materials_re = re.compile(
    r'<diesel\b[^>]*\bmaterials\s*=\s*"([^"]+)"', re.IGNORECASE)
_material_open_re = re.compile(r'<material\b[^>]*>', re.IGNORECASE)
# Some .unit files name their material_config directly instead of (or as
# well as) inheriting the one the .object points at. When present this is
# an override and wins over the .object's <diesel materials="..."/>.
_unit_matconfig_re = re.compile(
    r'<material_config\b[^>]*\b(?:file|name)\s*=\s*"([^"]+)"', re.IGNORECASE)
# Last-ditch scan for a materials="..." attribute on ANY tag, used when an
# .object file has no <diesel> element. Some objects hang the reference off
# a different tag, and requiring <diesel> meant those models resolved no
# config at all and came through untextured.
_loose_materials_re = re.compile(
    r'\bmaterials\s*=\s*"([^"]+)"', re.IGNORECASE)
_attr_re = re.compile(r'([A-Za-z_][\w]*)\s*=\s*"([^"]*)"')

_matconfig_cache = {}   # abs path -> {mat_name_lower: mat_dict}
_texture_cache = {}     # diesel path -> bpy image name or None
_normal_mode_cache = {}  # image name -> 0.0 (RG) or 1.0 (AG / DXT5nm)

TEXTURE_EXTS = (".texture", ".dds", ".tga", ".png", ".jpg", ".bmp")


def _asset_file(assets_dir, diesel_path, exts):
    """Resolve an extension-less diesel path to an existing file."""
    rel = diesel_path.strip().replace("\\", "/").strip("/")
    base = os.path.normpath(os.path.join(assets_dir, rel.replace("/", os.sep)))
    for ext in exts:
        cand = base if base.lower().endswith(ext) else base + ext
        if os.path.isfile(cand):
            return cand
    return None


def find_material_config_for_unit(unit_path, assets_dir):
    """Follow  .unit -> .object -> <diesel materials=...>  and return the
    absolute path of the .material_config file (or None). The .unit chain
    may hop through further .unit files, mirroring find_model_file."""
    ck = (unit_path, assets_dir)
    if ck in _matconfig_for_unit:
        return _matconfig_for_unit[ck]
    result = _find_material_config_uncached(unit_path, assets_dir)
    _matconfig_for_unit[ck] = result
    return result


def _find_material_config_uncached(unit_path, assets_dir):
    """Walk  .unit -> .object -> <diesel materials="..."/>  and return the
    material_config path.

    Resolution order, first hit wins:

      1. <unit_path>.material_config — a config file sitting right next to
         the unit under exactly the unit's own name. Checked before
         anything is parsed, because when it exists it is essentially
         always the right answer and it costs one stat call.
      2. The .unit's own <material_config file="..."/>, which overrides
         whatever the .object points at.
      3. The .object's <diesel materials="..."/>.
      4. Any materials="..." attribute anywhere in the .object, for objects
         that don't wrap the reference in a <diesel> tag.

    Steps 2-4 repeat for each hop of the .unit -> .object chain, exactly
    like find_model_file walks it. If every explicit reference comes up
    empty, a final pass looks for a same-named .material_config beside each
    path visited and beside the resolved .model, so a dangling or absent
    reference no longer leaves the model completely untextured."""
    # --- 1. a .material_config carrying the unit's own name ---------------
    mc = _asset_file(assets_dir, unit_path, (".material_config",))
    if mc:
        vlog(f"  material_config by unit name: {unit_path}")
        return mc

    seen = set()
    visited = []
    dangling = []
    path = unit_path

    while path and path not in seen:
        seen.add(path)
        visited.append(path)

        # --- 2. a material_config named by the .unit itself ---------------
        unit_file = _asset_file(assets_dir, path, (".unit",))
        if unit_file:
            text = _read_text_tolerant(unit_file)
            um = _unit_matconfig_re.search(text) if text else None
            if um:
                mc = _asset_file(assets_dir, um.group(1),
                                 (".material_config",))
                if mc:
                    vlog(f"  material_config via .unit override: "
                         f"{um.group(1)}")
                    return mc
                dangling.append(um.group(1))

        obj_ref = _object_path_from_unit(path, assets_dir)
        candidates = []
        if obj_ref:
            candidates.append(obj_ref)
        candidates.append(path)  # some assets: .object shares the unit name
        for ref in candidates:
            if ref not in visited:
                visited.append(ref)
            obj_file = _asset_file(assets_dir, ref, (".object",))
            if not obj_file:
                continue
            text = _read_text_tolerant(obj_file)
            if not text:
                continue
            # --- 3. the <diesel materials="..."/> reference ---------------
            m = _diesel_materials_re.search(text)
            how = ".object <diesel>"
            if not m:
                # --- 4. no <diesel> tag: take any materials="..." --------
                m = _loose_materials_re.search(text)
                how = ".object loose materials="
            if m:
                mc = _asset_file(assets_dir, m.group(1),
                                 (".material_config",))
                if mc:
                    vlog(f"  material_config via {how}: {m.group(1)}")
                    return mc
                # Referenced but absent from disk. Don't give up here —
                # returning None at this point was why a single bad
                # reference left the model untextured even when a usable
                # config was sitting right beside it.
                dangling.append(m.group(1))
        # Follow the unit chain one hop deeper (obj_ref is already resolved
        # above, no need to re-read the .unit)
        if obj_ref is None or obj_ref == path:
            break
        path = obj_ref

    # --- fallback: same-name .material_config beside anything visited -----
    for ref in visited:
        mc = _asset_file(assets_dir, ref, (".material_config",))
        if mc:
            vlog(f"  material_config by sibling name: {ref}")
            return mc
    model_file = find_model_file(unit_path, assets_dir)
    if model_file:
        cand = os.path.splitext(model_file)[0] + ".material_config"
        if os.path.isfile(cand):
            vlog(f"  material_config beside the .model: "
                 f"{os.path.basename(cand)}")
            return cand

    for ref in dangling:
        log_error(f"  material_config not found on disk: {ref}")
    return None


def parse_material_config(mc_path):
    """Parse a .material_config into {name_lower: material dict}. Uses a
    tolerant regex scan (these files vary wildly in formatting and are not
    always well-formed XML). Each material dict has:
      name, render_template, textures {slot_name: diesel_path}"""
    cached = _matconfig_cache.get(mc_path)
    if cached is not None:
        return cached
    mats = {}
    text = _read_text_tolerant(mc_path)
    if text:
        # Split on <material ...> openings; each chunk runs to the next one
        opens = list(_material_open_re.finditer(text))
        for i, m in enumerate(opens):
            end = opens[i + 1].start() if i + 1 < len(opens) else len(text)
            attrs = dict(_attr_re.findall(m.group(0)))
            name = attrs.get("name", "")
            if not name:
                continue
            body = text[m.end():end]
            textures = {}
            # Child elements like <diffuse_texture file="..."/> — also accept
            # src=; slot is the element tag name.
            for tm in re.finditer(
                    r'<\s*([A-Za-z_][\w]*)\b[^>]*\b(?:file|src)\s*=\s*"([^"]+)"',
                    body):
                slot = tm.group(1).lower()
                if slot.endswith("_texture") or slot == "texture":
                    textures[slot] = tm.group(2).strip()
            variables = {}
            for vm in re.finditer(
                    r'<\s*variable\b([^>]*)/?>', body):
                va = dict(_attr_re.findall(vm.group(1)))
                if va.get("name"):
                    variables[va["name"].lower()] = va.get("value", "")
            mats[name.lower()] = {
                "name": name,
                "render_template": attrs.get("render_template", ""),
                "textures": textures,
                "variables": variables,
            }
    _matconfig_cache[mc_path] = mats
    return mats


# ---- BC4/BC5 (ATI1/ATI2/3Dc) decoding ------------------------------------
# Blender's DDS loader only understands DXT1/3/5 and uncompressed data. PD2
# normal maps are usually stored as ATI2/BC5 (two-channel 3Dc), which makes
# Blender print "Unable to find a suitable DXT compression, falling back to
# uncompressed" and hand back garbage channels — the reason those maps came
# in red-only or red/white. These are decoded here (numpy, vectorised) and
# cached as PNGs, so Blender gets a clean X=red / Y=green normal map.

_BC45_FOURCC = {b"ATI1": 1, b"BC4U": 1, b"BC4S": 1,
                b"ATI2": 2, b"BC5U": 2, b"BC5S": 2, b"A2XY": 2}


# DX10-extended DDS: BC4/BC5 are identified by a DXGI format number in a
# 20-byte header that follows the normal one.
_DXGI_BC = {80: 1, 81: 1, 82: 1,      # BC4 typeless/unorm/snorm
            83: 2, 84: 2, 85: 2}      # BC5 typeless/unorm/snorm


def _dds_header(path):
    """Return (fourcc-or-synthetic tag, width, height, data_offset) or None.
    DX10-extended files are mapped onto the ATI1/ATI2 tags so BC4/BC5 in
    either container decodes the same way."""
    try:
        with open(path, "rb") as f:
            head = f.read(148)
    except OSError:
        return None
    if len(head) < 128 or head[:4] != b"DDS ":
        return None
    height, width = struct.unpack_from("<2I", head, 12)
    fourcc = head[84:88]
    if fourcc == b"DX10" and len(head) >= 148:
        dxgi = struct.unpack_from("<I", head, 128)[0]
        ch = _DXGI_BC.get(dxgi)
        if ch:
            return (b"ATI1" if ch == 1 else b"ATI2"), width, height, 148
        vlog(f"  DDS DX10 dxgiFormat {dxgi} not handled: "
             f"{os.path.basename(path)}")
        return b"DX10", width, height, 148
    return fourcc, width, height, 128


def _bc_channel_planes(raw, n_blocks, stride, offset):
    """Decode one BC4-style channel from n_blocks 8-byte chunks.
    Returns float32 array (n_blocks, 16) of 0..255 values."""
    import numpy as np
    buf = np.frombuffer(raw, dtype=np.uint8,
                        count=n_blocks * stride).reshape(n_blocks, stride)
    b = buf[:, offset:offset + 8]
    e0 = b[:, 0].astype(np.float32)
    e1 = b[:, 1].astype(np.float32)

    bits = np.zeros(n_blocks, dtype=np.uint64)
    for i in range(6):
        bits |= b[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    idx = np.empty((n_blocks, 16), dtype=np.int64)
    for i in range(16):
        idx[:, i] = ((bits >> np.uint64(3 * i)) & np.uint64(7)).astype(np.int64)

    pal = np.zeros((n_blocks, 8), dtype=np.float32)
    pal[:, 0] = e0
    pal[:, 1] = e1
    gt = e0 > e1
    # 8-value interpolation when e0 > e1, otherwise 6 values + 0/255
    for i in range(1, 7):
        pal[:, i + 1] = np.where(gt, ((7 - i) * e0 + i * e1) / 7.0, 0.0)
    ngt = ~gt
    if ngt.any():
        for i in range(1, 5):
            pal[ngt, i + 1] = ((5 - i) * e0[ngt] + i * e1[ngt]) / 5.0
        pal[ngt, 6] = 0.0
        pal[ngt, 7] = 255.0
    return np.take_along_axis(pal, idx, axis=1)


def _blocks_to_image(planes, bw, bh, width, height):
    """(n_blocks, 16) -> (height, width) image array."""
    img = planes.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3)
    img = img.reshape(bh * 4, bw * 4)
    return img[:height, :width]


_reported_dds_formats = set()


def _report_dds_format(path):
    """Log the pixel-format details of any DDS we don't decode ourselves,
    once per distinct format. Blender printing 'Unable to find a suitable
    DXT compression' means IT could not read the file either, so knowing
    the exact format is what makes a decoder possible."""
    try:
        with open(path, "rb") as f:
            head = f.read(148)
    except OSError:
        return
    if len(head) < 128:
        return
    pf_flags, fourcc = struct.unpack_from("<I", head, 80)[0], head[84:88]
    bitcount = struct.unpack_from("<I", head, 88)[0]
    masks = struct.unpack_from("<4I", head, 92)
    dxgi = (struct.unpack_from("<I", head, 128)[0]
            if fourcc == b"DX10" and len(head) >= 148 else None)
    key = (fourcc, pf_flags, bitcount, masks, dxgi)
    if key in _reported_dds_formats:
        return
    _reported_dds_formats.add(key)
    log(f"  UNDECODED DDS FORMAT: {os.path.basename(path)} "
        f"fourcc={fourcc!r} pf_flags=0x{pf_flags:x} bits={bitcount} "
        f"masks={[hex(m) for m in masks]}"
        + (f" dxgiFormat={dxgi}" if dxgi is not None else ""))


def _decode_bc45_texture(src, channels):
    """Decode an ATI1/ATI2 .dds into a cached PNG. Returns the PNG path or
    None. BC5 output: R = X, G = Y, B = reconstructed Z, A = 1."""
    info = _dds_header(src)
    if info is None:
        return None
    _fourcc, width, height, data_off = info
    if width <= 0 or height <= 0:
        return None
    cache_dir = os.path.join(tempfile.gettempdir(), TEX_CACHE_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        st = os.stat(src)
        key = f"{os.path.normcase(src)}|{st.st_size}|{st.st_mtime_ns}|bc{channels}"
    except OSError:
        return None
    png_path = os.path.join(cache_dir, f"{diesel_hash(key):016x}_dec.png")
    if os.path.isfile(png_path) and os.path.getsize(png_path) > 0:
        return png_path

    try:
        import numpy as np
    except ImportError:
        log_error("  numpy unavailable — cannot decode BC5 normal maps")
        return None

    stride = 8 * channels
    bw = (width + 3) // 4
    bh = (height + 3) // 4
    n_blocks = bw * bh
    try:
        with open(src, "rb") as f:
            f.seek(data_off)
            raw = f.read(n_blocks * stride)
        if len(raw) < n_blocks * stride:
            return None
        x = _blocks_to_image(_bc_channel_planes(raw, n_blocks, stride, 0),
                             bw, bh, width, height) / 255.0
        if channels == 2:
            y = _blocks_to_image(
                _bc_channel_planes(raw, n_blocks, stride, 8),
                bw, bh, width, height) / 255.0
        else:
            y = x

        nx = x * 2.0 - 1.0
        ny = y * 2.0 - 1.0
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        rgba = np.empty((height, width, 4), dtype=np.float32)
        rgba[..., 0] = x
        rgba[..., 1] = y
        rgba[..., 2] = nz * 0.5 + 0.5
        rgba[..., 3] = 1.0
        # DDS rows are top-down; Blender images are bottom-up
        rgba = rgba[::-1]

        img = bpy.data.images.new(os.path.basename(png_path), width, height,
                                  alpha=False, float_buffer=False)
        img.colorspace_settings.name = 'Non-Color'
        img.pixels.foreach_set(rgba.ravel())
        img.filepath_raw = png_path
        img.file_format = 'PNG'
        img.save()
        bpy.data.images.remove(img)
        vlog(f"  decoded BC{channels} texture -> {os.path.basename(png_path)}")
        return png_path
    except Exception as e:
        log_error(f"  BC{channels} decode failed for {src}: {e}")
        return None


def _load_texture(diesel_path, assets_dir, is_color):
    """Load a game texture as a bpy Image (cached). .texture files are DDS
    payloads with a renamed extension, so they're copied to a .dds in the
    cache dir first so Blender's loader accepts them."""
    # Key includes the colour role: the same .texture is often used as sRGB
    # colour in one material and as Non-Color data in another. Keyed on the
    # path alone, whichever loaded second silently flipped the colorspace of
    # the shared datablock for everyone already using it.
    key = (diesel_path.lower(), bool(is_color))
    if key in _texture_cache:
        name = _texture_cache[key]
        return bpy.data.images.get(name) if name else None
    src = _asset_file(assets_dir, diesel_path, TEXTURE_EXTS)
    img = None
    if src:
        load_path = src
        if src.lower().endswith(".texture"):
            dds_dir = os.path.join(tempfile.gettempdir(), TEX_CACHE_DIRNAME)
            os.makedirs(dds_dir, exist_ok=True)
            dds_path = os.path.join(
                dds_dir, f"{diesel_hash(os.path.normcase(src)):016x}_"
                         f"{os.path.basename(src)[:-8]}.dds")
            try:
                if not os.path.isfile(dds_path):
                    shutil.copyfile(src, dds_path)
                load_path = dds_path
            except OSError:
                load_path = src
        # ATI1/ATI2 (BC4/BC5) can't be read by Blender — decode ourselves
        hdr = _dds_header(load_path)
        if hdr and hdr[0] in _BC45_FOURCC:
            dec = _decode_bc45_texture(load_path, _BC45_FOURCC[hdr[0]])
            if dec:
                load_path = dec
        elif hdr and hdr[0] not in (b"DXT1", b"DXT3", b"DXT5"):
            _report_dds_format(load_path)
        want_cs = "sRGB" if is_color else "Non-Color"
        try:
            img = bpy.data.images.load(load_path, check_existing=True)
            if img.get("pd2_colorspace") not in (None, want_cs):
                # Already in use as the other role — take a private copy so
                # the two uses cannot overwrite each other's colorspace.
                img = img.copy()
            img.colorspace_settings.name = want_cs
            img["pd2_colorspace"] = want_cs
            # Diesel textures pack unrelated data in alpha (specular,
            # opacity, normal X); straight alpha would let Blender
            # premultiply/associate it with the colour channels.
            try:
                img.alpha_mode = 'CHANNEL_PACKED'
            except (AttributeError, TypeError):
                pass
        except Exception as e:
            log_error(f"  failed to load texture {src}: {e}")
            img = None
    else:
        vlog(f"  texture not found: {diesel_path}")
    _texture_cache[key] = img.name if img else None
    return img


def _is_cubemap_strip(diesel_path, assets_dir):
    """PD2 cubemaps are stored as a 1x6 vertical strip of faces, which is
    non-power-of-two — Blender can't load it and it wouldn't map correctly
    as an equirectangular environment anyway. Detect and skip those."""
    src = _asset_file(assets_dir, diesel_path, TEXTURE_EXTS)
    if not src:
        return False
    hdr = _dds_header(src)
    if not hdr:
        return False
    _fc, w, h, _off = hdr
    return (h == w * 6) or (w == h * 6)


def _detect_normal_mode(img):
    """Classify the normal map's channel layout by sampling ~256 pixels.
    Returns (swap_xy, x_from_alpha):

    Case A — only RED carries data (green & blue black):
        Y = red,   X = alpha          -> (1.0, 1.0)
    Everything else:
        Y = green, X = alpha          -> (0.0, 1.0)

    X ALWAYS comes from the alpha channel — every branch returns
    x_from_alpha = 1.0, so only the Y source is actually detected. PD2's
    normal maps store X in alpha, and trying to auto-detect an RGB layout
    for the exceptions did more harm than good: any map whose red or blue
    channel carried DXT compression noise fell through to the RGB branch and
    got X wired to red, which broke maps that were previously fine.

    Z is always reconstructed in the node group."""
    if img is None:
        return (0.0, 1.0)
    cached = _normal_mode_cache.get(img.name)
    if cached is not None:
        return cached
    mode = (0.0, 1.0)
    try:
        w, h = img.size
        ch = img.channels
        if w and h and ch >= 3:
            px = img.pixels  # flat float list, `ch` values per pixel
            n = w * h
            step = max(1, n // 256)  # ~256 samples
            r_sum = g_sum = b_sum = 0.0
            r_min = g_min = b_min = 1.0
            r_max = g_max = b_max = 0.0
            cnt = 0
            for i in range(0, n, step):
                base = i * ch
                r, g, b = px[base], px[base + 1], px[base + 2]
                r_sum += r; g_sum += g; b_sum += b
                r_min = min(r_min, r); r_max = max(r_max, r)
                g_min = min(g_min, g); g_max = max(g_max, g)
                b_min = min(b_min, b); b_max = max(b_max, b)
                cnt += 1
            cnt = max(cnt, 1)
            r_mean, g_mean, b_mean = r_sum / cnt, g_sum / cnt, b_sum / cnt
            r_range = r_max - r_min
            g_range = g_max - g_min
            b_range = b_max - b_min

            g_black = g_mean < 0.12 and g_range < 0.15
            b_black = b_mean < 0.12 and b_range < 0.15
            r_has_data = r_range > 0.05 or (0.2 < r_mean < 0.8)

            if g_black and b_black and r_has_data:
                # Only red carries data -> Y comes from red instead of green
                mode = (1.0, 1.0)
            else:
                mode = (0.0, 1.0)

            vlog(f"  normal '{img.name}': swap_xy={mode[0]:g} "
                 f"x_from_alpha={mode[1]:g} | "
                 f"R {r_mean:.2f}/{r_range:.2f} G {g_mean:.2f}/{g_range:.2f} "
                 f"B {b_mean:.2f}/{b_range:.2f}")
    except Exception as e:
        log_error(f"  normal-map detection failed for {img.name}: {e}")
        mode = (0.0, 1.0)
    _normal_mode_cache[img.name] = mode
    return mode


def _ng_new_node(ng, type_, x, y, **props):
    n = ng.nodes.new(type_)
    n.location = (x, y)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def get_pd2_node_group():
    """Build (once) the shared 'PD2 Shader' node group.

    Inputs:
      Diffuse (color), Diffuse Alpha, Opacity (fac, default 1),
      GSMA Color, GSMA Alpha, Has GSMA,
      Normal Color, Normal Alpha, Has Normal, Normal AG Mode,
      Cube Reflection, Spec From Diffuse Alpha,
      Alpha Mode (0 off / 1 masked / 2 blend),
      Self Illumination, Illum Bloom
    Output: BSDF (shader)

    GSMA channels: R gloss, G specular, B cubemap mask, A opacity.
    Normal maps: OpenGL, Z reconstructed from X/Y; AG Mode swaps X source
    to the alpha channel for legacy DXT5nm maps."""
    ng = bpy.data.node_groups.get(PD2_NODE_GROUP_NAME)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(PD2_NODE_GROUP_NAME, 'ShaderNodeTree')
    iface = ng.interface

    def sock_in(name, stype, default=None, minmax=None):
        s = iface.new_socket(name=name, in_out='INPUT', socket_type=stype)
        if default is not None:
            s.default_value = default
        if minmax and hasattr(s, "min_value"):
            s.min_value, s.max_value = minmax
        return s

    sock_in("Diffuse", 'NodeSocketColor', (0.8, 0.8, 0.8, 1.0))
    sock_in("Diffuse Alpha", 'NodeSocketFloat', 1.0, (0.0, 1.0))
    sock_in("Opacity", 'NodeSocketFloat', 1.0, (0.0, 1.0))
    sock_in("GSMA Color", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0))
    sock_in("GSMA Alpha", 'NodeSocketFloat', 1.0, (0.0, 1.0))
    sock_in("Has GSMA", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Normal Color", 'NodeSocketColor', (0.5, 0.5, 1.0, 1.0))
    sock_in("Normal Alpha", 'NodeSocketFloat', 1.0, (0.0, 1.0))
    sock_in("Has Normal", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Normal Swap XY", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Normal X From Alpha", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Normal Strength", 'NodeSocketFloat', 1.0, (0.0, 10.0))
    sock_in("Base Roughness", 'NodeSocketFloat', 0.65, (0.0, 1.0))
    sock_in("Illum Tint", 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0))
    sock_in("Specular Strength", 'NodeSocketFloat', 1.5, (0.0, 10.0))
    sock_in("Gloss Boost", 'NodeSocketFloat', 1.5, (0.05, 10.0))
    sock_in("Cube Reflection", 'NodeSocketFloat', 0.0, (0.0, 2.0))
    sock_in("Cubemap", 'NodeSocketColor', (0.45, 0.48, 0.52, 1.0))
    sock_in("Metallic", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Spec From Diffuse Alpha", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Alpha Mode", 'NodeSocketFloat', 0.0, (0.0, 2.0))
    sock_in("Clip Threshold", 'NodeSocketFloat', 0.15, (0.0, 1.0))
    sock_in("Fresnel Strength", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Fresnel Bias", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Fresnel Scale", 'NodeSocketFloat', 1.0, (0.0, 10.0))
    sock_in("Fresnel Power", 'NodeSocketFloat', 2.0, (0.05, 16.0))
    sock_in("Alpha Direct", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    sock_in("Self Illumination", 'NodeSocketFloat', 0.0, (0.0, 100.0))
    sock_in("Illum Bloom", 'NodeSocketFloat', 0.0, (0.0, 1.0))
    iface.new_socket(name="BSDF", in_out='OUTPUT',
                     socket_type='NodeSocketShader')

    links = ng.links
    gi = _ng_new_node(ng, 'NodeGroupInput', -1400, 0)
    go = _ng_new_node(ng, 'NodeGroupOutput', 1000, 0)

    # --- GSMA split ---
    gsep = _ng_new_node(ng, 'ShaderNodeSeparateColor', -1100, -150)
    links.new(gi.outputs["GSMA Color"], gsep.inputs["Color"])

    # Roughness = 1 - gloss (only when GSMA present, else 0.65 default)
    inv = _ng_new_node(ng, 'ShaderNodeMath', -900, -100,
                       operation='SUBTRACT')
    inv.inputs[0].default_value = 1.0
    links.new(gsep.outputs["Red"], inv.inputs[1])
    rough_mix = _ng_new_node(ng, 'ShaderNodeMix', -700, -100,
                             data_type='FLOAT')
    links.new(gi.outputs["Base Roughness"], rough_mix.inputs["A"])
    links.new(gi.outputs["Has GSMA"], rough_mix.inputs["Factor"])
    links.new(inv.outputs[0], rough_mix.inputs["B"])
    # Gloss Boost divides roughness: higher = glossier, less matte
    rough_boost = _ng_new_node(ng, 'ShaderNodeMath', -540, -100,
                               operation='DIVIDE', use_clamp=True)
    links.new(rough_mix.outputs["Result"], rough_boost.inputs[0])
    links.new(gi.outputs["Gloss Boost"], rough_boost.inputs[1])

    # Specular: GSMA green when present, else optionally diffuse alpha
    spec_da = _ng_new_node(ng, 'ShaderNodeMix', -900, -300,
                           data_type='FLOAT')
    spec_da.inputs["A"].default_value = 0.5
    links.new(gi.outputs["Spec From Diffuse Alpha"],
              spec_da.inputs["Factor"])
    links.new(gi.outputs["Diffuse Alpha"], spec_da.inputs["B"])
    spec_mix = _ng_new_node(ng, 'ShaderNodeMix', -700, -300,
                            data_type='FLOAT')
    links.new(gi.outputs["Has GSMA"], spec_mix.inputs["Factor"])
    links.new(spec_da.outputs["Result"], spec_mix.inputs["A"])
    links.new(gsep.outputs["Green"], spec_mix.inputs["B"])

    # Cubemap reflection strength = Cube Reflection * (GSMA ? blue : 1)
    cube_mask = _ng_new_node(ng, 'ShaderNodeMix', -700, -480,
                             data_type='FLOAT')
    cube_mask.inputs["A"].default_value = 1.0
    links.new(gi.outputs["Has GSMA"], cube_mask.inputs["Factor"])
    links.new(gsep.outputs["Blue"], cube_mask.inputs["B"])
    cube_amt = _ng_new_node(ng, 'ShaderNodeMath', -500, -480,
                            operation='MULTIPLY')
    links.new(gi.outputs["Cube Reflection"], cube_amt.inputs[0])
    links.new(cube_mask.outputs["Result"], cube_amt.inputs[1])

    # --- Normal map: channel select + Z reconstruction (OpenGL, no G flip)
    nsep = _ng_new_node(ng, 'ShaderNodeSeparateColor', -1100, -700)
    links.new(gi.outputs["Normal Color"], nsep.inputs["Color"])
    # Channel select: standard maps are X=red / Y=green ("orange" maps);
    # legacy red-dominant maps store Y in RED (Swap XY = 1 swaps them).
    x_rg = _ng_new_node(ng, 'ShaderNodeMix', -1000, -620,
                        data_type='FLOAT')
    links.new(gi.outputs["Normal Swap XY"], x_rg.inputs["Factor"])
    links.new(nsep.outputs["Red"], x_rg.inputs["A"])
    links.new(nsep.outputs["Green"], x_rg.inputs["B"])
    x_sel = _ng_new_node(ng, 'ShaderNodeMix', -880, -650,
                         data_type='FLOAT')
    links.new(gi.outputs["Normal X From Alpha"], x_sel.inputs["Factor"])
    links.new(x_rg.outputs["Result"], x_sel.inputs["A"])
    links.new(gi.outputs["Normal Alpha"], x_sel.inputs["B"])
    y_sel = _ng_new_node(ng, 'ShaderNodeMix', -900, -800,
                         data_type='FLOAT')
    links.new(gi.outputs["Normal Swap XY"], y_sel.inputs["Factor"])
    links.new(nsep.outputs["Green"], y_sel.inputs["A"])
    links.new(nsep.outputs["Red"], y_sel.inputs["B"])
    # x,y in [-1,1]
    xm = _ng_new_node(ng, 'ShaderNodeMath', -700, -650,
                      operation='MULTIPLY_ADD')
    xm.inputs[1].default_value = 2.0
    xm.inputs[2].default_value = -1.0
    links.new(x_sel.outputs["Result"], xm.inputs[0])
    ym = _ng_new_node(ng, 'ShaderNodeMath', -700, -800,
                      operation='MULTIPLY_ADD')
    ym.inputs[1].default_value = 2.0
    ym.inputs[2].default_value = -1.0
    links.new(y_sel.outputs["Result"], ym.inputs[0])
    # z = sqrt(max(0, 1 - x^2 - y^2))
    x2 = _ng_new_node(ng, 'ShaderNodeMath', -520, -650, operation='MULTIPLY')
    links.new(xm.outputs[0], x2.inputs[0]); links.new(xm.outputs[0], x2.inputs[1])
    y2 = _ng_new_node(ng, 'ShaderNodeMath', -520, -800, operation='MULTIPLY')
    links.new(ym.outputs[0], y2.inputs[0]); links.new(ym.outputs[0], y2.inputs[1])
    s1 = _ng_new_node(ng, 'ShaderNodeMath', -360, -700, operation='SUBTRACT')
    s1.inputs[0].default_value = 1.0
    links.new(x2.outputs[0], s1.inputs[1])
    s2 = _ng_new_node(ng, 'ShaderNodeMath', -360, -840, operation='SUBTRACT')
    links.new(s1.outputs[0], s2.inputs[0]); links.new(y2.outputs[0], s2.inputs[1])
    zmax = _ng_new_node(ng, 'ShaderNodeMath', -200, -760, operation='MAXIMUM')
    zmax.inputs[1].default_value = 0.0
    links.new(s2.outputs[0], zmax.inputs[0])
    z = _ng_new_node(ng, 'ShaderNodeMath', -60, -760, operation='SQRT')
    links.new(zmax.outputs[0], z.inputs[0])
    # back to color space [0,1]
    xr = _ng_new_node(ng, 'ShaderNodeMath', 80, -650,
                      operation='MULTIPLY_ADD')
    xr.inputs[1].default_value = 0.5; xr.inputs[2].default_value = 0.5
    links.new(xm.outputs[0], xr.inputs[0])
    yr = _ng_new_node(ng, 'ShaderNodeMath', 80, -760,
                      operation='MULTIPLY_ADD')
    yr.inputs[1].default_value = 0.5; yr.inputs[2].default_value = 0.5
    links.new(ym.outputs[0], yr.inputs[0])
    zr = _ng_new_node(ng, 'ShaderNodeMath', 80, -870,
                      operation='MULTIPLY_ADD')
    zr.inputs[1].default_value = 0.5; zr.inputs[2].default_value = 0.5
    links.new(z.outputs[0], zr.inputs[0])
    ncomb = _ng_new_node(ng, 'ShaderNodeCombineColor', 230, -760)
    links.new(xr.outputs[0], ncomb.inputs["Red"])
    links.new(yr.outputs[0], ncomb.inputs["Green"])
    links.new(zr.outputs[0], ncomb.inputs["Blue"])
    nmap = _ng_new_node(ng, 'ShaderNodeNormalMap', 380, -760)
    links.new(ncomb.outputs["Color"], nmap.inputs["Color"])
    nstr = _ng_new_node(ng, 'ShaderNodeMath', 380, -600,
                        operation='MULTIPLY')
    links.new(gi.outputs["Has Normal"], nstr.inputs[0])
    links.new(gi.outputs["Normal Strength"], nstr.inputs[1])
    links.new(nstr.outputs[0], nmap.inputs["Strength"])

    # --- Principled ---
    bsdf = _ng_new_node(ng, 'ShaderNodeBsdfPrincipled', 560, 100)
    links.new(gi.outputs["Diffuse"], bsdf.inputs["Base Color"])
    links.new(rough_boost.outputs[0], bsdf.inputs["Roughness"])
    spec_sock = ("Specular IOR Level"
                 if "Specular IOR Level" in bsdf.inputs else "Specular")
    spec_boost = _ng_new_node(ng, 'ShaderNodeMath', -540, -300,
                              operation='MULTIPLY', use_clamp=True)
    links.new(spec_mix.outputs["Result"], spec_boost.inputs[0])
    links.new(gi.outputs["Specular Strength"], spec_boost.inputs[1])
    links.new(spec_boost.outputs[0], bsdf.inputs[spec_sock])
    links.new(gi.outputs["Metallic"], bsdf.inputs["Metallic"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    # Emission: diffuse * self illumination (bloom pushes strength higher)
    if "Emission Color" in bsdf.inputs:
        em_tint = _ng_new_node(ng, 'ShaderNodeMix', 230, 160)
        em_tint.data_type = 'RGBA'
        em_tint.blend_type = 'MULTIPLY'
        _ec_in = [s for s in em_tint.inputs if s.type == 'RGBA']
        _ec_out = [s for s in em_tint.outputs if s.type == 'RGBA']
        em_tint.inputs[0].default_value = 1.0
        links.new(gi.outputs["Diffuse"], _ec_in[0])
        links.new(gi.outputs["Illum Tint"], _ec_in[1])
        links.new(_ec_out[0], bsdf.inputs["Emission Color"])
    bloom_boost = _ng_new_node(ng, 'ShaderNodeMath', 230, 300,
                               operation='MULTIPLY_ADD')
    bloom_boost.inputs[1].default_value = 4.0  # bloom multiplies strength x5
    links.new(gi.outputs["Illum Bloom"], bloom_boost.inputs[0])
    links.new(gi.outputs["Self Illumination"], bloom_boost.inputs[2])
    illum = _ng_new_node(ng, 'ShaderNodeMath', 380, 300,
                         operation='MULTIPLY')
    links.new(gi.outputs["Self Illumination"], illum.inputs[0])
    links.new(bloom_boost.outputs[0], illum.inputs[1])
    # strength = SI * (1 + 4*bloom*SI) ~ SI when no bloom; boosted with bloom
    links.new(illum.outputs[0], bsdf.inputs["Emission Strength"])

    # --- Alpha: masked uses diffuse alpha; blend uses alpha*opacity ---
    a_mul = _ng_new_node(ng, 'ShaderNodeMath', 230, 480,
                         operation='MULTIPLY')
    links.new(gi.outputs["Diffuse Alpha"], a_mul.inputs[0])
    links.new(gi.outputs["Opacity"], a_mul.inputs[1])
    # GSMA alpha channel also carries opacity — multiply in when present
    ga_mix = _ng_new_node(ng, 'ShaderNodeMix', 230, 620, data_type='FLOAT')
    ga_mix.inputs["A"].default_value = 1.0
    links.new(gi.outputs["Has GSMA"], ga_mix.inputs["Factor"])
    links.new(gi.outputs["GSMA Alpha"], ga_mix.inputs["B"])
    a_mul2 = _ng_new_node(ng, 'ShaderNodeMath', 400, 520,
                          operation='MULTIPLY')
    links.new(a_mul.outputs[0], a_mul2.inputs[0])
    links.new(ga_mix.outputs["Result"], a_mul2.inputs[1])
    # Alpha Mode: 0 = opaque (1.0), 1 = masked (hard step at Clip
    # Threshold — makes foliage read thick and dense), 2 = blended (raw
    # alpha). Built as two mixes: first pick step vs raw by (mode-1),
    # then pick opaque vs that by min(mode, 1).
    stepped = _ng_new_node(ng, 'ShaderNodeMath', 400, 700,
                           operation='GREATER_THAN')
    links.new(a_mul2.outputs[0], stepped.inputs[0])
    links.new(gi.outputs["Clip Threshold"], stepped.inputs[1])
    blend_fac = _ng_new_node(ng, 'ShaderNodeMath', 400, 840,
                             operation='SUBTRACT', use_clamp=True)
    links.new(gi.outputs["Alpha Mode"], blend_fac.inputs[0])
    blend_fac.inputs[1].default_value = 1.0
    a_sel = _ng_new_node(ng, 'ShaderNodeMix', 540, 700, data_type='FLOAT')
    links.new(blend_fac.outputs[0], a_sel.inputs["Factor"])
    links.new(stepped.outputs[0], a_sel.inputs["A"])
    links.new(a_mul2.outputs[0], a_sel.inputs["B"])
    use_a = _ng_new_node(ng, 'ShaderNodeMath', 400, 980,
                         operation='MINIMUM')
    use_a.inputs[1].default_value = 1.0
    links.new(gi.outputs["Alpha Mode"], use_a.inputs[0])
    a_final = _ng_new_node(ng, 'ShaderNodeMix', 680, 700, data_type='FLOAT')
    a_final.inputs["A"].default_value = 1.0
    links.new(use_a.outputs[0], a_final.inputs["Factor"])
    links.new(a_sel.outputs["Result"], a_final.inputs["B"])
    # --- Fresnel: a transparent surface seen edge-on reflects almost
    # everything and lets almost nothing through, which is what stops
    # imported glass reading as a flat grey film. Built as the Schlick
    # form the engine's fresnel_settings vector describes,
    #     fresnel = bias + scale * facing ** power
    # rather than a Fresnel node, so the material_config's own numbers
    # drive it directly. Layer Weight's Facing output is ~0 head-on and
    # ~1 at grazing incidence, which is exactly the (1 - N·V) term.
    # Fresnel Strength is the master switch and defaults to 0.0, so this
    # whole branch stays inert for everything that isn't glass.
    lwf = _ng_new_node(ng, 'ShaderNodeLayerWeight', 400, 1120)
    lwf.inputs["Blend"].default_value = 0.5
    f_pow = _ng_new_node(ng, 'ShaderNodeMath', 560, 1120, operation='POWER')
    links.new(lwf.outputs["Facing"], f_pow.inputs[0])
    links.new(gi.outputs["Fresnel Power"], f_pow.inputs[1])
    f_scale = _ng_new_node(ng, 'ShaderNodeMath', 700, 1120,
                           operation='MULTIPLY')
    links.new(f_pow.outputs[0], f_scale.inputs[0])
    links.new(gi.outputs["Fresnel Scale"], f_scale.inputs[1])
    f_bias = _ng_new_node(ng, 'ShaderNodeMath', 840, 1120,
                          operation='ADD', use_clamp=True)
    links.new(f_scale.outputs[0], f_bias.inputs[0])
    links.new(gi.outputs["Fresnel Bias"], f_bias.inputs[1])
    fres_amt = _ng_new_node(ng, 'ShaderNodeMath', 980, 1120,
                            operation='MULTIPLY', use_clamp=True)
    links.new(f_bias.outputs[0], fres_amt.inputs[0])
    links.new(gi.outputs["Fresnel Strength"], fres_amt.inputs[1])

    # Gate the whole thing by the surface's OWN alpha before it is allowed
    # to push anything towards opaque. Without this, a bias of 0.6 turned a
    # fully transparent pixel into 0.4*0 + 0.6 = 0.6 opaque, so the cut-out
    # areas of decals and the missing pane of a broken window rendered as
    # their black diffuse. Multiplying by the base alpha keeps zero at
    # zero, leaves opaque pixels alone, and still lets genuinely
    # translucent glass firm up towards grazing angles — proportionally to
    # how much surface is actually there.
    fres_gate = _ng_new_node(ng, 'ShaderNodeMath', 1120, 1120,
                             operation='MULTIPLY', use_clamp=True)
    links.new(fres_amt.outputs[0], fres_gate.inputs[0])
    links.new(a_final.outputs["Result"], fres_gate.inputs[1])

    a_fres = _ng_new_node(ng, 'ShaderNodeMix', 820, 700, data_type='FLOAT')
    a_fres.inputs["B"].default_value = 1.0
    links.new(fres_gate.outputs[0], a_fres.inputs["Factor"])
    links.new(a_final.outputs["Result"], a_fres.inputs["A"])

    # --- Alpha Direct: hand the raw Opacity input straight to the BSDF,
    # skipping the diffuse-alpha multiply, the GSMA alpha multiply, the
    # alpha-mode step and the fresnel above. Used for plain "generic"
    # templates that carry an opacity texture and simply want it applied
    # as-is rather than run through the glass machinery.
    a_direct = _ng_new_node(ng, 'ShaderNodeMix', 960, 700, data_type='FLOAT')
    links.new(gi.outputs["Alpha Direct"], a_direct.inputs["Factor"])
    links.new(a_fres.outputs["Result"], a_direct.inputs["A"])
    links.new(gi.outputs["Opacity"], a_direct.inputs["B"])
    links.new(a_direct.outputs["Result"], bsdf.inputs["Alpha"])

    # --- Cubemap reflection: a proper view-dependent reflection layered on
    # top of the surface (NOT metallic — Metallic is a separate user input).
    # Strength = Cube Reflection x GSMA blue mask x fresnel, and it gets
    # weaker as the surface gets rougher.
    fres = _ng_new_node(ng, 'ShaderNodeFresnel', 380, 640)
    fres.inputs["IOR"].default_value = 1.45
    links.new(nmap.outputs["Normal"], fres.inputs["Normal"])
    gloss_fade = _ng_new_node(ng, 'ShaderNodeMath', 380, 500,
                              operation='SUBTRACT', use_clamp=True)
    gloss_fade.inputs[0].default_value = 1.0
    links.new(rough_boost.outputs[0], gloss_fade.inputs[1])
    refl_a = _ng_new_node(ng, 'ShaderNodeMath', 540, 560,
                          operation='MULTIPLY')
    links.new(fres.outputs["Fac"], refl_a.inputs[0])
    links.new(cube_amt.outputs[0], refl_a.inputs[1])
    refl_f = _ng_new_node(ng, 'ShaderNodeMath', 680, 500,
                          operation='MULTIPLY', use_clamp=True)
    links.new(refl_a.outputs[0], refl_f.inputs[0])
    links.new(gloss_fade.outputs[0], refl_f.inputs[1])
    refl_sh = _ng_new_node(ng, 'ShaderNodeEmission', 680, 340)
    links.new(gi.outputs["Cubemap"], refl_sh.inputs["Color"])
    mix_sh = _ng_new_node(ng, 'ShaderNodeMixShader', 820, 200)
    links.new(refl_f.outputs[0], mix_sh.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
    links.new(refl_sh.outputs["Emission"], mix_sh.inputs[2])
    links.new(mix_sh.outputs["Shader"], go.inputs["BSDF"])
    return ng


# Flags that are deliberately ignored when rebuilding materials
IGNORED_FLAG_PREFIXES = ("RL_", "SKINNED_", "CONTOUR", "DEPTH_SCALING",
                         "VERTEX_COLOR")


def _template_flags(render_template):
    return set(f for f in render_template.split(":") if f)


def make_color_mix(nt, blend_type='MIX', location=(0, 0), label=""):
    """Create a colour Mix node and return (node, factor, A, B, result).

    IMPORTANT: ShaderNodeMix has THREE sockets called "A" (Float, Vector,
    Colour) and three called "Result". Looking them up by name returns the
    FLOAT ones, which silently collapses colour data to one channel and
    renders everything dark. Sockets are therefore resolved by TYPE here.
    Falls back to the legacy MixRGB node on older Blender builds."""
    try:
        n = nt.nodes.new('ShaderNodeMix')
        n.data_type = 'RGBA'
        n.blend_type = blend_type
        n.clamp_factor = True
        col_in = [s for s in n.inputs if s.type == 'RGBA']
        col_out = [s for s in n.outputs if s.type == 'RGBA']
        fac = n.inputs[0]
        a_sock, b_sock = col_in[0], col_in[1]
        res = col_out[0]
    except (RuntimeError, TypeError, IndexError):
        n = nt.nodes.new('ShaderNodeMixRGB')
        n.blend_type = blend_type
        fac = n.inputs["Fac"]
        a_sock, b_sock = n.inputs["Color1"], n.inputs["Color2"]
        res = n.outputs["Color"]
    n.location = location
    if label:
        n.label = label
    return n, fac, a_sock, b_sock, res


def rebuild_pd2_material(mat, mat_info, assets_dir):
    """Rebuild one Blender material from its material_config entry using
    the shared PD2 Shader node group."""
    tpl = mat_info.get("render_template", "") or ""
    flags = _template_flags(tpl)
    textures = mat_info.get("textures", {})
    base = tpl.split(":", 1)[0] if tpl else ""

    variables = mat_info.get("variables", {})
    is_blend = ("BLEND_DIFFUSE" in flags
                and textures.get("diffuse_layer0_texture"))
    diffuse = (textures.get("diffuse_texture")
               or textures.get("diffuse0_texture")
               or (None if is_blend else
                   textures.get("diffuse_layer0_texture")))
    gsma = textures.get("material_texture")
    normal = textures.get("bump_normal_texture")
    opacity = textures.get("opacity_texture")
    self_illum = textures.get("self_illumination_texture")
    # Terrain / surface blending layers
    blend_diffuse2 = textures.get("diffuse_layer0_texture") if is_blend else None
    blend_mask = (textures.get("diffuse_layer1_texture")
                  if "BLEND_MASK_SEPERATE" in flags else None)
    blend_gsma2 = (textures.get("diffuse_layer2_texture")
                   if "BLEND_GSMA" in flags else None)
    blend_normal2 = (textures.get("normal_layer0_texture")
                     if "BLEND_NORMAL" in flags else None)
    is_mul_effect = (base == "effect" and "BLEND_MUL" in flags)
    is_additive_effect = (base == "effect" and not is_mul_effect)

    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (700, 0)
    grp = nt.nodes.new('ShaderNodeGroup')
    grp.node_tree = get_pd2_node_group()
    grp.location = (400, 0)
    grp.width = 240
    nt.links.new(grp.outputs["BSDF"], out.inputs["Surface"])

    y = 400

    def add_tex(img, label, color=True):
        nonlocal y
        node = nt.nodes.new('ShaderNodeTexImage')
        node.location = (0, y)
        node.label = label
        node.image = img
        y -= 300
        return node

    has_alpha_source = False

    # ---- Blending (terrain / layered surfaces) ----
    # Factor : the mask texture when the template declares one, otherwise
    #          the blend_control value from the material_config.
    # Mix A  : the *_layer0 texture.  Mix B : the base texture.
    blend_fac_out = None
    if is_blend:
        # blend_control: X = blending smoothness, Y = blend mask bias
        # (Z unused). Both paths run the factor through a SMOOTHSTEP Map
        # Range built from them, which is what gives the soft, gradual
        # transition the game has instead of a linear crossfade.
        smoothness, bias = 0.5, 0.5
        try:
            parts = [float(p) for p in
                     variables.get("blend_control", "").split()]
            if len(parts) >= 1:
                smoothness = max(parts[0], 0.001)
            if len(parts) >= 2:
                bias = parts[1]
        except (ValueError, AttributeError):
            pass

        raw_fac = None
        if blend_mask:
            mimg = _load_texture(blend_mask, assets_dir, is_color=True)
            if mimg:
                mn = nt.nodes.new('ShaderNodeTexImage')
                mn.location = (-940, 420)
                mn.label = "Blend Mask"
                mn.image = mimg
                # Cubic sampling: masks are often low-res, and linear
                # sampling leaves visible pixel steps along the seam.
                try:
                    mn.interpolation = 'Cubic'
                except (AttributeError, TypeError):
                    pass
                mbw = nt.nodes.new('ShaderNodeRGBToBW')
                mbw.location = (-700, 420)
                nt.links.new(mn.outputs["Color"], mbw.inputs["Color"])
                raw_fac = mbw.outputs["Val"]
        if raw_fac is None:
            val = nt.nodes.new('ShaderNodeValue')
            val.location = (-760, 620)
            val.label = "Blend Amount"
            val.outputs[0].default_value = min(max(bias, 0.0), 1.0)
            raw_fac = val.outputs[0]

        mr = nt.nodes.new('ShaderNodeMapRange')
        mr.location = (-520, 560)
        mr.label = f"Blend smoothness {smoothness:g} / bias {bias:g}"
        mr.interpolation_type = 'SMOOTHSTEP'
        mr.inputs["From Min"].default_value = max(bias - smoothness, 0.0)
        mr.inputs["From Max"].default_value = min(bias + smoothness, 1.0)
        nt.links.new(raw_fac, mr.inputs["Value"])
        blend_fac_out = mr.outputs["Result"]
        vlog(f"  BLEND '{mat.name}': factor="
            f"{'mask texture' if blend_mask else 'blend_control value'} | "
            f"A={os.path.basename(blend_diffuse2 or '-')} "
            f"B={os.path.basename(diffuse or '-')}")

    def blended(sock_name, base_path, layer_path, is_color, label):
        """Mix Color: A = layer0 texture, B = base texture, Factor = the
        blend factor. Returns (image node, output socket)."""
        n_base = n_layer = None
        img_base = (_load_texture(base_path, assets_dir, is_color)
                    if base_path else None)
        img_layer = (_load_texture(layer_path, assets_dir, is_color)
                     if layer_path and blend_fac_out is not None else None)
        if img_base:
            n_base = add_tex(img_base, label)
        if img_layer:
            n_layer = add_tex(img_layer, label + " (layer 0)")

        out = None
        if n_base and n_layer:
            _mx, m_fac, m_a, m_b, m_res = make_color_mix(
                nt, 'MIX', (240, n_base.location.y), label + " blend")
            nt.links.new(blend_fac_out, m_fac)
            nt.links.new(n_layer.outputs["Color"], m_a)   # A = layer0
            nt.links.new(n_base.outputs["Color"], m_b)    # B = base
            out = m_res
        elif n_base:
            out = n_base.outputs["Color"]
        elif n_layer:
            out = n_layer.outputs["Color"]
        if out is not None:
            nt.links.new(out, grp.inputs[sock_name])
        return (n_base or n_layer), out

    if diffuse:
        n, dif_out = blended("Diffuse", diffuse, blend_diffuse2, True,
                             "Diffuse")
        # VERTEX_COLOR in the render template: multiply the blend result by
        # the mesh's colour attribute and feed that to the shader.
        if "VERTEX_COLOR" in flags and dif_out is not None:
            vc = nt.nodes.new('ShaderNodeVertexColor')
            vc.location = (240, 980)
            vc.label = "Vertex Color"
            _vm, v_fac, v_a, v_b, v_res = make_color_mix(
                nt, 'MULTIPLY', (460, 900), "x Vertex Color")
            v_fac.default_value = 1.0
            nt.links.new(dif_out, v_a)
            nt.links.new(vc.outputs["Color"], v_b)
            nt.links.new(v_res, grp.inputs["Diffuse"])
        if n:
            if is_additive_effect:
                # Additive effect: diffuse brightness doubles as opacity
                bw = nt.nodes.new('ShaderNodeRGBToBW')
                bw.location = (220, n.location.y + 120)
                nt.links.new(n.outputs["Color"], bw.inputs["Color"])
                nt.links.new(bw.outputs["Val"], grp.inputs["Opacity"])
            elif is_mul_effect:
                # Multiplicative decal (oil stains, shoe marks...): the
                # texture's WHITE areas are the empty part of the decal, so
                # opacity is the inverted brightness. The colour stays the
                # texture itself, so the stain keeps its own detail rather
                # than reading as a flat block.
                bw = nt.nodes.new('ShaderNodeRGBToBW')
                bw.location = (220, n.location.y + 160)
                nt.links.new(n.outputs["Color"], bw.inputs["Color"])
                inv = nt.nodes.new('ShaderNodeMath')
                inv.operation = 'SUBTRACT'
                inv.inputs[0].default_value = 1.0
                inv.location = (220, n.location.y + 60)
                inv.label = "Inverted alpha (white = empty)"
                nt.links.new(bw.outputs["Val"], inv.inputs[1])
                # 'intensity' from the material_config scales how strongly
                # the decal shows ("identity" = 1.0).
                inten = variables.get("intensity", "")
                try:
                    power = float(inten)
                except (TypeError, ValueError):
                    power = 1.0
                pw = nt.nodes.new('ShaderNodeMath')
                pw.operation = 'MULTIPLY'
                pw.use_clamp = True
                pw.location = (400, n.location.y + 60)
                pw.label = "Opacity power (intensity)"
                pw.inputs[1].default_value = max(power, 0.0)
                nt.links.new(inv.outputs[0], pw.inputs[0])
                nt.links.new(pw.outputs[0], grp.inputs["Opacity"])
                grp.inputs["Specular Strength"].default_value = 0.0
            else:
                nt.links.new(n.outputs["Alpha"], grp.inputs["Diffuse Alpha"])
            has_alpha_source = True

    has_opacity_tex = False
    if opacity:
        img = _load_texture(opacity, assets_dir, is_color=False)
        if img:
            n = add_tex(img, "Opacity")
            # Straight into the group's Opacity input, which feeds the
            # Principled BSDF's Alpha — no Separate Color node in between.
            # Blender converts the Color output to a float on the way in.
            #
            # PD2 opacity textures are documented as black/white masks, so
            # all three channels normally carry the same value and the
            # implicit conversion is a no-op. Where they differ, the Diesel
            # channel layout is: R = fresnel strength, G = opacity,
            # B = cubemap reflection strength. Splitting off RED — which is
            # what this used to do — read the fresnel channel, not the
            # opacity one. Feeding the whole colour in is closer to correct
            # than that was; to follow the channel table strictly, put a
            # ShaderNodeSeparateColor back in and link its Green output.
            nt.links.new(n.outputs["Color"], grp.inputs["Opacity"])
            has_alpha_source = True
            has_opacity_tex = True

    has_gsma = False
    if gsma or blend_gsma2:
        # The render_template_database declares material_texture as
        # expects_gamma_corrected="true", so GSMA is sRGB data — reading it
        # as Non-Color skews gloss/spec and makes everything look flat.
        n, _ = blended("GSMA Color", gsma, blend_gsma2, True,
                       "GSMA (gloss/spec/cubemask/alpha)")
        if n:
            nt.links.new(n.outputs["Alpha"], grp.inputs["GSMA Alpha"])
            grp.inputs["Has GSMA"].default_value = 1.0
            has_gsma = True

    if normal or blend_normal2:
        n, _ = blended("Normal Color", normal, blend_normal2, False,
                       "Normal (OpenGL)")
        if n:
            nt.links.new(n.outputs["Alpha"], grp.inputs["Normal Alpha"])
            grp.inputs["Has Normal"].default_value = 1.0
            swap_xy, x_from_alpha = _detect_normal_mode(n.image)
            grp.inputs["Normal Swap XY"].default_value = swap_xy
            grp.inputs["Normal X From Alpha"].default_value = x_from_alpha

    # ---- Scalar / colour variables from the material_config ----
    def _var_floats(name):
        try:
            return [float(p) for p in variables.get(name, "").split()]
        except (ValueError, AttributeError):
            return []

    # glossiness_control (GLOSS_CONTROL_VALUE templates): the artist-set
    # gloss for materials with no GSMA texture.
    gv = _var_floats("glossiness_control")
    if gv:
        grp.inputs["Base Roughness"].default_value = min(
            max(1.0 - gv[0], 0.0), 1.0)

    # tint_color: declared as "Diffuse Tint Color (x2)", so it is doubled.
    tc = _var_floats("tint_color")
    if len(tc) >= 3 and "SIMPLE_TINT" in flags:
        dif_link = next((l for l in nt.links
                         if l.to_socket is grp.inputs["Diffuse"]), None)
        _tm, t_fac, t_a, t_b, t_res = make_color_mix(
            nt, 'MULTIPLY', (240, 1160), "Diffuse Tint (x2)")
        t_fac.default_value = 1.0
        t_b.default_value = (min(tc[0] * 2.0, 1.0), min(tc[1] * 2.0, 1.0),
                             min(tc[2] * 2.0, 1.0), 1.0)
        if dif_link is not None:
            nt.links.new(dif_link.from_socket, t_a)
            nt.links.remove(dif_link)
        else:
            t_a.default_value = (1.0, 1.0, 1.0, 1.0)
        nt.links.new(t_res, grp.inputs["Diffuse"])

    # Cubemap reflections: enabled by the template; blended with GSMA blue
    # inside the group. (The actual cube texture is skipped — Blender's
    # world lighting stands in for the baked cubemap.)
    if "CUBE_ENVIRONMENT_MAPPING" in flags or "GLOBAL_ENVIRONMENT_MAPPING" in flags:
        grp.inputs["Cube Reflection"].default_value = 0.5
        cube_path = textures.get("reflection_texture")
        if cube_path and not _is_cubemap_strip(cube_path, assets_dir):
            cimg = _load_texture(cube_path, assets_dir, is_color=True)
            if cimg:
                # NB: named tex_coord, not tc — `tc` is the tint_color list
                # a few blocks up and was being shadowed by a node here.
                tex_coord = nt.nodes.new('ShaderNodeTexCoord')
                tex_coord.location = (-500, 1050)
                env = nt.nodes.new('ShaderNodeTexEnvironment')
                env.location = (-300, 1050)
                env.label = "Reflection Cubemap"
                env.image = cimg
                nt.links.new(tex_coord.outputs["Reflection"],
                             env.inputs["Vector"])
                nt.links.new(env.outputs["Color"], grp.inputs["Cubemap"])

    # Additive effects glow: emissive + bloom, alpha-blended
    if is_additive_effect:
        grp.inputs["Self Illumination"].default_value = 1.0
        grp.inputs["Illum Bloom"].default_value = 1.0

    # Self illumination
    if "SELF_ILLUMINATION" in flags or self_illum:
        ilm = _var_floats("il_multiplier")
        grp.inputs["Self Illumination"].default_value = (
            min(max(ilm[0], 0.0), 100.0) if ilm else 1.0)
        ilt = _var_floats("il_tint")
        if len(ilt) >= 3:
            grp.inputs["Illum Tint"].default_value = (
                ilt[0], ilt[1], ilt[2], 1.0)
        if "SELF_ILLUMINATION_BLOOM" in flags:
            grp.inputs["Illum Bloom"].default_value = 1.0
        if self_illum:
            img = _load_texture(self_illum, assets_dir, is_color=True)
            if img:
                n = add_tex(img, "Self Illumination")
                # Use the illum texture as the emission source by mixing it
                # into Diffuse would tint everything; leave diffuse driving
                # emission (group design) unless there's no diffuse at all.
                if not diffuse:
                    nt.links.new(n.outputs["Color"], grp.inputs["Diffuse"])

    # Alpha / blend mode decision
    def _set_render_method(m, method):
        # Blender 4.2+ (EEVEE Next): 'DITHERED' or 'BLENDED'
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = method

    def _set_blend_method(m, method):
        # Material.blend_method was removed in Blender 4.3 (EEVEE Next uses
        # surface_render_method instead). Assigning it unguarded raised
        # AttributeError, which the caller swallowed as "failed to rebuild
        # material" — every material silently stayed untextured on 4.3+.
        if hasattr(m, "blend_method"):
            m.blend_method = method

    alpha_mode = 0.0
    if "ALPHA_MASKED" in flags:
        # Masked foliage etc.: hard clip inside the group at a LOW
        # threshold so leaves read thick/dense, rendered dithered so
        # sorting never breaks on dense vegetation.
        alpha_mode = 1.0
        # The material_config's own alpha_ref is the game's cutoff (some
        # foliage/decals use values as low as 0.01); fall back to 0.15.
        ar = _var_floats("alpha_ref")
        clip = min(max(ar[0], 0.0), 1.0) if ar else 0.15
        grp.inputs["Clip Threshold"].default_value = clip
        _set_blend_method(mat, 'CLIP')
        if hasattr(mat, "alpha_threshold"):
            mat.alpha_threshold = clip
        _set_render_method(mat, 'DITHERED')
    elif is_additive_effect or is_mul_effect:
        alpha_mode = 2.0
        _set_blend_method(mat, 'BLEND')
        _set_render_method(mat, 'BLENDED')
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = 'NONE'
    elif base == "opacity":
        # Glass and other translucent surfaces MUST be see-through even
        # when no texture supplies an alpha channel.
        alpha_mode = 2.0
        _set_blend_method(mat, 'BLEND')
        _set_render_method(mat, 'BLENDED')
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = 'NONE'
        if not has_alpha_source and not has_gsma:
            grp.inputs["Opacity"].default_value = 0.45
        elif has_alpha_source and not has_opacity_tex:
            # Diffuse alpha is often fully white on glass — cap via the
            # Opacity input so it still reads transparent. Skipped when a
            # real opacity texture is driving that input, since the default
            # would be ignored anyway and the cap is not what the texture
            # asked for.
            grp.inputs["Opacity"].default_value = 0.6
        # Removed in 4.2+; the old line was also a no-op self-assignment
        # that raised AttributeError on builds where the property is gone.
        if hasattr(mat, "use_screen_refraction"):
            mat.use_screen_refraction = True
    elif (base == "decal" or "OPACITY_TEXTURE" in flags
          or "GSMA_ALPHA_MASKING" in flags):
        if has_alpha_source or has_gsma:
            alpha_mode = 2.0
            _set_blend_method(mat, 'BLEND')
            _set_render_method(mat, 'BLENDED')
    # --- Fresnel, driven by the config's own numbers ---------------------
    # Applies to any material that declares fresnel_settings, not just the
    # opacity template — and to glass generally, which is what needs it.
    fs = _var_floats("fresnel_settings")
    # Cut-out masks are geometry, not glass: a decal's transparent surround
    # is "no surface here", and a fresnel lift on it is meaningless even
    # when the config happens to declare fresnel_settings.
    cutout_like = base in ("decal", "flesh") or "ALPHA_MASKING" in flags
    if len(fs) >= 3 and not cutout_like:
        parts = dict(zip(FRESNEL_SETTINGS_ORDER, fs[:3]))
        grp.inputs["Fresnel Power"].default_value = min(
            max(parts["power"], 0.05), 16.0)
        grp.inputs["Fresnel Scale"].default_value = min(
            max(parts["scale"], 0.0), 10.0)
        grp.inputs["Fresnel Bias"].default_value = min(
            max(parts["bias"], 0.0), 1.0)
        grp.inputs["Fresnel Strength"].default_value = GLASS_FRESNEL
        vlog(f"  fresnel '{mat.name}': power={parts['power']:g} "
             f"scale={parts['scale']:g} bias={parts['bias']:g}")
    elif base == "opacity" and not cutout_like:
        grp.inputs["Fresnel Strength"].default_value = GLASS_FRESNEL

    # --- Plain "generic" template + opacity texture: no fancy treatment.
    # Feed the opacity straight to the BSDF's Alpha and switch off the
    # glass machinery entirely.
    if base == "generic" and has_opacity_tex:
        grp.inputs["Alpha Direct"].default_value = 1.0
        grp.inputs["Fresnel Strength"].default_value = 0.0
        alpha_mode = 2.0
        _set_blend_method(mat, 'BLEND')
        _set_render_method(mat, 'BLENDED')
        vlog(f"  generic+opacity '{mat.name}': opacity wired straight "
             f"to Alpha")

    if has_opacity_tex and alpha_mode == 0.0:
        # A dedicated opacity texture was wired up, but nothing above put
        # the material into a transparent mode — Alpha Mode 0 forces alpha
        # to 1.0 inside the group, so the texture would have had no visible
        # effect at all.
        alpha_mode = 2.0
        _set_blend_method(mat, 'BLEND')
        _set_render_method(mat, 'BLENDED')
    grp.inputs["Alpha Mode"].default_value = alpha_mode

    # No GSMA, opaque material -> diffuse alpha likely stores specular
    if not has_gsma and alpha_mode == 0.0 and diffuse:
        grp.inputs["Spec From Diffuse Alpha"].default_value = 1.0

    if "DOUBLE_SIDED" in flags:
        mat.use_backface_culling = False
    else:
        mat.use_backface_culling = True

    mat["pd2_render_template"] = tpl
    mat["pd2_textured"] = True
    mat["pd2_tex_sig"] = material_signature(mat_info)


def material_signature(mat_info):
    """Stable id for a material_config entry: its template plus the exact
    set of textures. Two slots that share a NAME but differ here are
    genuinely different materials and must not share a datablock."""
    tex = sorted((k, v) for k, v in (mat_info.get("textures") or {}).items())
    payload = (mat_info.get("render_template", "") + "|"
               + "|".join(f"{k}={v}" for k, v in tex))
    return f"{diesel_hash(payload):016x}"


def apply_pd2_materials(objects, unit_path, assets_dir):
    """Entry point: for every material slot on the imported model's meshes,
    find its exact-name entry in the unit's material_config and rebuild it.
    Slot names must match config names exactly (case-insensitive,
    .001-suffixes stripped) — anti model-mixing is enforced by only ever
    touching slots that already exist on THIS model."""
    mc_path = find_material_config_for_unit(unit_path, assets_dir)
    if not mc_path:
        vlog(f"  no material_config resolved for {unit_path}")
        return 0
    mats_cfg = parse_material_config(mc_path)
    if not mats_cfg:
        return 0

    # name of the original material -> replacement material, or None when
    # the material was rebuilt in place / left alone. The old code kept a
    # plain "already seen" set and skipped repeat slots entirely, so when a
    # name clash forced a variant datablock only the first slot referencing
    # that material was remapped and every other slot silently kept the
    # wrong texture set.
    resolved = {}
    n = 0
    for o in objects:
        if o.type != 'MESH' or o.data is None:
            continue
        for slot in o.material_slots:
            mat = slot.material
            if mat is None:
                continue
            if mat.name in resolved:
                repl = resolved[mat.name]
                if repl is not None and slot.material is not repl:
                    slot.material = repl
                continue
            resolved[mat.name] = None
            original_name = mat.name
            m = _dup_suffix_re.match(mat.name)
            base_name = (m.group(1) if m else mat.name).lower()
            info = mats_cfg.get(base_name)
            if info is None:
                vlog(f"  material '{mat.name}' not in "
                     f"{os.path.basename(mc_path)} — left untouched")
                continue
            sig = material_signature(info)
            existing_sig = mat.get("pd2_tex_sig")
            if existing_sig == sig:
                continue          # already built from exactly this entry
            if existing_sig:
                # Same material NAME, different textures: two models reuse
                # a generic name like 'mat_wood'. Give this one its own
                # datablock so the texture sets can't overwrite each other.
                variant_name = f"{m.group(1) if m else mat.name}#{sig[:8]}"
                variant = bpy.data.materials.get(variant_name)
                if variant is None:
                    variant = mat.copy()
                    variant.name = variant_name
                    try:
                        rebuild_pd2_material(variant, info, assets_dir)
                        n += 1
                    except Exception as e:
                        log_error(f"  failed to rebuild {variant_name}: {e}")
                        traceback.print_exc()
                    log(f"  material name clash: '{mat.name}' has different "
                        f"textures here -> created '{variant_name}'")
                resolved[original_name] = variant
                slot.material = variant
                continue
            try:
                rebuild_pd2_material(mat, info, assets_dir)
                n += 1
            except Exception as e:
                log_error(f"  failed to rebuild material {mat.name}: {e}")
                traceback.print_exc()
    if n:
        vlog(f"  -> rebuilt {n} material(s) from {os.path.basename(mc_path)}")
    return n


# ----------------------------------------------------------------------------
# Material de-duplication
# ----------------------------------------------------------------------------

def merge_duplicate_materials():
    """Remap every material named 'X.001'..'X.999' onto 'X' and purge the dupes."""
    merged = 0
    removed = []
    for mat in list(bpy.data.materials):
        m = _dup_suffix_re.match(mat.name)
        if not m:
            continue
        base_name = m.group(1)
        base = bpy.data.materials.get(base_name)
        if (base and base is not mat
                and base.get("pd2_tex_sig") == mat.get("pd2_tex_sig")):
            mat.user_remap(base)
            removed.append(mat)
            merged += 1
        elif base is None:
            # True orphan: nothing owns the un-suffixed name, so take it.
            log(f"  Renaming orphan duplicate material {mat.name} -> {base_name}")
            mat.name = base_name
        # else: the base name belongs to a genuinely different material.
        # The old code tried to rename onto it anyway, which Blender simply
        # re-suffixed straight back — a no-op that logged "orphan" for
        # materials that were neither orphans nor renamed.

    for mat in removed:
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    if merged:
        log(f"Merged {merged} duplicate materials")
    return merged


# ----------------------------------------------------------------------------
# JSON parsing
# ----------------------------------------------------------------------------

def load_units_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    units = []
    statics = data.get("statics", {}) if isinstance(data, dict) else {}
    instances = data.get("instances", {}) if isinstance(data, dict) else {}
    if not isinstance(statics, dict):
        statics = {}
    if not isinstance(instances, dict):
        instances = {}
    if not statics and not instances:
        raise ValueError("JSON contains neither a 'statics' nor an 'instances' section")

    inst_list = []
    for inst_id, inst in instances.items():
        if not isinstance(inst, dict):
            continue
        inst_list.append({
            "id": inst_id,
            "folder": inst.get("folder", ""),
            "name": inst.get("name", f"instance_{inst_id}"),
            "position": parse_triplet(inst.get("position", ""), "Vector3"),
            "rotation": parse_triplet(inst.get("rotation", ""), "Rotation"),
            "continent": inst.get("continent", ""),
            "mission_placed": bool(inst.get("mission_placed", False)),
            "script": inst.get("script", ""),
        })

    for static_id, static in statics.items():
        if not isinstance(static, dict):
            continue
        ud = static.get("unit_data", {})
        if not isinstance(ud, dict):
            continue
        path = ud.get("name", "")
        if not isinstance(path, str) or not path:
            continue
        lights = ud.get("lights", {})
        if not isinstance(lights, dict):
            # create_unit_lights iterates .items(); a stray list here used to
            # pass the truthiness check and then crash the whole import.
            log_error(f"  unit {static_id}: 'lights' is "
                      f"{type(lights).__name__}, expected object — ignored")
            lights = {}
        name_id = ud.get("name_id")
        if not isinstance(name_id, str) or not name_id:
            name_id = f"unit_{static_id}"
        units.append({
            "path": path,
            "name_id": name_id,
            "unit_id": ud.get("unit_id", 0),
            "position": parse_triplet(ud.get("position", ""), "Vector3"),
            "rotation": parse_triplet(ud.get("rotation", ""), "Rotation"),
            "lights": lights,
        })
    return units, inst_list


# ----------------------------------------------------------------------------
# Preferences
# ----------------------------------------------------------------------------

class PD2LevelImporterPreferences(AddonPreferences):
    bl_idname = __name__

    assets_directory: StringProperty(
        name="Assets Directory",
        description="Root folder containing the extracted game assets (must contain the 'units' folder)",
        subtype='DIR_PATH',
        default="",
    )

    parser_exe: StringProperty(
        name="PD2ModelParser.exe (optional)",
        description=("Manual path to PD2ModelParser.exe. Leave empty to "
                     "auto-detect it from the Diesel Model Tool Wrapper "
                     "addon's preferences"),
        subtype='FILE_PATH',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="PAYDAY 2 Extracted Assets Location:")
        layout.prop(self, "assets_directory")
        box = layout.box()
        # bpy.path.abspath resolves Blender's '//' relative paths. Without it
        # this panel reported "Directory does not exist" for a perfectly
        # valid relative path that the importer itself accepted.
        assets_dir = (bpy.path.abspath(self.assets_directory)
                      if self.assets_directory else "")
        if not self.assets_directory:
            box.label(text="Please set the assets directory", icon='ERROR')
        elif not os.path.isdir(assets_dir):
            box.label(text="Directory does not exist", icon='ERROR')
        elif not os.path.isdir(os.path.join(assets_dir, "units")):
            box.label(text="'units' folder not found inside this directory", icon='ERROR')
        else:
            box.label(text="Assets directory OK", icon='CHECKMARK')

        layout.separator()
        layout.label(text="Model Conversion:")
        layout.prop(self, "parser_exe")
        layout.label(text="Leave empty to auto-detect from the Diesel Model Tool Wrapper addon")

        layout.separator()
        # draw() runs on every redraw of the prefs panel, and this used to
        # stat every file in the cache each time — noticeably laggy once a
        # few thousand .glb files have accumulated. Sampled at most once
        # every few seconds instead.
        n, size = _cached_cache_stats()
        layout.label(text=f"Conversion cache: {n} models, {size / 1e6:.1f} MB")
        layout.operator(PD2_OT_clear_glb_cache.bl_idname, icon='TRASH')


_cache_stats = (0.0, 0, 0)   # (timestamp, n files, total bytes)
_CACHE_STATS_TTL = 5.0


def _cached_cache_stats(force=False):
    global _cache_stats
    now = time.time()
    if force or now - _cache_stats[0] > _CACHE_STATS_TTL:
        n_files, total = _dir_stats(
            os.path.join(tempfile.gettempdir(), GLB_CACHE_DIRNAME))
        _cache_stats = (now, n_files, total)
    return _cache_stats[1], _cache_stats[2]


class PD2_OT_clear_glb_cache(Operator):
    """Delete every cached .glb conversion (they will be re-converted on
    the next import)"""
    bl_idname = "pd2_importer.clear_glb_cache"
    bl_label = "Clear Conversion Cache"

    def execute(self, context):
        n = 0
        size = 0
        # Also clears the decoded-texture cache (BC4/BC5 PNGs and the .dds
        # copies of .texture files), which the old version left behind.
        for dirname in (GLB_CACHE_DIRNAME, TEX_CACHE_DIRNAME):
            d = os.path.join(tempfile.gettempdir(), dirname)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                p = os.path.join(d, fn)
                try:
                    size += os.path.getsize(p)
                    os.remove(p)
                    n += 1
                except OSError:
                    pass
        reset_module_caches()
        _cached_cache_stats(force=True)
        self.report({'INFO'},
                    f"Cleared {n} cached file(s) ({size / 1e6:.1f} MB)")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# Import operator
# ----------------------------------------------------------------------------

class IMPORT_OT_pd2_level_json(Operator, ImportHelper):
    """Import a PAYDAY 2 level .json and auto-import its .model files"""
    bl_idname = "import_scene.pd2_level_json"
    bl_label = "Import PAYDAY 2 Level (.json)"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    create_empties: BoolProperty(
        name="Empties for Missing Models",
        description="Create an empty placeholder when a .model file cannot be found",
        default=True,
    )

    merge_materials: BoolProperty(
        name="Merge Duplicate Materials",
        description="Combine materials with .001–.999 suffixes into the base material",
        default=True,
    )

    instance_repeats: BoolProperty(
        name="Instance Repeated Models (fast)",
        description=("Import each unique model once, then place repeats as "
                     "linked duplicates sharing mesh data. Massively faster "
                     "and lighter, and avoids .001 material duplicates. "
                     "Disable only if every unit must have fully independent "
                     "mesh data"),
        default=True,
    )

    parallel_workers: IntProperty(
        name="Parallel Conversions",
        description=("How many PD2ModelParser.exe processes to run at once "
                     "during the conversion pass. Default leaves a couple of "
                     "cores free so your PC stays responsive; raise it to "
                     "your full core count for maximum conversion speed at "
                     "the cost of 100% CPU usage"),
        default=max(1, (os.cpu_count() or 4) - 2),
        min=1,
        max=64,
    )

    use_conversion_cache: BoolProperty(
        name="Cache Converted Models",
        description=("Keep every converted .glb in a persistent cache "
                     "(keyed by the .model file's path, size and date, so "
                     "edited models are always re-converted). Re-importing "
                     "a level then skips the conversion pass almost "
                     "entirely. Cache lives in the system temp folder"),
        default=True,
    )

    low_cpu_priority: BoolProperty(
        name="Low CPU Priority Conversions",
        description=("Run the PD2ModelParser processes at below-normal OS "
                     "priority. They still use spare CPU, but Blender and "
                     "your other applications always get the CPU first, so "
                     "the machine stays usable even during heavy conversion"),
        default=True,
    )

    convert_instances: BoolProperty(
        name="Convert Instance .continent to JSON",
        description=("For every entry in the level's 'instances' section, "
                     "find the instance's binary .continent scriptdata file "
                     "in the assets directory, decode it with the built-in "
                     "Diesel scriptdata parser, and write it as JSON into a "
                     "'converted_instances' folder next to the level JSON"),
        default=True,
    )

    import_instances: BoolProperty(
        name="Import Instance Contents",
        description=("Also place every instance's units in the scene: each "
                     "instance gets a root empty at its level-JSON "
                     "position/rotation, with the units from its .continent "
                     "file imported and parented under it"),
        default=True,
    )

    import_massunits: BoolProperty(
        name="Import Massunits (scatter)",
        description=("Also import the level's massunit.massunit file (one "
                     "folder above the level JSON): the mass-placed scatter "
                     "props — foliage, rocks, small clutter — that aren't "
                     "listed in the level's statics. Silently skipped when "
                     "no massunit file exists"),
        default=True,
    )

    fast_import: BoolProperty(
        name="Fast Import (defer viewport updates)",
        description=("Hide the level collection from the view layer while "
                     "importing so Blender doesn't re-evaluate every "
                     "already-placed object on each model import. Prevents "
                     "the import from getting progressively slower as the "
                     "scene fills up. The level appears all at once at the "
                     "end. Also disables global undo during the import"),
        default=True,
    )

    verbose_log: BoolProperty(
        name="Verbose Console Log",
        description=("Print a console line for every placed unit and light. "
                     "Disable for large levels — printing thousands of lines "
                     "to the console is itself a noticeable slowdown "
                     "(progress is still logged every 100 units)"),
        default=False,
    )

    light_cone_mode: EnumProperty(
        name="Light Cone Meshes",
        description=("PAYDAY 2 fakes light shafts with cone geometry that "
                     "the engine draws additively. Imported normally they "
                     "show up as solid grey cones"),
        items=[
            ('FAKE', "Fake volumetric",
             "Additive surface that fades towards its silhouette to mimic "
             "looking through fog. Uses the texture's real UVs and costs "
             "almost nothing to render"),
            ('VOLUME', "True volumetric",
             "Shade the cone mesh as actual fog. Heavier, and the texture "
             "has to fall back to Generated coordinates"),
            ('HIDE', "Hide",
             "Keep the objects but hide them in the viewport and renders"),
            ('KEEP', "Leave as imported",
             "Do nothing — they will look like solid geometry"),
        ],
        default='FAKE',
    )

    light_cone_strength: FloatProperty(
        name="Light Cone Strength",
        description="Emission strength for light cone geometry",
        default=0.4, min=0.0, soft_max=20.0,
    )

    glass_fresnel: FloatProperty(
        name="Glass Fresnel",
        description=("How strongly glass turns opaque at grazing angles. "
                     "0 disables it and leaves glass evenly transparent. "
                     "The curve itself comes from the material_config's "
                     "fresnel_settings when it states one"),
        default=1.0, min=0.0, max=1.0,
    )

    light_cone_falloff: FloatProperty(
        name="Light Cone Edge Falloff",
        description=("How quickly a fake volumetric cone fades towards its "
                     "silhouette. 1 is a soft even wash, higher values pull "
                     "the glow into the middle of the shaft"),
        default=1.6, min=0.1, soft_max=8.0,
    )

    light_cone_density: FloatProperty(
        name="Light Cone Density",
        description=("Fog density for volumetric light cones. Lower reads "
                     "as a thin haze, higher as a solid beam"),
        default=1.0, min=0.0, soft_max=10.0,
    )

    only_g_meshes: BoolProperty(
        name="Only Import 'g_' Meshes",
        description=("Delete every imported mesh whose name doesn't start with "
                     "'g_' (collision, shadow, destruction meshes etc.), keeping "
                     "only the visible graphics meshes"),
        default=False,
    )

    import_lights: BoolProperty(
        name="Import Lights",
        description=("Create Blender lights from the JSON light source data "
                     "(color, far_range for reach/power, multiplier for "
                     "brightness), parented to each unit"),
        default=True,
    )

    import_textures: BoolProperty(
        name="Import Textures & Materials",
        description=("Follow each unit's .unit -> .object -> "
                     ".material_config chain, load its diffuse/GSMA/normal/"
                     "opacity textures and rebuild the model's materials "
                     "through the shared 'PD2 Shader' node group. Only "
                     "materials whose names exactly match the model's own "
                     "slots are touched (no cross-model mixing)"),
        default=True,
    )

    light_power_scale: FloatProperty(
        name="Light Power Scale",
        description=("Global brightness scale for imported lights. Light energy "
                     "= multiplier x (far_range/100)^2 x this value, in watts"),
        default=10.0,
        min=0.01,
        max=10000.0,
    )

    rotation_order: EnumProperty(
        name="Rotation Order",
        description=("Order the euler axes are applied in (left to right). "
                     "XYZ applies the -90 X upright first and the JSON yaw (Z) "
                     "last about the world vertical — usually correct for levels"),
        items=[
            ('XYZ', "XYZ (upright first, yaw last)", "Recommended: X applied first, Z last"),
            ('XZY', "XZY", "X first, then Z, then Y"),
            ('YXZ', "YXZ", "Y first, then X, then Z"),
            ('YZX', "YZX", "Y first, then Z, then X"),
            ('ZXY', "ZXY (yaw first)", "Z applied first — the old behavior"),
            ('ZYX', "ZYX", "Z first, then Y, then X"),
        ],
        default='XYZ',
    )

    flip_rot_x: BoolProperty(name="Invert Rot X", default=False,
                             description="Negate the JSON's X (2nd) rotation value")
    flip_rot_y: BoolProperty(name="Invert Rot Y", default=False,
                             description="Negate the JSON's Y (3rd) rotation value")
    flip_rot_z: BoolProperty(name="Invert Rot Z (Yaw)", default=False,
                             description="Negate the JSON's Z/yaw (1st) rotation value")

    flip_pos_x: BoolProperty(name="Invert Pos X", default=False,
                             description="Negate the X position")
    flip_pos_y: BoolProperty(name="Invert Pos Y", default=False,
                             description="Negate the Y position")
    flip_pos_z: BoolProperty(name="Invert Pos Z", default=False,
                             description="Negate the Z position")

    def execute(self, context):
        t_start = time.time()
        global VERBOSE, GLASS_FRESNEL
        VERBOSE = self.verbose_log
        GLASS_FRESNEL = self.glass_fresnel

        log(f"PAYDAY 2 Level Importer v"
            f"{'.'.join(str(v) for v in bl_info['version'])}")
        prefs = context.preferences.addons[__name__].preferences
        assets_dir = bpy.path.abspath(prefs.assets_directory) if prefs.assets_directory else ""

        if not assets_dir or not os.path.isdir(assets_dir):
            self.report({'ERROR'},
                        "Assets directory not set or missing. Set it in "
                        "Edit > Preferences > Add-ons > PAYDAY 2 Level Importer.")
            return {'CANCELLED'}

        parser_exe, wrapper_found = find_parser_exe(prefs)
        if parser_exe is None:
            if wrapper_found:
                msg = ("Found the Diesel Model Tool Wrapper addon but could not "
                       "locate PD2ModelParser.exe in its preferences. Set the "
                       "path manually in this addon's preferences.")
            else:
                msg = ("Cannot import .model files without the "
                       "'io_scene_dieselmodeltoolwrapper' (or _master) addon "
                       "installed/enabled, or PD2ModelParser.exe set in this "
                       "addon's preferences.")
            log_error(msg)
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        # --- Load JSON ---
        try:
            units, instances = load_units_from_json(self.filepath)
        except Exception as e:
            log_error(f"Failed to read JSON: {e}")
            traceback.print_exc()
            self.report({'ERROR'}, f"Failed to read JSON: {e}")
            return {'CANCELLED'}

        log(f"Loaded {len(units)} units, {len(instances)} instances "
            f"from {os.path.basename(self.filepath)}")

        # --- Convert instance .continent files to JSON ---
        n_inst_ok = n_inst_missing = 0
        inst_groups = []
        if instances and (self.convert_instances or self.import_instances):
            load_hashlist(parser_exe)
        if self.convert_instances and instances:
            out_dir = os.path.join(os.path.dirname(self.filepath),
                                   "converted_instances")
            log(f"Converting {len(instances)} instance continent files -> "
                f"{out_dir}")
            n_inst_ok, n_inst_missing = convert_instances_to_json(
                instances, assets_dir, out_dir)
        if self.import_instances and instances:
            log(f"Collecting units from {len(instances)} instances...")
            inst_groups = collect_instance_units(instances, assets_dir)
        n_inst_units = sum(len(g["units"]) for g in inst_groups)

        # --- Massunits (mass-placed scatter, one folder above the JSON) ---
        mass_instances = []
        if self.import_massunits:
            mass_path = find_massunit_file(self.filepath)
            if mass_path:
                log(f"Found massunit file: {mass_path}")
                load_hashlist(parser_exe)
                mass_instances = parse_massunit_file(mass_path)
            else:
                log("No massunit file found — skipping scatter import")

        # --- Level collection ---
        level_name = os.path.splitext(os.path.basename(self.filepath))[0]
        level_col = bpy.data.collections.new(level_name)
        context.scene.collection.children.link(level_col)

        # --- Temp dir for converted .glb files ---
        # Created before the try block below only as a name; the mkdtemp and
        # progress_begin calls themselves are inside it, so an exception in
        # the setup between them can no longer leak a temp dir or leave the
        # progress cursor spinning.
        tmp_dir = None
        missing = set()
        n_imported = n_empty = n_failed = n_lights = 0
        t_convert = t_import = 0.0

        # Unique model paths, in first-seen order (level units + instance units)
        all_paths = [u["path"] for u in units]
        for g in inst_groups:
            all_paths.extend(iu["path"] for iu in g["units"])
        all_paths.extend(mi["path"] for mi in mass_instances)
        unique_paths = list(dict.fromkeys(all_paths))
        log(f"{len(units)} units + {n_inst_units} instance units -> "
            f"{len(unique_paths)} unique models")

        wm = context.window_manager

        # Hidden collection holding one imported "prototype" per unique model.
        # Every placed unit is a fast linked-data duplicate of its prototype.
        proto_col = bpy.data.collections.new(level_name + "_prototypes")
        prototypes = {}  # unit path -> list of prototype object names
        # Paths whose model imported fine but contained no g_ mesh at all, so
        # the only-g_ filter emptied the prototype. Distinct from `missing`
        # (conversion/import failure) because these units still have real
        # JSON light data that must be placed.
        no_geometry = set()
        n_instances_placed = 0
        n_mass_placed = 0
        progress_started = False

        # --- Speed setup: keep the growing level out of the depsgraph ---
        # Every bpy.ops call (each glTF prototype import) re-evaluates all
        # visible objects, so imports get slower as the scene fills. Excluding
        # the level collection keeps each import O(new objects) instead of
        # O(everything imported so far). Done LAST before the try block so
        # the finally clause is guaranteed to restore undo/visibility.
        level_excluded = False
        undo_prefs = context.preferences.edit
        saved_global_undo = undo_prefs.use_global_undo
        tmp_dir = tempfile.mkdtemp(prefix="pd2_level_import_")
        wm.progress_begin(0, len(unique_paths) + len(units) + n_inst_units
                          + len(mass_instances))
        progress_started = True
        if self.fast_import:
            level_excluded = set_collection_excluded(context, level_col, True)
            undo_prefs.use_global_undo = False
            log("Fast import: level collection hidden until finished, "
                "global undo off")

        try:
            # --- PASS 1: convert all unique .models in PARALLEL ---
            t0 = time.time()
            glb_map, conv_missing = convert_all_models_parallel(
                parser_exe, unique_paths, assets_dir, tmp_dir,
                max_workers=self.parallel_workers,
                progress_cb=lambda d, n: wm.progress_update(d),
                low_priority=self.low_cpu_priority,
                use_cache=self.use_conversion_cache)
            missing |= conv_missing
            n_failed += len(conv_missing)
            t_convert = time.time() - t0
            log(f"Conversion pass done in {t_convert:.1f}s "
                f"({len(glb_map)} ok, {len(conv_missing)} missing/failed)")

            # --- PASS 2: import each unique .glb ONCE as a prototype ---
            # NOTE: we cache object NAMES, not references. Every bpy.ops call
            # (like importing the next prototype) may invalidate Python
            # references to previously created objects (StructRNA removed),
            # so handles must be re-fetched fresh each time they're used.

            def _fetch(names):
                objs = []
                for n in names:
                    o = bpy.data.objects.get(n)
                    if o is None:
                        return None
                    objs.append(o)
                return objs

            def _import_prototype(path):
                """Returns the prototype object list, or None when the model
                could not be imported at all. An EMPTY list is a valid,
                non-failing result: the model imported but the only-g_ filter
                removed everything. Callers must distinguish the two, because
                a unit with no geometry can still carry JSON lights."""
                glb_path = glb_map.get(path)
                if glb_path is None:
                    return None
                objs = import_glb(glb_path, proto_col)
                if not objs:
                    log_error(f"  glTF import produced no objects for {path}")
                    missing.add(path)
                    return None
                # Record each light's shadow-projection verdict while the
                # model hierarchy is still intact — the filter below deletes
                # the parent nodes whose names carry it.
                stamp_light_shadow_flags(objs)
                if self.only_g_meshes:
                    objs = filter_g_meshes(objs)
                    if not objs:
                        no_geometry.add(path)
                        vlog(f"  -> no g_ meshes in {path}; unit will be "
                             f"placed as an empty (lights still apply)")
                if self.import_textures and objs:
                    apply_pd2_materials(objs, path, assets_dir)
                if objs:
                    apply_light_cone_mode(objs, self.light_cone_mode,
                                          self.light_cone_strength,
                                          self.light_cone_density,
                                          self.light_cone_falloff)

                prototypes[path] = [o.name for o in objs]
                return objs

            def get_prototype(path):
                names = prototypes.get(path)
                if names is not None:
                    objs = _fetch(names)
                    if objs is not None:
                        return objs
                    log(f"  -> prototype references stale, re-importing {path}")
                return _import_prototype(path)

            # Import every prototype UP FRONT, before any duplicates exist.
            # These models reuse generic node names (g_g, dm_wood, c_box_01
            # ...), and Blender's find-a-free-".NNN"-suffix naming gets
            # slower as thousands of same-named duplicates pile up — which
            # is why lazy mid-placement imports crept from ~0.01s to ~0.25s.
            # Importing into a near-empty datablock pool keeps every glTF
            # import at full speed.
            t0 = time.time()
            for pi, path in enumerate(unique_paths, 1):
                if path not in missing:
                    if _import_prototype(path) is None:
                        n_failed += 1
                if pi % 50 == 0 or pi == len(unique_paths):
                    log(f"Importing prototypes... {pi}/{len(unique_paths)}")
            t_import += time.time() - t0

            # --- PASS 3: place every unit ---
            for i, unit in enumerate(units, 1):
                wm.progress_update(len(unique_paths) + i)
                path = unit["path"]
                name_id = unit["name_id"]
                if self.verbose_log:
                    log(f"({i}/{len(units)}) {name_id}  <-  {path}")
                elif i % 100 == 0 or i == len(units):
                    log(f"Placing units... {i}/{len(units)}")

                root = None
                new_objects = []

                if path in missing:
                    if self.verbose_log:
                        log("  -> known missing/failed, skipping")
                else:
                    t0 = time.time()
                    protos = get_prototype(path)
                    if protos:
                        if self.instance_repeats:
                            try:
                                new_objects = duplicate_objects(protos, level_col)
                            except ReferenceError:
                                # References died mid-loop; re-import the
                                # prototype once and retry.
                                log("  -> stale references, rebuilding prototype")
                                prototypes.pop(path, None)
                                protos = get_prototype(path)
                                new_objects = (duplicate_objects(protos, level_col)
                                               if protos else [])
                        else:
                            # Full independent re-import per unit (slow path)
                            new_objects = import_glb(glb_map[path], level_col)
                            if new_objects:
                                stamp_light_shadow_flags(new_objects)
                            if self.only_g_meshes and new_objects:
                                new_objects = filter_g_meshes(new_objects)
                                if not new_objects:
                                    no_geometry.add(path)
                            if new_objects:
                                apply_light_cone_mode(
                                    new_objects, self.light_cone_mode,
                                    self.light_cone_strength,
                                    self.light_cone_density,
                                    self.light_cone_falloff)
                            if self.import_textures and new_objects:
                                apply_pd2_materials(new_objects, path,
                                                    assets_dir)
                    t_import += time.time() - t0
                    if (not new_objects and path not in missing
                            and path not in no_geometry):
                        n_failed += 1
                        missing.add(path)

                # Build the unit: fresh zeroed root empty + parenting
                if new_objects:
                    root = build_unit(new_objects, name_id, level_col)
                    n_imported += 1

                if root is None:
                    if not self.create_empties and not (
                            self.import_lights and unit["lights"]):
                        continue
                    root = bpy.data.objects.new(name_id, None)
                    root.empty_display_type = 'ARROWS'
                    root.empty_display_size = 0.25
                    level_col.objects.link(root)
                    n_empty += 1
                    if self.verbose_log:
                        log("  -> placeholder empty created (no model)")

                # 4. Lights: apply JSON settings onto the model's imported
                # light nodes (keeping their rotation), or strip model
                # lights entirely if light import is disabled
                if self.import_lights and unit["lights"]:
                    root_name = root.name
                    n_lights += create_unit_lights(
                        unit, root, level_col, self.light_power_scale,
                        unit_objects=new_objects, unit_path=path)
                    # light removal can invalidate references — re-fetch
                    root = bpy.data.objects.get(root_name) or root
                elif self.import_lights and new_objects:
                    # No JSON light settings: KEEP the model's light nodes,
                    # just apply the shadow policy
                    n_lights += apply_model_light_defaults(new_objects,
                                                           [path])
                elif new_objects:
                    # Light import disabled: remove model light nodes
                    root_name = root.name
                    new_objects = strip_imported_lights(new_objects)
                    root = bpy.data.objects.get(root_name) or root

                # 5. Apply THIS unit's JSON position/rotation to THIS root
                apply_unit_transform(
                    root, unit["position"], unit["rotation"],
                    rotation_order=self.rotation_order,
                    flip_rot=(self.flip_rot_x, self.flip_rot_y, self.flip_rot_z),
                    flip_pos=(self.flip_pos_x, self.flip_pos_y, self.flip_pos_z),
                )
                root["unit_id"] = unit["unit_id"]
                root["unit_path"] = path
                if self.verbose_log:
                    e = root.rotation_euler
                    log(f"  -> rot({self.rotation_order}) "
                        f"X={math.degrees(e.x):.2f} Y={math.degrees(e.y):.2f} "
                        f"Z={math.degrees(e.z):.2f}  "
                        f"pos=({root.location.x:.3f}, {root.location.y:.3f}, {root.location.z:.3f})")

            # --- PASS 4: place instance contents ---
            n_instances_placed = 0
            prog = len(unique_paths) + len(units)
            for g in inst_groups:
                inst = g["instance"]
                inst_col = bpy.data.collections.new(inst["name"])
                level_col.children.link(inst_col)

                inst_root = bpy.data.objects.new(inst["name"], None)
                inst_root.empty_display_type = 'CUBE'
                inst_root.empty_display_size = 0.5
                inst_col.objects.link(inst_root)
                brute_force_zero_transform(inst_root)
                inst_root["instance_folder"] = inst["folder"]
                inst_root["instance_id"] = inst["id"]

                log(f"Instance '{inst['name']}' "
                    f"({len(g['units'])} units)...")

                for iu in g["units"]:
                    prog += 1
                    wm.progress_update(prog)
                    path = iu["path"]
                    objs = []
                    # An unavailable or empty prototype used to `continue`
                    # here, which jumped straight over the light block below
                    # — so every JSON light on an instance unit whose model
                    # failed, or which the only-g_ filter emptied, was
                    # silently discarded. The main unit pass has always
                    # handled this by falling through to a placeholder root;
                    # this pass now does the same.
                    if path not in missing:
                        protos = get_prototype(path)
                        if protos is None:
                            n_failed += 1
                            missing.add(path)
                        elif protos:
                            try:
                                objs = duplicate_objects(protos, inst_col)
                            except ReferenceError:
                                prototypes.pop(path, None)
                                protos = get_prototype(path)
                                objs = (duplicate_objects(protos, inst_col)
                                        if protos else [])
                            if not objs:
                                n_failed += 1

                    if objs:
                        sub_root = build_unit(objs, iu["name_id"], inst_col)
                        n_imported += 1
                    else:
                        if not self.create_empties and not (
                                self.import_lights and iu["lights"]):
                            continue
                        sub_root = bpy.data.objects.new(iu["name_id"], None)
                        sub_root.empty_display_type = 'ARROWS'
                        sub_root.empty_display_size = 0.25
                        inst_col.objects.link(sub_root)
                        brute_force_zero_transform(sub_root)
                        n_empty += 1
                    sub_root.parent = inst_root
                    sub_root.matrix_parent_inverse.identity()
                    sub_name, inst_name = sub_root.name, inst_root.name
                    if self.import_lights and iu["lights"]:
                        n_lights += create_unit_lights(
                            iu, sub_root, inst_col, self.light_power_scale,
                            unit_objects=objs, unit_path=path)
                    elif self.import_lights and objs:
                        n_lights += apply_model_light_defaults(objs, [path])
                    elif objs:
                        objs = strip_imported_lights(objs)
                    sub_root = bpy.data.objects.get(sub_name) or sub_root
                    inst_root = bpy.data.objects.get(inst_name) or inst_root
                    apply_instance_unit_transform(
                        sub_root, iu["position"], iu["quat"])
                    sub_root["unit_path"] = path

                # Instance root gets the level-JSON transform LAST, so the
                # children keep their local continent placements. No -90 X
                # here: each child already carries its own upright fix.
                apply_unit_transform(
                    inst_root, inst["position"], inst["rotation"],
                    rotation_order=self.rotation_order,
                    flip_rot=(self.flip_rot_x, self.flip_rot_y, self.flip_rot_z),
                    flip_pos=(self.flip_pos_x, self.flip_pos_y, self.flip_pos_z),
                    upright=False,
                )
                n_instances_placed += 1

            # --- PASS 5: massunit scatter ---
            # World-space absolute placement, same convention as instance
            # units (position/100 + quaternion + the -90 X upright fix).
            if mass_instances:
                mass_col = bpy.data.collections.new(level_name + "_massunits")
                level_col.children.link(mass_col)
                for mi_i, mi in enumerate(mass_instances, 1):
                    prog += 1
                    wm.progress_update(prog)
                    path = mi["path"]
                    if path in missing:
                        continue
                    protos = get_prototype(path)
                    if protos is None:
                        missing.add(path)
                        n_failed += 1
                        continue
                    if not protos:
                        # Filtered to nothing by only-g_. Massunits carry no
                        # JSON light data, so there is nothing to place —
                        # but this is not a failure and must not poison the
                        # path for the other passes.
                        continue
                    try:
                        objs = duplicate_objects(protos, mass_col)
                    except ReferenceError:
                        prototypes.pop(path, None)
                        protos = get_prototype(path)
                        objs = (duplicate_objects(protos, mass_col)
                                if protos else [])
                    if not objs:
                        n_failed += 1
                        continue
                    root = build_unit(objs, os.path.basename(path), mass_col)
                    if self.import_lights:
                        apply_model_light_defaults(objs, [path])
                    else:
                        strip_imported_lights(objs)
                    apply_instance_unit_transform(
                        root, mi["position"], mi["quat"])
                    root["unit_path"] = path
                    root["massunit"] = True
                    n_mass_placed += 1
                    if mi_i % 250 == 0 or mi_i == len(mass_instances):
                        log(f"Placing massunits... {mi_i}/"
                            f"{len(mass_instances)}")
        finally:
            if progress_started:
                wm.progress_end()
            # Restore undo and reveal the finished level
            undo_prefs.use_global_undo = saved_global_undo
            if level_excluded:
                set_collection_excluded(context, level_col, False)
            # Delete the prototype objects (mesh data is shared with the
            # placed copies, so it survives) and the hidden collection.
            # Data left with zero users (models that were converted but never
            # successfully placed) is freed too, so it doesn't bloat the file.
            orphan_data = []
            for names in prototypes.values():
                for n in names:
                    o = bpy.data.objects.get(n)
                    if o is not None:
                        try:
                            data = o.data
                            bpy.data.objects.remove(o, do_unlink=True)
                            if data is not None and data.users == 0:
                                orphan_data.append(data)
                        except Exception:
                            pass
            for data in orphan_data:
                try:
                    if isinstance(data, bpy.types.Mesh):
                        bpy.data.meshes.remove(data)
                    elif isinstance(data, bpy.types.Light):
                        bpy.data.lights.remove(data)
                    elif isinstance(data, bpy.types.Armature):
                        bpy.data.armatures.remove(data)
                except Exception:
                    pass
            try:
                bpy.data.collections.remove(proto_col)
            except Exception:
                pass
            # Clean up temp .glb files
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            # Strip the importer's private markers from the finished level
            clear_internal_markers(level_col)

        # --- Material merge pass ---
        if self.merge_materials:
            log("Merging duplicate materials...")
            merge_duplicate_materials()

        bpy.context.view_layer.update()

        elapsed = time.time() - t_start
        summary = (f"Done in {elapsed:.1f}s (convert {t_convert:.1f}s, "
                   f"gltf {t_import:.1f}s) — {n_imported} models, "
                   f"{n_lights} lights, {n_empty} empties, {n_failed} failed, "
                   f"{len(missing)} missing/failed paths, "
                   f"{n_instances_placed} instances placed, "
                   + (f"{n_mass_placed} massunits placed, "
                      if n_mass_placed else "")
                   + f"{n_inst_ok} instances converted"
                   + (f" ({n_inst_missing} instance files missing)"
                      if n_inst_missing else ""))
        log(summary)
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        prefs = context.preferences.addons[__name__].preferences

        box = layout.box()
        box.label(text="Assets Directory (set in Preferences):", icon='FILEBROWSER')
        if prefs.assets_directory:
            box.label(text=prefs.assets_directory)
            if not os.path.isdir(bpy.path.abspath(prefs.assets_directory)):
                box.label(text="Directory does not exist!", icon='ERROR')
        else:
            box.label(text="Not set — configure in addon preferences", icon='ERROR')

        layout.separator()
        layout.prop(self, "create_empties")
        layout.prop(self, "merge_materials")
        layout.prop(self, "instance_repeats")
        layout.prop(self, "fast_import")
        layout.prop(self, "verbose_log")
        layout.prop(self, "parallel_workers")
        layout.prop(self, "low_cpu_priority")
        layout.prop(self, "use_conversion_cache")
        layout.prop(self, "only_g_meshes")
        layout.prop(self, "glass_fresnel")
        layout.prop(self, "light_cone_mode")
        if self.light_cone_mode in {'FAKE', 'VOLUME'}:
            layout.prop(self, "light_cone_strength")
        if self.light_cone_mode == 'FAKE':
            layout.prop(self, "light_cone_falloff")
        if self.light_cone_mode == 'VOLUME':
            layout.prop(self, "light_cone_density")
        layout.prop(self, "import_textures")
        layout.prop(self, "convert_instances")
        layout.prop(self, "import_instances")
        layout.prop(self, "import_massunits")

        layout.separator()
        layout.label(text="Lights:")
        layout.prop(self, "import_lights")
        row = layout.row()
        row.enabled = self.import_lights
        row.prop(self, "light_power_scale")

        layout.separator()
        layout.label(text="Transform Tuning:")
        layout.prop(self, "rotation_order")
        row = layout.row(align=True)
        row.prop(self, "flip_rot_z", toggle=True)
        row.prop(self, "flip_rot_x", toggle=True)
        row.prop(self, "flip_rot_y", toggle=True)
        row = layout.row(align=True)
        row.prop(self, "flip_pos_x", toggle=True)
        row.prop(self, "flip_pos_y", toggle=True)
        row.prop(self, "flip_pos_z", toggle=True)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_pd2_level_json.bl_idname,
                         text="PAYDAY 2 Level (.json)")


classes = (PD2LevelImporterPreferences, PD2_OT_clear_glb_cache,
           IMPORT_OT_pd2_level_json)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    # These map onto bpy datablock NAMES, which mean nothing after a reload.
    reset_module_caches()


if __name__ == "__main__":
    register()