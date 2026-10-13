import numpy as np
from typing import List

def find_target_texture_id(
    sim,
    body_id: int,
    search_keywords_list: List[List[str]],
) -> int:
    """Find the MuJoCo texture used by the target object's visual material."""
    candidate_geoms = []
    for geom_id in range(sim.model.ngeom):
        name = None
        try:
            name = sim.model.geom_id2name(geom_id)
        except Exception:
            pass
        belongs_to_body = body_id >= 0 and int(sim.model.geom_bodyid[geom_id]) == int(body_id)
        name_matches = bool(
            name
            and any(all(keyword in name for keyword in keywords)
                    for keywords in search_keywords_list)
        )
        if belongs_to_body or name_matches:
            candidate_geoms.append(geom_id)

    for geom_id in candidate_geoms:
        try:
            material_id = int(sim.model.geom_matid[geom_id])
        except Exception:
            material_id = -1
        if material_id < 0:
            continue
        tex_ids = np.asarray(sim.model.mat_texid[material_id]).reshape(-1)
        valid_tex_ids = tex_ids[tex_ids >= 0]
        if len(valid_tex_ids):
            return int(valid_tex_ids[0])

    # Fallback for assets whose geom/body names do not carry the object name.
    for material_id in range(sim.model.nmat):
        name = None
        try:
            name = sim.model.mat_id2name(material_id)
        except Exception:
            pass
        if not name or not any(
            all(keyword in name for keyword in keywords)
            for keywords in search_keywords_list
        ):
            continue
        tex_ids = np.asarray(sim.model.mat_texid[material_id]).reshape(-1)
        valid_tex_ids = tex_ids[tex_ids >= 0]
        if len(valid_tex_ids):
            return int(valid_tex_ids[0])
    return -1


def _upload_mujoco_texture(sim, texture_id: int) -> bool:
    """Upload modified model.tex_rgb data to every available MuJoCo GL context."""
    import mujoco

    contexts = []
    for attr in ("render_contexts", "_render_context_offscreen", "render_context_offscreen"):
        value = getattr(sim, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            contexts.extend(value)
        else:
            contexts.append(value)

    uploaded = False
    seen = set()
    for context in contexts:
        if id(context) in seen:
            continue
        seen.add(id(context))
        con = getattr(context, "con", None)
        context_model = getattr(context, "model", None)
        if con is None or context_model is None:
            continue
        gl_context = getattr(context, "gl_ctx", None)
        if gl_context is not None:
            gl_context.make_current()
        native_model = getattr(context_model, "_model", context_model)
        mujoco.mjr_uploadTexture(native_model, con, texture_id)
        uploaded = True
    return uploaded


def inject_mujoco_texture(sim, texture_id: int, texture_rgb: np.ndarray):
    """Replace one MuJoCo texture in memory and return data needed for restoration."""
    if texture_id < 0:
        raise ValueError("Invalid MuJoCo texture id")

    # robosuite's compatibility MjModel forwards most MuJoCo arrays, but not
    # tex_rgb in this release. Read all texture storage from the native model.
    native_model = getattr(sim.model, "_model", sim.model)
    height = int(native_model.tex_height[texture_id])
    width = int(native_model.tex_width[texture_id])
    offset = int(native_model.tex_adr[texture_id])
    channels = int(native_model.tex_nchannel[texture_id]) if hasattr(
        native_model, "tex_nchannel"
    ) else 3
    byte_count = height * width * channels
    texture_storage = getattr(
        native_model, "tex_data", getattr(native_model, "tex_rgb", None)
    )
    if texture_storage is None:
        raise RuntimeError("MuJoCo model exposes neither tex_data nor tex_rgb")

    texture_rgb = np.asarray(texture_rgb, dtype=np.uint8)
    if texture_rgb.ndim != 3 or texture_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB texture, got {texture_rgb.shape}")
    if texture_rgb.shape[:2] != (height, width):
        texture_rgb = np.asarray(
            Image.fromarray(texture_rgb).resize((width, height), Image.BILINEAR),
            dtype=np.uint8,
        )
    if channels == 4:
        alpha = np.full((*texture_rgb.shape[:2], 1), 255, dtype=np.uint8)
        texture_rgb = np.concatenate([texture_rgb, alpha], axis=2)
    elif channels != 3:
        raise ValueError(f"Unsupported MuJoCo texture channel count: {channels}")

    original = np.asarray(
        texture_storage[offset:offset + byte_count], dtype=np.uint8
    ).copy()
    texture_storage[offset:offset + byte_count] = texture_rgb.reshape(-1)
    uploaded = _upload_mujoco_texture(sim, texture_id)
    if not uploaded:
        print(
            "[warning] MuJoCo texture data changed, but no upload_texture context "
            "was found; the video renderer may keep the old GPU texture"
        )
    return original, offset, byte_count


def restore_mujoco_texture(sim, texture_id: int, original, offset: int, byte_count: int):
    native_model = getattr(sim.model, "_model", sim.model)
    texture_storage = getattr(
        native_model, "tex_data", getattr(native_model, "tex_rgb", None)
    )
    if texture_storage is None:
        raise RuntimeError("MuJoCo model exposes neither tex_data nor tex_rgb")
    texture_storage[offset:offset + byte_count] = original
    _upload_mujoco_texture(sim, texture_id)

