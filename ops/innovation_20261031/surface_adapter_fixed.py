"""Explicit opt-in support for the LIBERO Object floor scene.

Never estimates a support surface from movable objects. Historical placement is
kept as the default in attack_pi0_cloth; this module is selected explicitly with
REFCLOTH_SURFACE_MODE=object_fixed_floor for a separately identified experiment.
"""
import numpy as np


def align_fixed_floor(env, cloth_body_id, xy=(0.4, 0.0), lift=0.005):
    sim=env.env.sim
    sim.forward()
    model=getattr(sim.model,'_model',sim.model)
    data=getattr(sim.data,'_data',sim.data)
    for sid in range(model.nsite):
        if sim.model.site_id2name(sid)=='table_top':
            raise RuntimeError('object_fixed_floor mode cannot be used in a tabletop scene')
    try:gid=sim.model.geom_name2id('floor')
    except Exception as exc:raise RuntimeError('explicit floor geom not found') from exc
    if gid<0 or int(model.geom_type[gid])!=0:
        raise RuntimeError('floor must be an explicit MuJoCo plane')
    support_body=int(model.geom_bodyid[gid])
    if int(model.body_dofnum[support_body])!=0:
        raise RuntimeError('floor support must be static')
    normal=np.asarray(data.geom_xmat[gid]).reshape(3,3)[:,2]
    if not np.allclose(normal,[0,0,1],atol=1e-6):
        raise RuntimeError('floor must be horizontal and upward facing')
    if int(model.body_parentid[cloth_body_id])!=0:
        raise RuntimeError('cloth body must use the world coordinate frame')
    cloth_geoms=np.flatnonzero(np.asarray(model.geom_bodyid)==cloth_body_id)
    if not len(cloth_geoms):raise RuntimeError('cloth has no visual geometry')
    if np.any(model.geom_contype[cloth_geoms]!=0) or np.any(model.geom_conaffinity[cloth_geoms]!=0):
        raise RuntimeError('cloth must remain non-colliding')
    if not 0<lift<=.01:raise ValueError('cloth lift must be in (0, 0.01] meters')
    floor_z=float(data.geom_xpos[gid,2])
    model.body_pos[cloth_body_id]=[float(xy[0]),float(xy[1]),floor_z+lift]
    sim.forward()
    return np.asarray(model.body_pos[cloth_body_id]).copy()
