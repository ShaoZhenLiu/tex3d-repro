import types
import unittest
import numpy as np

from surface_adapter_fixed import align_fixed_floor


def fixture():
    model=types.SimpleNamespace(nsite=0,geom_type=np.array([0,6]),geom_bodyid=np.array([1,2]),
        body_dofnum=np.array([0,0,0]),body_parentid=np.array([0,0,0]),
        geom_contype=np.array([1,0]),geom_conaffinity=np.array([1,0]),
        body_pos=np.array([[0.,0.,0.],[0.,0.,0.],[-.6,0.,.37]]))
    model.geom_name2id=lambda name:0 if name=='floor' else -1
    data=types.SimpleNamespace(geom_xmat=np.array([np.eye(3).reshape(-1)]*2),geom_xpos=np.zeros((2,3)))
    sim=types.SimpleNamespace(model=model,data=data,forward=lambda:None)
    return types.SimpleNamespace(env=types.SimpleNamespace(sim=sim))


class SurfaceTests(unittest.TestCase):
    def test_floating_start_corrected_and_repeat_stable(self):
        env=fixture();first=align_fixed_floor(env,2);second=align_fixed_floor(env,2)
        np.testing.assert_allclose(first,[.4,0,.005]);np.testing.assert_array_equal(first,second)

    def test_rejects_movable_support(self):
        env=fixture();env.env.sim.model.body_dofnum[1]=6
        with self.assertRaises(RuntimeError):align_fixed_floor(env,2)

    def test_rejects_colliding_cloth(self):
        env=fixture();env.env.sim.model.geom_contype[1]=1
        with self.assertRaises(RuntimeError):align_fixed_floor(env,2)

    def test_rejects_tabletop_mode(self):
        env=fixture();env.env.sim.model.nsite=1;env.env.sim.model.site_id2name=lambda sid:'table_top'
        with self.assertRaises(RuntimeError):align_fixed_floor(env,2)

    def test_rejects_non_horizontal_plane(self):
        env=fixture();env.env.sim.data.geom_xmat[0]=np.array([1,0,0,0,0,1,0,-1,0])
        with self.assertRaises(RuntimeError):align_fixed_floor(env,2)


if __name__=='__main__':unittest.main()
