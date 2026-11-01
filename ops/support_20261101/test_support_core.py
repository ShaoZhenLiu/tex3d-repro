import unittest
import numpy as np
from support_core import binary_texture,opacity_texture,texture_stats,action_metrics,best_deletion


class Tests(unittest.TestCase):
    def test_exact_endpoints_and_export_orientation(self):
        base=np.full((8,8,3),128,dtype=np.uint8)
        full=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        np.testing.assert_array_equal(binary_texture(full,base,[]),base)
        np.testing.assert_array_equal(binary_texture(full,base,range(16)),full)
        out=binary_texture(full,base,[0])
        np.testing.assert_array_equal(out[:2,:2],full[:2,:2])
        np.testing.assert_array_equal(out[2:,:],base[2:,:])

    def test_no_unintended_colors(self):
        base=np.full((256,256,3),123,dtype=np.uint8)
        full=np.random.RandomState(7).randint(0,256,size=base.shape).astype(np.uint8)
        out=binary_texture(full,base,[0,3,6,9])
        self.assertTrue(np.all((out==full)|(out==base)))
        self.assertEqual(texture_stats(out,base,[0,3,6,9])['area_fraction'],.25)
        np.testing.assert_array_equal(opacity_texture(full,base,1),full)
        np.testing.assert_array_equal(opacity_texture(full,base,0),base)

    def test_error_is_relative_to_frozen_teacher(self):
        plain=np.zeros((8,7)); teacher=np.ones((8,7)); scale=np.arange(1,8)
        keep=action_metrics(teacher,teacher,plain,scale)
        self.assertEqual(keep['relative_fidelity'],0)
        self.assertEqual(keep['action_energy_retention'],1)
        gone=action_metrics(plain,teacher,plain,scale)
        self.assertEqual(gone['relative_fidelity'],1)
        self.assertEqual(gone['action_energy_retention'],0)
        self.assertIsNone(gone['displacement_cosine'])
        self.assertIsNone(action_metrics(plain,plain,plain,scale)['relative_fidelity'])

    def test_nonadditive_greedy_must_recompute(self):
        keep,removed=best_deletion((0,1,2),{0:.01,1:.02,2:.03})
        self.assertEqual(removed,0)
        # After removing 0, region 1 becomes essential: old static ranks fail.
        next_keep,removed=best_deletion(keep,{1:10,2:.04})
        self.assertEqual(removed,2)
        self.assertEqual(next_keep,(1,))

    def test_pose_units_and_gripper_separation(self):
        p=np.zeros((8,7)); t=np.ones((8,7)); c=t.copy(); c[:5,3:6]+=1
        m=action_metrics(c,t,p,[2,2,2,3,3,3,1])
        self.assertEqual(m['translation_control_mse'],0)
        self.assertEqual(m['rotation_control_mse'],9)
        c=t.copy(); c[:,6]=-1
        m=action_metrics(c,t,p,[1]*7)
        self.assertEqual(m['relative_fidelity'],0)
        self.assertEqual(m['normalized_gripper_sign_disagreement'],1)


if __name__=='__main__':
    unittest.main()
