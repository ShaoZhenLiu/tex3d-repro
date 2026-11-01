import unittest
import numpy as np
from measurement_core import measure_three


class Tests(unittest.TestCase):
    def test_low_contrast_does_not_mean_invisible_geometry(self):
        p=np.full((4,4,3),100,dtype=np.uint8)
        imgs=dict(plain=p,hidden=p.copy(),refcloth=p+20,marker_a=p+80,marker_b=p-80)
        r=measure_three(imgs)['thresholds']['10']
        self.assertEqual(r['marker_fraction'],1)
        self.assertEqual(r['plain_hidden_fraction'],0)
        self.assertEqual(r['ref_plain_fraction'],1)
        self.assertEqual(r['hidden_missed_marker_fraction'],1)

    def test_unsigned_subtraction_and_threshold_units(self):
        a=np.zeros((2,2,3),dtype=np.uint8);b=np.full_like(a,255)
        r=measure_three(dict(plain=a,hidden=b,refcloth=a,marker_a=a,marker_b=b))
        self.assertEqual(r['maxabs_rgb255']['plain_hidden'],255)
        self.assertEqual(r['thresholds']['20']['plain_hidden_fraction'],1)

    def test_empty_marker_is_not_silently_a_zero_miss_rate(self):
        p=np.zeros((2,2,3))
        r=measure_three(dict(plain=p,hidden=p,refcloth=p,marker_a=p,marker_b=p))
        self.assertIsNone(r['thresholds']['10']['hidden_missed_marker_fraction'])

    def test_model_float_roundoff_is_not_a_real_threshold_crossing(self):
        p=np.zeros((2,2,3));b=p+10.00001
        r=measure_three(dict(plain=p,hidden=b,refcloth=b,marker_a=p,marker_b=b))
        self.assertEqual(r['thresholds']['10']['plain_hidden_fraction'],0)
        self.assertEqual(r['thresholds']['5']['plain_hidden_fraction'],1)


if __name__=='__main__':unittest.main()
