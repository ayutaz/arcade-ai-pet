"""先行契約: 学習用maskの軸・境界・独立RNG。Torch不要。"""
import unittest
import numpy as np
from arcade_ai_pet import spectral_mask as mask


class SpectralMaskPlanTests(unittest.TestCase):
    def test_plan_is_typed_bounded_reproducible_and_advances_only_its_rng(self):
        np.random.seed(7); global_state=np.random.get_state()
        a=np.random.default_rng(43); b=np.random.default_rng(43)
        first=mask.make_plan(1024,a)
        self.assertEqual(first.shape,(1024,4)); self.assertEqual(first.dtype,np.int32)
        np.testing.assert_array_equal(first,mask.make_plan(1024,b))
        self.assertFalse(np.array_equal(first,mask.make_plan(1024,a)))
        self.assertTrue(np.all((first[:,1]>=0)&(first[:,1]<=6)))
        self.assertTrue(np.all((first[:,3]>=0)&(first[:,3]<=3)))
        self.assertTrue(np.all((first[:,0]>=0)&(first[:,0]+first[:,1]<=64)))
        self.assertTrue(np.all((first[:,2]>=0)&(first[:,2]+first[:,3]<=32)))
        self.assertEqual(set(first[:,1]),set(range(7)));self.assertEqual(set(first[:,3]),set(range(4)))
        after=np.random.get_state()
        for x,y in zip(global_state,after,strict=True):np.testing.assert_array_equal(x,y)

    def test_zero_width_and_boundary_plans_are_valid_without_mutation(self):
        plans=np.array([[64,0,32,0],[58,6,29,3]],np.int32);before=plans.copy()
        mask.validate_plan(plans,2);np.testing.assert_array_equal(plans,before)

    def test_invalid_plan_axes_dtype_shape_and_width_refuse(self):
        for row in ([59,6,0,0],[0,7,0,0],[0,0,30,3],[0,0,0,4],[-1,1,0,0],[0,-1,0,0]):
            with self.assertRaises(ValueError):mask.validate_plan(np.array([row],np.int32),1)
        for bad in (np.zeros((1,4),np.int64),np.zeros((1,3),np.int32),[[0,0,0,0]],np.zeros((0,4),np.int32)):
            with self.assertRaises(ValueError):mask.validate_plan(bad,1)

    def test_generator_and_row_count_require_explicit_bounded_values(self):
        for rows in (0,1025,True,1.5):
            with self.assertRaises(ValueError):mask.make_plan(rows,np.random.default_rng(1))
        for rng in (1,None,np.random):
            with self.assertRaises(ValueError):mask.make_plan(1,rng)

if __name__=='__main__':unittest.main()
