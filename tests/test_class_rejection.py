"""固定logitsに汎用のクラス別棄却を校正し、全条件を維持する。"""
import unittest
import numpy as np
from arcade_ai_pet import class_rejection as rejection
from arcade_ai_pet.domain_training import select_threshold
from arcade_ai_pet.training import decisions as previous
from arcade_ai_pet.channel_robustness import measure


def fixture():
    y=np.tile(np.arange(7),16);groups=np.repeat(np.arange(16),7)
    logits=np.eye(7,dtype=np.int64)[y]*100
    logits[y==0]=0;logits[y==0,0]=2
    logits[y==5]=0;logits[y==5,1]=3
    return y,logits,groups


class ClassRejectionTests(unittest.TestCase):
    def test_equal_thresholds_match_existing_decisions_and_inputs_are_unchanged(self):
        logits=np.random.default_rng(5).integers(-300,301,(100,7),dtype=np.int64);before=logits.copy()
        for threshold in (0,100,4096):
            vector=np.full(5,threshold,dtype=np.int32)
            np.testing.assert_array_equal(rejection.decisions(logits,vector),previous(logits,threshold))
        np.testing.assert_array_equal(logits,before)

    def test_only_commands_are_rejected_and_ties_keep_the_first_argmax(self):
        logits=np.array([[5,4,0,0,0,0,0],[0,0,0,0,0,9,8],[0,0,0,0,0,8,9],[5,5,0,0,0,0,0]],dtype=np.int64)
        np.testing.assert_array_equal(rejection.decisions(logits,np.array([2,0,0,0,0])),[5,5,6,5])
        np.testing.assert_array_equal(rejection.decisions(logits,np.zeros(5,dtype=np.int64)),[0,5,6,0])

    def test_wide_int32_margin_does_not_overflow(self):
        logits=np.full((1,7),-(2**31),dtype=np.int64);logits[0,0]=2**31-1
        np.testing.assert_array_equal(rejection.decisions(logits,np.full(5,4096)),[0])

    def test_invalid_logits_and_threshold_tables_are_rejected(self):
        good=np.zeros((1,7),dtype=np.int64);vector=np.zeros(5,dtype=np.int64)
        for bad in (np.zeros((0,7),dtype=np.int64),np.zeros((1,6),dtype=np.int64),np.zeros((1,7)),np.full((1,7),2**31,dtype=np.int64)):
            with self.assertRaises(ValueError):rejection.decisions(bad,vector)
        for bad in (np.zeros(4,dtype=np.int64),np.zeros(5),np.full(5,-1),np.full(5,4097),np.zeros(5,dtype=bool)):
            with self.assertRaises(ValueError):rejection.decisions(good,bad)

    def test_calibration_can_solve_different_class_confidences_without_global_relaxation(self):
        y,logits,g=fixture();before=logits.copy()
        self.assertFalse(select_threshold(y,logits,g)[2])
        result=rejection.calibrate(y,logits,g,initial_threshold=4,max_passes=4)
        self.assertTrue(result['accepted_validation']);self.assertEqual(len(result['thresholds_q10']),5)
        self.assertLess(result['thresholds_q10'][0],result['thresholds_q10'][1])
        self.assertEqual(len(result['domain_profiles']),16);self.assertEqual(len(result['pooled_profiles']),8)
        prediction=rejection.decisions(logits,np.asarray(result['thresholds_q10']))
        for index,row in enumerate(result['domain_profiles']):
            self.assertEqual(row['macro_f1'],measure(y[g==index],prediction[g==index])['macro_f1'])
        for index,row in enumerate(result['pooled_profiles']):
            self.assertEqual(row['macro_f1'],measure(y[g%8==index],prediction[g%8==index])['macro_f1'])
        np.testing.assert_array_equal(logits,before)
        self.assertLessEqual(result['evaluations'],2*(1+4*(len(y)+5)))

    def test_one_bad_domain_is_not_hidden_and_row_order_is_irrelevant(self):
        y,logits,g=fixture();logits[0]=0;logits[0,2]=5000
        a=rejection.calibrate(y,logits,g,initial_threshold=4,max_passes=4)
        order=np.random.default_rng(1).permutation(len(y))
        b=rejection.calibrate(y[order],logits[order],g[order],initial_threshold=4,max_passes=4)
        self.assertFalse(a['accepted_validation']);self.assertEqual(a['thresholds_q10'],b['thresholds_q10'])

    def test_valid_unsigned_labels_and_groups_keep_the_same_calibration(self):
        y,logits,g=fixture();expected=rejection.calibrate(y,logits,g,initial_threshold=4)
        actual=rejection.calibrate(y.astype(np.uint64),logits,g.astype(np.uint64),initial_threshold=4)
        self.assertEqual(actual['thresholds_q10'],expected['thresholds_q10'])
        self.assertEqual(actual['accepted_validation'],expected['accepted_validation'])

    def test_incomplete_groups_and_unbounded_search_options_are_rejected(self):
        y,logits,g=fixture()
        with self.assertRaises(ValueError):rejection.calibrate(y,logits,g%8)
        for bad in (0,9,True):
            with self.assertRaises(ValueError):rejection.calibrate(y,logits,g,max_passes=bad)
        for bad in (-1,4097,True):
            with self.assertRaises(ValueError):rejection.calibrate(y,logits,g,initial_threshold=bad)


if __name__=='__main__':unittest.main()
