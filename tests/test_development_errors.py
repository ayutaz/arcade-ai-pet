"""判定前の誤順位と正解棄却を独立した手計算fixtureで分離する。"""
import unittest
import numpy as np
from arcade_ai_pet.development_errors import audit


def fixture():
    labels=np.array([0,0,1,2,3,5,6,4],dtype=np.int64)
    logits=np.full((8,7),-2,dtype=np.int32)
    for row,best,score,second,value in ((0,0,20,1,0),(1,0,5,1,0),(2,2,20,1,19),
                                       (3,3,20,2,0),(4,5,20,3,0),(5,0,20,5,0),
                                       (6,4,5,6,0),(7,4,10,5,10)):
        logits[row,best]=score;logits[row,second]=value
    return labels,logits,np.array([10,10,10,10,0],dtype=np.int32)


class DevelopmentErrorTests(unittest.TestCase):
    def test_rank_reject_false_activation_and_per_class_partition(self):
        y,z,t=fixture();t[4]=10
        result=audit(y,z,t)
        self.assertEqual(result['examples'],8)
        self.assertEqual(result['commands'],dict(examples=6,raw_top1_correct=3,final_correct=1,
                         correct_top1_rejected=2,wrong_top1=3,wrong_command_accepted=1))
        self.assertEqual(result['negatives'],dict(examples=2,raw_false_activation=2,
                         final_false_activation=1,command_rejected=1))
        self.assertEqual(result['by_class'][0],dict(label=0,examples=2,raw_top1_correct=2,
                         final_correct=1,correct_top1_rejected=1,wrong_top1=0))
        self.assertEqual(result['raw_confusion'][1][2],1)
        self.assertEqual(result['final_confusion'][1][5],1)
        self.assertEqual(result['final_confusion'][4][5],1)
        self.assertEqual(result['correct_rejection_margins_q10'],dict(minimum=0,median=2.5,maximum=5))

    def test_ties_threshold_boundary_large_signed_margins_and_input_invariance(self):
        y,z,t=fixture();before=[a.copy() for a in (y,z,t)]
        r=audit(y,z,t)
        self.assertEqual(r['final_confusion'][4][4],1) # tie first index, margin==threshold
        self.assertEqual(r['commands']['final_correct'],2)
        for a,b in zip((y,z,t),before):np.testing.assert_array_equal(a,b)
        extreme=np.array([[2**31-1,-2**31,-2**31,-2**31,-2**31,-2**31,-2**31]],dtype=np.int32)
        r=audit(np.array([0]),extreme,np.full(5,4096,dtype=np.int32))
        self.assertEqual(r['commands']['final_correct'],1)
        self.assertIsNone(r['correct_rejection_margins_q10'])

    def test_invalid_or_mismatched_arrays_refuse(self):
        y,z,t=fixture()
        cases=[(y[:-1],z,t),(y.astype(float),z,t),(np.full(8,7),z,t),
               (y.reshape(2,4),z,t),(y,z.astype(float),t),(y,z[:,:6],t),
               (y,z,np.zeros(4,dtype=int)),(y,z,np.full(5,4097)),
               (y[:0],z[:0],t),(np.zeros(65537,dtype=int),np.zeros((65537,7),dtype=np.int32),t)]
        for args in cases:
            with self.subTest(shapes=[a.shape for a in args]),self.assertRaises(ValueError):audit(*args)


if __name__=='__main__':unittest.main()
