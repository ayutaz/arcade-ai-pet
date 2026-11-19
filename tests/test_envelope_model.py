"""2145入力の整数CNNを別scalarと照合し、旧形式との混在を拒否する。"""
import unittest
import numpy as np
from arcade_ai_pet import dscnn_model as model
from test_dscnn_model import fixture as previous_fixture,reference


def fixture():
    layers=previous_fixture();rng=np.random.default_rng(414)
    layers[-1]['weight']=np.concatenate((layers[-1]['weight'],rng.integers(-2,3,(7,32),dtype=np.int8)),axis=1)
    return layers


class EnvelopeModelTests(unittest.TestCase):
    def test_fixed_envelope_head_numeric_size_and_cross_format_rejection(self):
        layers=fixture();self.assertEqual(model.weights_bytes(layers,envelope=True)+20,23635)
        model.validate(layers,envelope=True)
        with self.assertRaises(ValueError):model.validate(layers)
        with self.assertRaises(ValueError):model.validate(previous_fixture(),envelope=True)
        self.assertEqual(model.activation_layout()['ping_pong_bytes_estimate'],12288)

    def test_complete_integer_pipeline_matches_independent_scalar(self):
        rng=np.random.default_rng(32);values=rng.uniform(-1,1,(1,2145)).astype(np.float32)
        values[:,2048:2112]=-abs(values[:,2048:2112]);values[:,2112]=2;values[0,-1]=-1.75
        before=values.copy();layers=fixture();actual=model.scores(values,layers,envelope=True)
        np.testing.assert_array_equal(actual[0],reference(values[0],layers));np.testing.assert_array_equal(values,before)
        self.assertEqual(actual.dtype,np.int32)

    def test_extra_mean_tail_has_its_own_signed_q5_head_contribution(self):
        layers=fixture()
        for row in layers:row['weight'][:]=0;row['bias'][:]=0;row['shift'][:]=0
        layers[-1]['weight'][0,3137]=2
        values=np.zeros((1,2145),dtype=np.float32);values[0,2113]=-.25;values[0,2112]=4
        np.testing.assert_array_equal(model.scores(values,layers,envelope=True)[0],[-512,0,0,0,0,0,0])

    def test_width_tail_bounds_and_mode_are_explicit(self):
        good=np.zeros((1,2145),dtype=np.float32);good[0,2112]=4;good[0,-1]=-16
        model.scores(good,fixture(),envelope=True)
        invalid=[np.zeros((1,2113),dtype=np.float32),good.astype(float)]
        for index,value in ((2112,5),(2048,1),(2113,17),(2144,np.nan)):
            bad=good.copy();bad[0,index]=value;invalid.append(bad)
        for bad in invalid:
            with self.assertRaises(ValueError):model.scores(bad,fixture(),envelope=True)
        with self.assertRaises(ValueError):model.scores(good,fixture(),envelope='yes')
        broken=fixture();broken[-1]['weight']=broken[-1]['weight'][:,:-1]
        with self.assertRaises(ValueError):model.validate(broken,envelope=True)

    def test_batch_and_individual_results_are_equal(self):
        values=np.zeros((33,2145),dtype=np.float32);values[:,2112]=2;values[:,2113:]=np.linspace(-2,2,33)[:,None]
        layers=fixture();actual=model.scores(values,layers,envelope=True)
        expected=np.vstack([model.scores(v[None],layers,envelope=True)[0] for v in values])
        np.testing.assert_array_equal(actual,expected)

if __name__=='__main__':unittest.main()
