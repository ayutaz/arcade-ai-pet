"""時間畳み込みのQ10演算・容量・旧export拒否の契約。"""
import tempfile,unittest
from pathlib import Path
import numpy as np
from arcade_ai_pet import temporal_model as temporal
from arcade_ai_pet.frozen import read_layers


def fixture(seed=9):
    rng=np.random.default_rng(seed)
    return [dict(weight=rng.integers(-3,4,shape,dtype=np.int8),bias=rng.integers(-500,501,shape[0],dtype=np.int32),shift=3) for shape in ((24,65),(32,72),(32,96),(64,160),(7,64))]


def reference(x,layers):
    q=np.rint(x*1024).astype(np.int64)
    sequence=np.column_stack((q[:240].reshape(20,12),q[240:]))
    for layer,kernel in zip(layers[:3],(5,3,3),strict=True):
        width=sequence.shape[1];padded=np.pad(sequence,((kernel//2,kernel//2),(0,0)));out=np.zeros((20,layer['weight'].shape[0]),dtype=np.int64)
        for time in range(20):
            # Independent scalar reference: weight order is channel, then kernel time.
            for channel in range(out.shape[1]):
                value=int(layer['bias'][channel])
                for incoming in range(width):
                    for offset in range(kernel):value+=int(padded[time+offset,incoming])*int(layer['weight'][channel,incoming*kernel+offset])
                out[time,channel]=min(6144,max(0,value>>layer['shift']))
        sequence=out
    pooled=np.array([sum(sequence[t,c] for t in range(group*4,(group+1)*4))//4 for group in range(5) for c in range(32)],dtype=np.int64)
    for index,layer in enumerate(layers[3:]):
        pooled=np.array([(sum(int(w)*int(v) for w,v in zip(row,pooled,strict=True))+int(bias))>>layer['shift'] for row,bias in zip(layer['weight'],layer['bias'],strict=True)],dtype=np.int64)
        if index==0:pooled=np.clip(pooled,0,6144)
    return pooled


class TemporalModelTests(unittest.TestCase):
    def test_feature_layout_preserves_time_and_appends_energy_channel(self):
        x=np.zeros((2,260),dtype=np.float32);x[0,12:24]=2;x[0,241]=3
        sequence=temporal.sequence(x)
        self.assertEqual(sequence.shape,(2,20,13));np.testing.assert_array_equal(sequence[0,1],np.array([2]*12+[3]))
        self.assertEqual(sequence[1].sum(),0)

    def test_integer_pipeline_matches_independent_scalar_reference(self):
        layers=fixture();x=np.random.default_rng(3).uniform(-3,3,(2,260)).astype(np.float32)
        expected=np.stack([reference(row,layers) for row in x]);actual=temporal.scores(x,layers)
        np.testing.assert_array_equal(actual,expected);self.assertTrue(np.issubdtype(actual.dtype,np.integer))
        np.testing.assert_array_equal(actual[:1],temporal.scores(x[:1],layers))

    def test_bounds_and_numeric_budget_are_explicit(self):
        self.assertEqual(temporal.weights_bytes(fixture()),18280)
        for bad in (np.zeros((1,259)),np.full((1,260),np.nan),np.full((1,260),17.),np.full((1,260),np.iinfo(np.int64).min),np.zeros((0,260))):
            with self.assertRaises(ValueError):temporal.scores(bad,fixture())
        for field,value in (('shift',True),('shift',17),('weight',np.zeros((24,65),dtype=np.int16))):
            layers=fixture();layers[0][field]=value
            with self.assertRaises(ValueError):temporal.scores(np.zeros((1,260)),layers)

    def test_legacy_two_layer_reader_refuses_temporal_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.npz';layers=fixture()
            np.savez(path,**{f'{k}{i}':r[k] for i,r in enumerate(layers) for k in ('weight','bias','shift')})
            with self.assertRaises(ValueError):read_layers(path)


if __name__=='__main__':unittest.main()
