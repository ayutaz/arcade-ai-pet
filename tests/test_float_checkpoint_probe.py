"""先行契約: float/整数の順位差と型付きfloat stateを独立に点検する。"""
import copy,json,unittest
import numpy as np
from arcade_ai_pet import dscnn_probe as probe
from arcade_ai_pet.dscnn_model import SHAPES


def state_fixture(*,envelope=False,time_pool=False):
    state={}
    for i,shape in enumerate(SHAPES[:5]):
        state[f'blocks.{i}.0.weight']=np.zeros(shape,dtype=np.float32)
        for name,value in (('weight',1),('bias',0),('running_mean',0),('running_var',1)):
            state[f'blocks.{i}.1.{name}']=np.full(shape[0],value,dtype=np.float32)
        state[f'blocks.{i}.1.num_batches_tracked']=np.array(0,dtype=np.int64)
    state['head.weight']=np.zeros((7,865 if time_pool else 3169 if envelope else 3137),dtype=np.float32)
    state['head.bias']=np.zeros(7,dtype=np.float32)
    return state


def fixture():
    labels=np.tile(np.arange(7),16);groups=np.repeat(np.arange(16),7)
    floats=np.zeros((112,7),dtype=np.float32);integers=np.zeros((112,7),dtype=np.int32)
    for i,y in enumerate(labels):
        a,b={0:(0,0),1:(1,2),2:(3,2),3:(4,4),4:(0,1),5:(5,5),6:(6,6)}[int(y)]
        if y:floats[i,a]=1;integers[i,b]=1024
    return labels,floats,integers,groups


class FloatCheckpointProbeTests(unittest.TestCase):
    def test_raw_pair_counts_ties_and_groups_match_hand_calculation(self):
        args=fixture();before=[a.copy() for a in args];r=probe.compare(*args)
        expected=dict(examples=112,float_correct=64,integer_correct=64,both_correct=48,float_only_correct=16,integer_only_correct=16,both_wrong=32,raw_agreement=64)
        self.assertEqual(r['totals'],expected);self.assertEqual(len(r['groups']),16)
        for group in r['groups']:
            self.assertEqual(group['totals'],{k:v//16 for k,v in expected.items()})
            self.assertEqual(group['by_class'][1]['float_only_correct'],1)
            self.assertEqual(group['by_class'][2]['integer_only_correct'],1)
        json.dumps(r,allow_nan=False)
        for a,b in zip(args,before,strict=True):np.testing.assert_array_equal(a,b)

    def test_invalid_probe_shapes_dtypes_ranges_and_incomplete_groups_refuse(self):
        y,f,i,g=fixture()
        cases=[(y,f.astype(float),i,g),(y,f,i.astype(np.int64),g),(y,f[:,:6],i,g),(y,f,i,g%8)]
        for value in (np.nan,np.inf,1e6+1):
            bad=f.copy();bad[0,0]=value;cases.append((y,bad,i,g))
        for args in cases:
            with self.assertRaises(ValueError):probe.compare(*args)

    def test_all_state_modes_have_stable_order_independent_digest_without_mutation(self):
        for options in ({},{'envelope':True},{'envelope':True,'time_pool':True}):
            state=state_fixture(**options);before=copy.deepcopy(state);probe.validate_state(state,**options)
            self.assertEqual(len(state),32)
            digest=probe.state_digest(state,**options)
            self.assertEqual(digest,probe.state_digest(dict(reversed(list(state.items()))),**options))
            state['head.bias'][0]=.125;self.assertNotEqual(digest,probe.state_digest(state,**options))
            for key in state:
                if key!='head.bias':np.testing.assert_array_equal(state[key],before[key])

    def test_foreign_state_dtype_keys_shape_bn_variance_and_counter_refuse(self):
        for kind in ('extra','missing','shape','dtype','nan','variance','counter'):
            s=state_fixture()
            if kind=='extra':s['foreign']=np.zeros(1,dtype=np.float32)
            elif kind=='missing':del s['head.bias']
            elif kind=='shape':s['head.weight']=s['head.weight'][:,:-1]
            elif kind=='dtype':s['head.bias']=s['head.bias'].astype(float)
            elif kind=='nan':s['head.bias'][0]=np.nan
            elif kind=='variance':s['blocks.0.1.running_var'][0]=-1
            else:s['blocks.0.1.num_batches_tracked'][...]=-1
            with self.assertRaises(ValueError):probe.validate_state(s)
        with self.assertRaises(ValueError):probe.validate_state(state_fixture(envelope=True))
        with self.assertRaises(ValueError):probe.validate_state(state_fixture(),time_pool=True)
        with self.assertRaises(ValueError):probe.validate_state(state_fixture(),envelope=1)

    def test_zero_float_state_binds_known_fixed_integer_weights_and_bias(self):
        state=state_fixture();before=copy.deepcopy(state);layers=probe.quantized_state(state)
        self.assertEqual(len(layers),6)
        for layer,shape in zip(layers,SHAPES,strict=True):
            self.assertEqual(layer['weight'].shape,shape);self.assertEqual(layer['weight'].dtype,np.int8)
            self.assertFalse(layer['weight'].any());self.assertFalse(layer['bias'].any())
            np.testing.assert_array_equal(layer['shift'],np.full(shape[0],16,dtype=np.int32))
        for key in state:np.testing.assert_array_equal(state[key],before[key])

    def test_float_logit_digest_is_canonical_bounded_and_not_a_prediction(self):
        values=np.arange(14,dtype=np.float32).reshape(2,7)
        self.assertEqual(probe.logits_digest(values),probe.logits_digest(np.asfortranarray(values)))
        changed=values.copy();changed[0,0]+=.125
        self.assertNotEqual(probe.logits_digest(values),probe.logits_digest(changed))
        for bad in (values.astype(float),values[:,:6],np.zeros((0,7),dtype=np.float32),np.full((1,7),np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):probe.logits_digest(bad)

if __name__=='__main__':unittest.main()
