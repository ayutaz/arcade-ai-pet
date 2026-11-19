"""旧データの多さで新しい声の失敗が隠れない選定用の検査。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import domain_validation as domain
from arcade_ai_pet.channel_robustness import PROFILES
import test_frozen as fixture


def model(root):
    path=root/'candidate';fixture.checkpoint(path)
    (path/'validation.json').write_text(json.dumps(dict(weights_sha256=hashlib.sha256((path/'weights.npz').read_bytes()).hexdigest(),threshold_q10=0,candidate='hp160',frontend_version='adaptive',accepted_validation=True)))
    return path


def dataset(root,name,role='development',shared=None):
    path=root/name;path.mkdir();source=shared or name
    (path/'manifest.json').write_text(json.dumps(dict(role=role,records=[dict(id=hashlib.sha256(source.encode()).hexdigest(),sha256=hashlib.sha256((source+'wave').encode()).hexdigest())])))
    return path


def arrays(correct=True):
    y=np.tile(np.arange(7),8);x=np.zeros((56,260),dtype=np.float32)
    if correct:x[np.arange(56),y]=4
    return dict(validation=dict(x=x,y=y,profiles=[p for p in PROFILES for _ in range(7)]))


class DomainValidationTests(unittest.TestCase):
    def test_bad_new_domain_rejects_even_when_aggregate_success_flag_is_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();candidate=model(root);old=dataset(root,'old');new=dataset(root,'new')
            with patch.object(domain,'make_arrays',side_effect=lambda path,*a,**k:arrays(path==old)):
                report=domain.validate_domains(candidate,{'legacy':old,'neural':new})
            self.assertFalse(report['accepted_all_domains']);self.assertTrue(report['domains']['legacy']['accepted'])
            self.assertFalse(report['domains']['neural']['accepted']);self.assertTrue(report['validation_only'])
            self.assertFalse(report['real_voice_tested']);self.assertFalse(report['device_tested'])

    def test_both_domains_require_all_eight_profiles_with_fixed_checkpoint_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();candidate=model(root);old=dataset(root,'old');new=dataset(root,'new')
            with patch.object(domain,'make_arrays',side_effect=lambda *a,**k:arrays()):
                report=domain.validate_domains(candidate,{'legacy':old,'neural':new})
            self.assertTrue(report['accepted_all_domains']);self.assertEqual(report['threshold_q10'],0)
            for result in report['domains'].values():self.assertEqual({r['profile'] for r in result['profiles']},set(PROFILES))

    def test_final_role_changed_weights_and_overlapping_sources_refuse_before_features(self):
        for change in ('final','weights','overlap'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();candidate=model(root);old=dataset(root,'old');new=dataset(root,'new',role='final' if change=='final' else 'development',shared='old' if change=='overlap' else None)
                if change=='weights':
                    with (candidate/'weights.npz').open('ab') as stream:stream.write(b'changed')
                with patch.object(domain,'make_arrays',side_effect=AssertionError('must refuse before features')):
                    with self.assertRaises(ValueError):domain.validate_domains(candidate,{'legacy':old,'neural':new})


if __name__=='__main__':unittest.main()
