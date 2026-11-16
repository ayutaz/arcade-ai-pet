import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from arcade_ai_pet.voice_trials import trial_plan,summarize,main


def results():
    return {'schema':1,'scope':'host_fixture','records':[
        {'trial':row['trial'],'observed':row['expected'],'latency_ms':100+row['trial']}
        for row in trial_plan()]}


class VoiceTrialsTests(unittest.TestCase):
    def test_fixed_human_protocol_has_fifteen_commands_ten_unknown_and_five_silences(self):
        rows=trial_plan();self.assertEqual(len(rows),30)
        self.assertEqual([r['trial'] for r in rows],list(range(1,31)))
        self.assertEqual([sum(r['expected']==label for r in rows) for label in range(7)],[3,3,3,3,3,10,5])
        self.assertTrue(all(type(r['prompt']) is str and r['prompt'] for r in rows))
        rows[0]['expected']=99;self.assertEqual(trial_plan()[0]['expected'],0)

    def test_boundary_is_twelve_of_fifteen_and_at_most_one_negative_activation(self):
        data=results();report=summarize(data)
        self.assertEqual((report['command_correct'],report['command_total'],report['negative_total']),(15,15,15))
        self.assertEqual(report['false_activations'],0);self.assertTrue(report['thresholds_met'])
        self.assertFalse(report['hardware_acceptance_proven'])
        for index in range(3):data['records'][index]['observed']=5
        data['records'][15]['observed']=0
        self.assertTrue(summarize(data)['thresholds_met'])
        data['records'][3]['observed']=5;self.assertFalse(summarize(data)['thresholds_met'])
        data=results();data['records'][15]['observed']=0;data['records'][16]['observed']=1
        self.assertFalse(summarize(data)['thresholds_met'])

    def test_an_inference_error_cannot_make_a_negative_trial_look_successful(self):
        data=results();data['records'][-1]['observed']=-1
        report=summarize(data);self.assertEqual(report['errors'],1);self.assertFalse(report['thresholds_met'])
        data=results();data['scope']='device_human'
        self.assertEqual(summarize(data)['reported_scope'],'device_human')
        self.assertFalse(summarize(data)['hardware_acceptance_proven'])

    def test_acoustic_synthetic_trials_are_distinct_from_human_voice(self):
        data=results();data['scope']='device_synthetic'
        report=summarize(data)
        self.assertEqual(report['reported_scope'],'device_synthetic')
        self.assertFalse(report['hardware_acceptance_proven'])

    def test_missing_duplicate_foreign_or_noncanonical_results_never_pass(self):
        changes=[lambda d:d['records'].pop(),lambda d:d['records'].append(dict(d['records'][0])),
                 lambda d:d['records'][0].update(trial=31),lambda d:d['records'][0].update(observed=True),
                 lambda d:d['records'][0].update(observed=7),lambda d:d['records'][0].update(latency_ms=math.nan),
                 lambda d:d['records'][0].update(latency_ms=10**1000),
                 lambda d:d['records'][0].update(latency_ms=-1),lambda d:d['records'][0].update(latency_ms=True),
                 lambda d:d['records'][0].update(audio='must-not-be-stored'),lambda d:d.update(scope='arbitrary'),
                 lambda d:d.update(mac='must-not-be-stored')]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                data=results();change(data)
                with self.assertRaises(ValueError):summarize(data)

    def test_cli_generates_a_plan_and_summarizes_bounded_results_without_echoing_bad_input(self):
        out=io.StringIO();self.assertEqual(main(['plan'],out=out),0)
        self.assertEqual(len(json.loads(out.getvalue())['trials']),30)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'results.json';path.write_text(json.dumps(results()))
            out=io.StringIO();self.assertEqual(main(['summarize',str(path)],out=out),0)
            self.assertEqual(json.loads(out.getvalue())['command_correct'],15)
            path.write_text('{do-not-echo-secret');out=io.StringIO()
            self.assertEqual(main(['summarize',str(path)],out=out),2)
            self.assertNotIn('do-not-echo',out.getvalue())

    def test_cli_refuses_a_fifo_without_waiting_for_its_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'results';os.mkfifo(path,0o600)
            proc=subprocess.run([sys.executable,'-m','arcade_ai_pet.voice_trials','summarize',str(path)],
                                capture_output=True,text=True,timeout=.5)
            self.assertEqual(proc.returncode,2)
