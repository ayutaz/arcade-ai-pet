"""料金不明/再試行/同時予約/回収失敗で予算を解放しない。ネット接続なし。"""
import copy,tempfile,threading,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from arcade_ai_pet.cloud_budget import BudgetLedger,estimate,default_ledger_path


def quote(**updates):
    return dict(observed_at=1000,running_usd_per_hour='1',stopped_usd_per_hour='0.1',
                download_usd_per_gb='.5',upload_usd_per_gb='.25',download_gb='2',upload_gb='1',
                recovery_seconds=360,retention_seconds=3600,**updates)


class CloudBudgetTests(unittest.TestCase):
    def test_exact_cost_includes_transfer_recovery_and_stopped_storage(self):
        r=estimate(quote(),now=1000,requested_seconds=3600,available_micro=2_000_000)
        self.assertEqual(r['job_seconds'],1980);self.assertEqual(r['reserved_micro'],2_000_000)
        self.assertEqual(r['execution_deadline'],2980);self.assertEqual(r['recovery_deadline'],3340)
        self.assertEqual(r['retention_deadline'],6940)
        with self.assertRaises(ValueError):estimate(quote(),now=1000,requested_seconds=3600,minimum_seconds=2000,available_micro=2_000_000)

    def test_unknown_nonfinite_or_stale_cost_is_rejected(self):
        for value in (None,True,'NaN','Infinity','-1','1e-1000000'):
            with self.subTest(value=value):
                q=quote();q['stopped_usd_per_hour']=value
                with self.assertRaises(ValueError):estimate(q,now=1000,requested_seconds=3600)
        for now in (999,1061):
            with self.assertRaises(ValueError):estimate(quote(),now=now,requested_seconds=3600)
        for n in (True,0,21601):
            with self.assertRaises(ValueError):estimate(quote(),now=1000,requested_seconds=n)

    def test_reopening_ledger_preserves_project_spend_and_same_run_cannot_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'budget.sqlite3';ledger=BudgetLedger(path)
            r=ledger.reserve('run-one',quote(),now=1000,requested_seconds=3600,billing_complete=True)
            self.assertEqual(ledger.snapshot()['reserved_micro'],r['reserved_micro'])
            ledger.mark_creating('run-one',now=1000)
            with self.assertRaises(ValueError):ledger.mark_creating('run-one',now=1000)
            ledger.bind_instance('run-one',123,ownership_verified=True)
            ledger.settle('run-one','21.50',destroyed=True,artifacts_verified=True,billing_complete=True)
            reopened=BudgetLedger(path);s=reopened.snapshot()
            self.assertEqual(s['project'],'arcade-ai-pet');self.assertEqual(s['spent_micro'],21_500_000)
            self.assertEqual(s['available_micro'],68_500_000)
            with self.assertRaises(ValueError):reopened.reserve('run-one',quote(),now=1000,requested_seconds=60,billing_complete=True)
            self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_unknown_billing_or_orphan_blocks_new_work_without_releasing_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp).resolve()/'budget.sqlite3')
            with self.assertRaises(ValueError):ledger.reserve('no-bill',quote(),now=1000,requested_seconds=60,billing_complete=False)
            before=ledger.snapshot();self.assertEqual(before['reserved_micro'],0)
            r=ledger.reserve('uncertain',quote(),now=1000,requested_seconds=60,billing_complete=True)
            ledger.mark_creating('uncertain',now=1000);ledger.mark_unknown('uncertain')
            with self.assertRaises(ValueError):ledger.reserve('retry',quote(),now=1000,requested_seconds=60,billing_complete=True)
            self.assertEqual(ledger.snapshot()['reserved_micro'],r['reserved_micro'])
            ledger.bind_instance('uncertain',456,ownership_verified=True)
            self.assertTrue(ledger.owns_instance('uncertain',456));self.assertFalse(ledger.owns_instance('uncertain',789))

    def test_failed_recovery_or_missing_destruction_never_settles_or_starts_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp).resolve()/'budget.sqlite3');r=ledger.reserve('run',quote(),now=1000,requested_seconds=60,billing_complete=True)
            ledger.mark_creating('run',now=1000);ledger.bind_instance('run',9,ownership_verified=True);ledger.recovery_failed('run')
            for fields in (dict(destroyed=False,artifacts_verified=True,billing_complete=True),dict(destroyed=True,artifacts_verified=False,billing_complete=True),dict(destroyed=True,artifacts_verified=True,billing_complete=False)):
                with self.assertRaises(ValueError):ledger.settle('run','0.10',**fields)
            self.assertEqual(ledger.snapshot()['reserved_micro'],r['reserved_micro'])
            with self.assertRaises(ValueError):ledger.reserve('next',quote(),now=1000,requested_seconds=60,billing_complete=True)

    def test_concurrent_reservations_share_one_budget_and_quote_deadline_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'budget.sqlite3';BudgetLedger(path);barrier=threading.Barrier(2)
            q=quote();q.update(running_usd_per_hour='60',stopped_usd_per_hour='0',download_gb='0',upload_gb='0',recovery_seconds=1,retention_seconds=0)
            def reserve(name):
                ledger=BudgetLedger(path);barrier.wait()
                try:ledger.reserve(name,q,now=1000,requested_seconds=3600,minimum_seconds=3600,billing_complete=True);return True
                except ValueError:return False
            with ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(reserve,('a','b')))
            self.assertEqual(sum(result),1);self.assertLessEqual(BudgetLedger(path).snapshot()['reserved_micro'],90_000_000)
            winner=('a','b')[result.index(True)];ledger=BudgetLedger(path)
            with self.assertRaises(ValueError):ledger.mark_creating(winner,now=1061)

    def test_cost_overrun_is_recorded_instead_of_erased_and_blocks_future_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp).resolve()/'budget.sqlite3');ledger.reserve('run',quote(),now=1000,requested_seconds=60,billing_complete=True)
            ledger.mark_creating('run',now=1000);ledger.bind_instance('run',3,ownership_verified=True)
            ledger.settle('run','100.01',destroyed=True,artifacts_verified=True,billing_complete=True)
            self.assertEqual(ledger.snapshot()['spent_micro'],100_010_000);self.assertTrue(ledger.snapshot()['budget_breached'])
            with self.assertRaises(ValueError):ledger.reserve('next',quote(),now=1000,requested_seconds=60,billing_complete=True)

    def test_only_never_attempted_reservations_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp).resolve()/'budget.sqlite3')
            ledger.reserve('unused',quote(),now=1000,requested_seconds=60,billing_complete=True)
            ledger.cancel_unused('unused');self.assertEqual(ledger.snapshot()['reserved_micro'],0)
            ledger.reserve('attempted',quote(),now=1000,requested_seconds=60,billing_complete=True)
            ledger.mark_creating('attempted',now=1000)
            with self.assertRaises(ValueError):ledger.cancel_unused('attempted')
            self.assertGreater(ledger.snapshot()['reserved_micro'],0)

    def test_shared_default_path_and_invalid_symlink_are_not_repo_scoped(self):
        self.assertEqual(default_ledger_path().name,'arcade-ai-pet.sqlite3');self.assertFalse(default_ledger_path().is_relative_to(Path.cwd()))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();target=root/'real';target.write_text('not a ledger');link=root/'link';link.symlink_to(target)
            with self.assertRaises(ValueError):BudgetLedger(link)


if __name__=='__main__':unittest.main()
