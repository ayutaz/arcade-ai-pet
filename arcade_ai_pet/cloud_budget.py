"""Project-wide quote/reservation ledger only; no provider API or paid operation."""
from contextlib import contextmanager
from decimal import Decimal,InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat

PROJECT='arcade-ai-pet'
CAP_MICRO=100_000_000
RESERVE_MICRO=10_000_000
NORMAL_MICRO=CAP_MICRO-RESERVE_MICRO
FIELDS={'observed_at','running_usd_per_hour','stopped_usd_per_hour','download_usd_per_gb','upload_usd_per_gb',
        'download_gb','upload_gb','recovery_seconds','retention_seconds'}


def default_ledger_path():
    return Path.home()/'.local/state/scratch-arcade-lab/vast/arcade-ai-pet.sqlite3'


def _number(value):
    if type(value) not in (str,int,float,Decimal) or len(str(value))>80:
        raise ValueError('explicit finite nonnegative price required')
    try:
        value=Decimal(str(value))
        if not value.is_finite() or value<0 or value>1_000_000 or value!=0 and value.adjusted() < -18:
            raise ValueError('bounded finite nonnegative price required')
    except InvalidOperation:
        raise ValueError('explicit finite price required') from None
    return Fraction(value)


def _int(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError('bounded integer required')
    return value


def _ceil(value):
    return (value.numerator+value.denominator-1)//value.denominator


def estimate(quote, *, now, requested_seconds, minimum_seconds=1, available_micro=NORMAL_MICRO):
    if type(quote) is not dict or set(quote)!=FIELDS:
        raise ValueError('complete explicit cost quotation required')
    _int(now,0,2**53)
    observed=_int(quote['observed_at'],0,2**53)
    if not 0<=now-observed<=60:
        raise ValueError('fresh quotation required')
    _int(requested_seconds,1,21600)
    _int(minimum_seconds,1,requested_seconds)
    _int(available_micro,0,NORMAL_MICRO)
    recovery=_int(quote['recovery_seconds'],1,3600)
    retention=_int(quote['retention_seconds'],0,86400)
    rate=_number(quote['running_usd_per_hour'])
    if rate<=0:
        raise ValueError('positive complete running rate required')
    fixed=(rate*recovery/3600 + _number(quote['stopped_usd_per_hour'])*retention/3600 +
           _number(quote['download_usd_per_gb'])*_number(quote['download_gb']) +
           _number(quote['upload_usd_per_gb'])*_number(quote['upload_gb']))
    allowance=(Fraction(available_micro,1_000_000)-fixed)*3600/rate
    seconds=min(requested_seconds,allowance.numerator//allowance.denominator)
    if seconds<minimum_seconds:
        raise ValueError('remaining project budget cannot cover execution and recovery')
    reserved=_ceil((fixed+rate*seconds/3600)*1_000_000)
    if reserved>available_micro:
        raise ValueError('quotation exceeds reservation allowance')
    return dict(job_seconds=seconds,reserved_micro=reserved,quote_expires=observed+60,
                execution_deadline=now+seconds,recovery_deadline=now+seconds+recovery,
                retention_deadline=now+seconds+recovery+retention)


class BudgetLedger:
    def __init__(self,path=None):
        self.path=Path(path if path is not None else default_ledger_path()).absolute()
        if self.path.resolve()!=self.path:
            raise ValueError('real shared ledger path required')
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        try:
            fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            os.close(fd)
        except FileExistsError:
            pass
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY CHECK(id=1), project TEXT NOT NULL, cap INTEGER NOT NULL, margin INTEGER NOT NULL, spent INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, state TEXT NOT NULL, reserved INTEGER NOT NULL, created INTEGER NOT NULL, quote_expires INTEGER NOT NULL, execution_deadline INTEGER NOT NULL, recovery_deadline INTEGER NOT NULL, retention_deadline INTEGER NOT NULL, quote_sha TEXT NOT NULL, instance_id INTEGER UNIQUE, actual INTEGER)')
            db.execute('INSERT OR IGNORE INTO meta VALUES (1,?,?,?,0)',(PROJECT,CAP_MICRO,RESERVE_MICRO))
            row=db.execute('SELECT project,cap,margin FROM meta WHERE id=1').fetchone()
            if tuple(row)!=(PROJECT,CAP_MICRO,RESERVE_MICRO):
                raise ValueError('existing project ledger identity changed')
            self._summary(db)

    @contextmanager
    def _transaction(self):
        info=self.path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o777!=0o600 or
                self.path.resolve()!=self.path):
            raise ValueError('private owned regular ledger required')
        db=sqlite3.connect(self.path,timeout=5,isolation_level=None)
        db.row_factory=sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except sqlite3.Error:
            db.rollback()
            raise ValueError('ledger unavailable or invalid; do not reset it') from None
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _summary(db):
        spent=db.execute('SELECT spent FROM meta WHERE id=1').fetchone()[0]
        actual=db.execute("SELECT COALESCE(SUM(actual),0) FROM runs WHERE state='settled'").fetchone()[0]
        if spent!=actual or spent<0:
            raise ValueError('ledger spend identity changed')
        reserved=db.execute("SELECT COALESCE(SUM(reserved),0) FROM runs WHERE state NOT IN ('settled','cancelled')").fetchone()[0]
        uncertain=bool(db.execute("SELECT 1 FROM runs WHERE state IN ('creating','unknown','recovering') LIMIT 1").fetchone())
        return dict(project=PROJECT,spent_micro=spent,reserved_micro=reserved,available_micro=max(0,NORMAL_MICRO-spent-reserved),
                    budget_breached=spent+reserved>CAP_MICRO,uncertain=uncertain)

    @staticmethod
    def _row(db,run_id):
        if type(run_id) is not str or not re.fullmatch('[a-z0-9][a-z0-9-]{0,63}',run_id):
            raise ValueError('bounded unique run identity required')
        row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        if row is None:
            raise ValueError('unknown owned run')
        return row

    def snapshot(self):
        with self._transaction() as db:
            return self._summary(db)

    def reserve(self,run_id,quote,*,now,requested_seconds,minimum_seconds=1,billing_complete):
        if billing_complete is not True or type(run_id) is not str or not re.fullmatch('[a-z0-9][a-z0-9-]{0,63}',run_id):
            raise ValueError('reconciled billing and unique run identity required')
        with self._transaction() as db:
            state=self._summary(db)
            if state['uncertain'] or db.execute('SELECT 1 FROM runs WHERE id=?',(run_id,)).fetchone():
                raise ValueError('unresolved prior run or reused identity; no new reservation')
            result=estimate(quote,now=now,requested_seconds=requested_seconds,minimum_seconds=minimum_seconds,available_micro=state['available_micro'])
            digest=hashlib.sha256(json.dumps(quote,sort_keys=True,default=str).encode()).hexdigest()
            db.execute('INSERT INTO runs (id,state,reserved,created,quote_expires,execution_deadline,recovery_deadline,retention_deadline,quote_sha) VALUES (?,?,?,?,?,?,?,?,?)',
                       (run_id,'reserved',result['reserved_micro'],now,result['quote_expires'],result['execution_deadline'],result['recovery_deadline'],result['retention_deadline'],digest))
            return dict(result,run_id=run_id,project=PROJECT)

    def mark_creating(self,run_id,*,now):
        _int(now,0,2**53)
        with self._transaction() as db:
            row=self._row(db,run_id)
            if row['state']!='reserved' or not row['created']<=now<=min(row['quote_expires'],row['execution_deadline']):
                raise ValueError('fresh never-attempted reservation required; do not retry creation')
            db.execute("UPDATE runs SET state='creating' WHERE id=?",(run_id,))

    def mark_unknown(self,run_id):
        with self._transaction() as db:
            row=self._row(db,run_id)
            if row['state'] not in ('creating','running','recovering','unknown'):
                raise ValueError('attempted run required')
            db.execute("UPDATE runs SET state='unknown' WHERE id=?",(run_id,))

    def bind_instance(self,run_id,instance_id,*,ownership_verified):
        if ownership_verified is not True:
            raise ValueError('provider ownership verification required')
        _int(instance_id,1,2**53)
        with self._transaction() as db:
            row=self._row(db,run_id)
            if row['state'] not in ('creating','unknown') or row['instance_id'] not in (None,instance_id):
                raise ValueError('single verified instance binding required')
            db.execute("UPDATE runs SET state='running',instance_id=? WHERE id=?",(instance_id,run_id))

    def owns_instance(self,run_id,instance_id):
        if type(instance_id) is not int or instance_id<=0:
            return False
        with self._transaction() as db:
            row=self._row(db,run_id)
            return row['instance_id']==instance_id and row['state'] in ('running','recovering','unknown')

    def recovery_failed(self,run_id):
        with self._transaction() as db:
            row=self._row(db,run_id)
            if row['state'] not in ('running','recovering'):
                raise ValueError('owned running instance required')
            db.execute("UPDATE runs SET state='recovering' WHERE id=?",(run_id,))

    def cancel_unused(self,run_id):
        with self._transaction() as db:
            row=self._row(db,run_id)
            if row['state']!='reserved':
                raise ValueError('creation was attempted; do not release uncertain cost')
            db.execute("UPDATE runs SET state='cancelled' WHERE id=?",(run_id,))

    def settle(self,run_id,actual_usd,*,destroyed,artifacts_verified,billing_complete):
        if not all(value is True for value in (destroyed,artifacts_verified,billing_complete)):
            raise ValueError('verified artifacts, instance destruction and final billing required')
        actual=_ceil(_number(actual_usd)*1_000_000)
        with self._transaction() as db:
            row=self._row(db,run_id)
            if row['state'] not in ('running','recovering','unknown') or row['instance_id'] is None:
                raise ValueError('single unsettled owned instance required')
            # Record a real overrun rather than hiding it by refusing accounting.
            db.execute("UPDATE runs SET state='settled',actual=? WHERE id=?",(actual,run_id))
            db.execute('UPDATE meta SET spent=spent+? WHERE id=1',(actual,))
            return self._summary(db)
