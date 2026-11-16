"""Build only the dedicated AI Pet package; never access a device or firmware."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SDK_COMMIT='3bb9548846505eac303927b9064c6a8d0eb44f79'


def load_sdk(path):
    path=Path(path).absolute()
    try:
        if path.resolve()!=path:raise ValueError()
        head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        changes=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True)
        if head!=SDK_COMMIT or changes.strip():raise ValueError()
    except (OSError,ValueError,subprocess.CalledProcessError):raise ValueError('clean pinned voice-capable SDK required') from None
    sys.path.insert(0,str(path/'prototypes/lab-os'))
    sdk=importlib.import_module('labos.project')
    if sdk.ROOT!=path:raise ValueError('another SDK already loaded')
    return sdk


def build_pet(sdk_root,output):
    output=Path(output).absolute()
    if output.resolve()!=output or output.exists():raise ValueError('fresh real output required')
    sdk=load_sdk(sdk_root)
    output.mkdir(parents=True)
    for name in ('app.c','app.json'):shutil.copyfile(ROOT/'app'/name,output/name)
    shutil.copyfile(ROOT/'native/behavior.c',output/'behavior.c')
    shutil.copyfile(ROOT/'models/behavior-v1/behavior_model.h',output/'behavior_model.h')
    result=sdk.build_project(output)
    package=Path(result['package']).read_bytes()
    return dict(result,package_sha256=hashlib.sha256(package).hexdigest(),sdk_commit=SDK_COMMIT,device_tested=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sdk',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(build_pet(args.sdk,args.output)))
