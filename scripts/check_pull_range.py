"""Stored-byte checks of partial download recovery; never contacts a remote service."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'.agents/skills/kaggle-training-notebook/scripts/pull_kernel_output.py'
spec=importlib.util.spec_from_file_location('pull_helper',path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class Response:
    def __init__(self,status,body,length=6,span=None):
        self.status_code=status;self.body=body;self.headers={'Content-Length':str(length)}
        if span:self.headers['Content-Range']=span
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def close(self):pass
    def raise_for_status(self):assert self.status_code<400
    def iter_content(self,size):yield self.body


class Session:
    def __init__(self,*responses):self.responses=list(responses);self.headers=[]
    def get(self,url,**kwargs):
        self.headers.append(kwargs.get('headers',{}));return self.responses.pop(0)


def main():
    base=ROOT/'papers/full-validation/smoke/pull-range-20261013'
    assert not base.exists();base.mkdir(parents=True)
    for name,response,initial,expected in (
        ('new',Response(200,b'abcdef'),None,6),
        ('append',Response(206,b'def',3,'bytes 3-5/6'),b'abc',3),
        ('ignored',Response(200,b'abcdef'),b'abc',6)):
        target=base/name
        if initial:target.with_name(name+'.part').write_bytes(initial)
        session=Session(response)
        assert helper.fetch(session,'private-signed-url',target,10)==('ok',expected)
        assert target.read_bytes()==b'abcdef'
        if initial:assert session.headers==[{'Range':'bytes=3-'}]
    target=base/'stale';target.with_name('stale.part').write_bytes(b'abcdef')
    session=Session(Response(416,b''),Response(200,b'abcdef'))
    assert helper.fetch(session,'private-signed-url',target,10)==('ok',6)
    assert target.read_bytes()==b'abcdef' and session.headers==[{'Range':'bytes=6-'},{}]
    for name,response in (
        ('wrong-range',Response(206,b'def',3,'bytes 2-5/6')),
        ('short',Response(200,b'abc'))):
        target=base/name;partial=target.with_name(name+'.part');partial.write_bytes(b'abc')
        try:helper.fetch(Session(response),'private-signed-url',target,10)
        except IOError:pass
        else:raise AssertionError('invalid partial accepted')
        assert partial.read_bytes()==b'abc' and not target.exists()
    assert (ROOT/'.claude/skills/kaggle-training-notebook/scripts/pull_kernel_output.py').read_bytes()==path.read_bytes()
    print('PASS new/206 append/ignored Range/416/invalid Range/short transfer; mirrors identical')


if __name__=='__main__':main()
