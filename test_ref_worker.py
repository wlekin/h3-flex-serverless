import importlib.util
from pathlib import Path
import sys,types,unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('ref_worker',Path(__file__).with_name('ref_worker.py'));ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
class ReferenceWorkerTest(unittest.TestCase):
 def setUp(self):
  self.registered={};self.base=types.SimpleNamespace(command=lambda r,w,o:['pika-h3-pro','--task',r['task']],handler=lambda j:j)
  self.slim=types.SimpleNamespace(validate_loras=lambda v:v)
  self.runpod=types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda d:self.registered.update(d)))
  self.modules=patch.dict(sys.modules,{'handler':self.base,'runpod':self.runpod,'lora_worker':self.slim});self.modules.start();ref.parent()
 def tearDown(self):self.modules.stop()
 def test_ref_routes_separate_adapter_and_keeps_regular_path(self):
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   for task,adapter in [('ref2va','ref2va'),('i2va','fl2va')]:
    command=self.base.command({'task':task},Path(d),Path(d)/'out.mp4');self.assertIn(str(ref.ROOT/'native-assets/adapter'/adapter),command)
 def test_reference_loras_are_forwarded_without_leaking_between_jobs(self):
  import tempfile,json
  captured=[]
  def generate(job):
   with tempfile.TemporaryDirectory() as d:
    argv=self.base.command({'task':job['input']['task']},Path(d),Path(d)/'out.mp4')
    captured.append(json.loads(Path(argv[3]).read_text()))
   return job
  self.base.handler=generate;h=self.registered['handler']
  spec={'name':'compatible.safetensors','strength':0.5}
  result=h({'input':{'task':'ref2va','loras':[spec],'references':[{'kind':'image','base64':'test'}]}})
  self.assertEqual(captured[-1],[spec]);self.assertNotIn('loras',result['input']);self.assertFalse(result['input']['cpu_offload'])
  h({'input':{'task':'ref2va','references':[{'kind':'image','base64':'test'}]}});self.assertEqual(captured[-1],[])
 def test_other_tasks_and_offload_rejected(self):
  for data in [{'task':'l2va'},{'task':'ref2va','cpu_offload':True}]:self.assertIn('error',self.registered['handler']({'input':data}))
if __name__=='__main__':unittest.main()
