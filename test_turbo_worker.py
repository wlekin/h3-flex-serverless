import importlib.util
from pathlib import Path
import sys,types,unittest,tempfile
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('turbo_worker',Path(__file__).with_name('turbo_worker.py'))
turbo=importlib.util.module_from_spec(spec);spec.loader.exec_module(turbo)

class TurboTest(unittest.TestCase):
 def test_euler_matches_flow_update_and_terminal_clean_prediction(self):
  self.assertAlmostEqual(turbo.euler_update(2.,10.,1.,.75),4.)
  self.assertEqual(turbo.euler_update(4.,10.,.75,0.),10.)
  x=2.
  sigmas=[1.,.9,.75,.5,0.]
  for a,b in zip(sigmas,sigmas[1:]):x=turbo.euler_update(x,10.,a,b)
  self.assertEqual(x,10.)
 def test_comparison_routes_children_and_keeps_lora_context_separate(self):
  registered={};captured=[]
  base=types.SimpleNamespace(command=lambda r,w,o:['pika-h3-pro','--task',r['task']])
  def handler(job):
   with tempfile.TemporaryDirectory() as tmp:
    args=base.command(job['input'],Path(tmp),Path(tmp)/'out.mp4');captured.append(args)
   return {'metadata':{'task':job['input']['task']}}
  base.handler=handler
  with patch.dict(sys.modules,{'handler':base,'lora_worker':types.SimpleNamespace(validate_loras=lambda x:x),'runpod':types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda d:registered.update(d)))}):
   turbo.parent();h=registered['handler']
   out=h({'input':{'profile':'compare','task':'ref2va','references':[{'kind':'image','base64':'x'}]}})
   self.assertEqual(set(out),{'baseline','turbo'});self.assertEqual(captured[0][1],'/app/ref_worker.py');self.assertTrue(captured[1][1].endswith('turbo_worker.py'))
   self.assertIn('error',h({'input':{'task':'i2va'}}))

if __name__=='__main__':unittest.main()
