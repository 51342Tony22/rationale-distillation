"""Small actual MLX kernel check on CPU; never loads a model or uses the GPU."""
import unittest
from types import SimpleNamespace
import numpy as np

from static_selection import MLXScorer


class StaticMLXCPUTests(unittest.TestCase):
    def test_chunk_alignment_clipping_and_ties_against_numpy_reference(self):
        import mlx.core as mx
        previous_device=mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            logits=np.zeros((4,105),dtype=np.float32)
            logits[0]=np.linspace(-1,1,105)
            logits[1]=np.arange(105,dtype=np.float32)/10
            logits[2]=0.;logits[2,1]=2.;logits[2,2]=2.  # strict-greater ties
            logits[3]=np.linspace(3,-3,105)
            tape=mx.array(logits[None])
            model=SimpleNamespace(model=lambda inputs:tape[:,:inputs.shape[1]],
                                  lm_head=lambda states:states,
                                  args=SimpleNamespace(tie_word_embeddings=False),eval=lambda:None)
            target=[0,2,1]
            array=logits[1:]
            maximum=array.max(axis=-1)
            chosen=array[np.arange(3),target]
            expected=maximum+np.log(np.exp(array-maximum[:,None]).sum(axis=-1))-chosen
            expected_ranks=np.minimum(1+np.sum(array>chosen[:,None],axis=-1),100).tolist()
            for chunk in [1,2,5]:
                actual=MLXScorer(model,chunk_tokens=chunk).statistics([3,4],target,include_ranks=True)
                self.assertTrue(np.allclose(actual['surprisals'],expected,atol=1e-5))
                self.assertEqual(actual['ranks'],expected_ranks)
                self.assertEqual(actual['ranks'][0],100)
                self.assertEqual(actual['ranks'][1],1)
        finally:
            mx.set_default_device(previous_device)


if __name__=='__main__':unittest.main()
