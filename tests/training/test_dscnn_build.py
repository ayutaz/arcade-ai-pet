"""実Torchモデルの寸法とBN fold/exportを確認する。認識精度の試験ではない。"""
import unittest
import numpy as np
import torch
from arcade_ai_pet import dscnn_training as trainer
from arcade_ai_pet.dscnn_model import weights_bytes
from arcade_ai_pet.dscnn_quantize import fold_batch_norm


class DscnnBuildTests(unittest.TestCase):
    def test_real_model_parameter_count_and_intermediate_dimensions(self):
        torch.manual_seed(42);net=trainer.build_model().eval();values=torch.zeros((2,2113))
        with torch.no_grad():
            output=net(values);image=values[:,:2048].reshape(-1,1,64,32);shapes=[]
            for block in net.blocks:image=block(image);shapes.append(tuple(image.shape[1:]))
        self.assertEqual(tuple(output.shape),(2,7));self.assertEqual(shapes,[(8,32,16),(8,32,16),(16,32,16),(16,16,8),(24,16,8)])
        self.assertEqual(sum(p.numel() for p in net.parameters()),22910)
        self.assertEqual(net.head.in_features,3137)

    def test_export_folds_real_bn_and_preserves_all_model_state(self):
        torch.manual_seed(16);net=trainer.build_model().eval()
        with torch.no_grad():
            for block in net.blocks:
                bn=block[1];bn.running_mean.copy_(torch.linspace(-.4,.4,bn.num_features));bn.running_var.copy_(torch.linspace(.1,2,bn.num_features));bn.weight.copy_(torch.linspace(.7,1.3,bn.num_features));bn.bias.copy_(torch.linspace(-.2,.2,bn.num_features))
        before={k:v.clone() for k,v in net.state_dict().items()};layers=trainer.quantize(net)
        self.assertEqual(weights_bytes(layers)+20,23411)
        for key,value in net.state_dict().items():self.assertTrue(torch.equal(value,before[key]))
        for block in net.blocks:
            conv,bn=block[0],block[1];w,b=fold_batch_norm(conv.weight.detach().numpy(),np.zeros(conv.out_channels),bn.running_mean.numpy(),bn.running_var.numpy(),bn.weight.detach().numpy(),bn.bias.detach().numpy(),eps=bn.eps)
            values=torch.randn((2,conv.in_channels,8,10))
            with torch.no_grad():
                original=bn(conv(values));folded=torch.nn.functional.conv2d(values,torch.tensor(w,dtype=torch.float32),torch.tensor(b,dtype=torch.float32),stride=conv.stride,padding=conv.padding,groups=conv.groups)
            torch.testing.assert_close(original,folded,rtol=1e-5,atol=1e-5)

    def test_export_requires_evaluation_mode(self):
        with self.assertRaises(ValueError):trainer.quantize(trainer.build_model())


if __name__=='__main__':unittest.main()
