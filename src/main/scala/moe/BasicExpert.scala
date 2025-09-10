package moe

import torch.{Default, Tensor, nn, FloatNN}
import torch.nn.modules.{HasParams, TensorModule}

class BasicExpert[ParamType <: FloatNN: Default](featureIn: Int, featureOut: Int) extends HasParams[ParamType]
  with TensorModule[ParamType]  {
  val linear = register(nn.Linear(featureIn, featureOut))
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    linear.forward(x)
  }
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    linear.forward(x)
  }
}