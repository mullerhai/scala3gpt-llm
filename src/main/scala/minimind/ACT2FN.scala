package minimind

import torch.{Default, Tensor, nn, FloatNN}
import torch.nn.modules.{HasParams, TensorModule}

// 激活函数映射
object ACT2FN {
  def apply[ParamType <: FloatNN: Default](actName: String): Tensor[ParamType] => Tensor[ParamType] = actName match {
    case "silu" => torch.nn.functional.silu[ParamType]
    case _ => throw new NotImplementedError(s"Unsupported activation function: $actName")
  }
}
