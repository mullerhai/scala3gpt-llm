package minimind

import torch.{Default, Tensor, nn, FloatNN}
import torch.nn.modules.{HasParams, TensorModule}

class RMSNorm[ParamType <: FloatNN : Default](dim: Int, eps: Double = 1e-5) extends HasParams[ParamType] with TensorModule[ParamType] {
  val epsValue = eps
  val weight = torch.ones(dim)

  register_parameter("weight", weight)

  private def _norm(x: Tensor[ParamType]): Tensor[ParamType] = {
    val xp = x.pow(2)
    val xpMean = xp.mean(-1, keepdim = true)
    val result = x * torch.rsqrt(xpMean + epsValue)
    result.to(x.dtype)
  }

  override def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    weight.to(x.dtype) * _norm(x) //.to(x.dtype)
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
