package minimind

import torch.{Default, Tensor, nn, FloatNN}
import torch.nn.modules.{HasParams, TensorModule}

class FeedForward[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
  val intermediateSize = config.intermediateSize.getOrElse {
    val tempSize = (config.hiddenSize * 8 / 3).toInt
    64 * ((tempSize + 64 - 1) / 64)
  }
  val gateProj = nn.Linear(config.hiddenSize, intermediateSize, bias = false)
  val downProj = nn.Linear(intermediateSize, config.hiddenSize, bias = false)
  val upProj = nn.Linear(config.hiddenSize, intermediateSize, bias = false)
  val dropout = nn.Dropout(config.dropout.toFloat)
  val actFn = ACT2FN[ParamType](config.hiddenAct)

  register_module("gateProj", gateProj)
  register_module("downProj", downProj)
  register_module("upProj", upProj)
  register_module("dropout", dropout)

  override def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    dropout(downProj(actFn(gateProj(x)) * upProj(x)))
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
