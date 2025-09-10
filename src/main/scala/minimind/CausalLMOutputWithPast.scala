package minimind

import torch.{Default, Tensor, FloatNN}

// 因果语言模型输出类
case class CausalLMOutputWithPast[ParamType <: FloatNN: Default](
                                                                  lastHiddenState: Tensor[ParamType],
                                                                  logits: Tensor[ParamType],
                                                                  auxLoss: Tensor[ParamType],
                                                                  pastKeyValues: List[(Tensor[ParamType], Tensor[ParamType])]
                                                                )