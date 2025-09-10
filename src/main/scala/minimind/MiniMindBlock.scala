package minimind

import torch.{Default, Tensor, nn, FloatNN}
import torch.nn.modules.{HasParams, TensorModule}

class MiniMindBlock[ParamType <: FloatNN: Default](layerId: Int, config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
  val numAttentionHeads = config.numAttentionHeads
  val hiddenSize = config.hiddenSize
  val headDim = config.hiddenSize / config.numAttentionHeads
  val selfAttn = new Attention[ParamType](config)
  val layerIdValue = layerId
  val inputLayernorm = new RMSNorm[ParamType](config.hiddenSize, eps = config.rmsNormEps)
  val postAttentionLayernorm = new RMSNorm[ParamType](config.hiddenSize, eps = config.rmsNormEps)
  val mlp = if (!config.useMoe) new FeedForward[ParamType](config) else new MOEFeedForward[ParamType](config)

  register_module("selfAttn", selfAttn)
  register_module("inputLayernorm", inputLayernorm)
  register_module("postAttentionLayernorm", postAttentionLayernorm)
  register_module("mlp", mlp)

  override def forward(hiddenStates: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) = {
    val residual = hiddenStates
    val (attnOutput, presentKeyValue) = selfAttn(inputLayernorm(hiddenStates), positionEmbeddings, pastKeyValue, useCache, attentionMask)
    val attnOutputWithResidual = attnOutput + residual
    val finalOutput = attnOutputWithResidual + mlp(postAttentionLayernorm(attnOutputWithResidual))
    (finalOutput, presentKeyValue)
  }

  def apply(hiddenStates: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) =
    forward(hiddenStates, positionEmbeddings, pastKeyValue, useCache, attentionMask)
}
