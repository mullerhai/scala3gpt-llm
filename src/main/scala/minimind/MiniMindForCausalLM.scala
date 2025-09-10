package minimind

import torch.{Default, Tensor, nn, FloatNN, ::}
import torch.nn.modules.{HasParams, TensorModule}

class MiniMindForCausalLM[ParamType <: FloatNN: Default](config: Option[MiniMindConfig] = None) extends HasParams[ParamType] with TensorModule[ParamType] {
  val configValue = config.getOrElse(new MiniMindConfig())
  val model = new MiniMindModel[ParamType](configValue)
  val lmHead = nn.Linear(configValue.hiddenSize, configValue.vocabSize, bias = false)

  register_module("model", model)
  register_module("lmHead", lmHead)

  // 共享权重
  model.embedTokens.weight = lmHead.weight

  override def forward(inputIds: Tensor[ParamType], 
                       attentionMask: Option[Tensor[ParamType]] = None, 
                       pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, 
                       useCache: Boolean = false, 
                       logitsToKeep: Any = 0, 
                       args: Map[String, Any] = Map.empty): CausalLMOutputWithPast[ParamType] = {
    val (h, pastKvs, auxLoss) = model(inputIds, attentionMask, pastKeyValues, useCache, args)
    val sliceIndices:Seq[Long] = logitsToKeep match {
      case i: Int => if (i > 0) -i until h.shape(1) else 0 until h.shape(1)
//      case t: Tensor[ParamType] => ??? // 需根据具体情况实现
    }
    val logits = lmHead(h(::,sliceIndices, ::))
    CausalLMOutputWithPast[ParamType](
      lastHiddenState = h,
      logits = logits,
      auxLoss = auxLoss,
      pastKeyValues = pastKvs
    )
  }

  def apply(inputIds: Tensor[ParamType], attentionMask: Option[Tensor[ParamType]] = None, pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, useCache: Boolean = false, logitsToKeep: Any = 0, args: Map[String, Any] = Map.empty): CausalLMOutputWithPast[ParamType] =
    forward(inputIds, attentionMask, pastKeyValues, useCache, logitsToKeep, args)
}

