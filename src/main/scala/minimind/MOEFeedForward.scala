package minimind

import torch.nn.modules.container.ModuleList
import torch.{::, DType, Default, Float16, FloatNN, Int64, Tensor, nn}
import torch.nn.modules.{HasParams, TensorModule}

class MOEFeedForward[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
  val configValue = config
  val experts = nn.ModuleList((0 until config.nRoutedExperts).map(num => new FeedForward[ParamType](config))*)
  val gate = new MoEGate[ParamType](config)
  val sharedExperts = if (config.nSharedExperts > 0) Some(ModuleList((0 until config.nSharedExperts).map(num => new FeedForward[ParamType](config))*)) else None
  var auxLoss = torch.tensor(0.0f)

  register_module("gate", gate)
//  registerModuleList("experts", experts)
//  sharedExperts.foreach(registerModuleList("sharedExperts", _))

  override def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    val identity = x
    val origShape = x.shape
    val (bsz, seqLen, _) = x.shape
    val (topkIdx, topkWeight, auxLossValue) = gate(x)
    val xFlattened = x.view(-1, x.shape.last)
    val flatTopkIdx = topkIdx.view(-1)
    auxLoss = auxLossValue.to(x.dtype)

    val y = if (isTraining) {
      val repeatedX = xFlattened.repeat_interleave(config.numExpertsPerTok, dim = 0)
      val yTensor = torch.emptyLike(repeatedX, dtype = DType.float32)
      experts.zipWithIndex.foreach { case (expert, i) =>
        val mask = flatTopkIdx == i
        yTensor(mask) = expert(repeatedX(mask)).to(yTensor.dtype)
      }
      (yTensor.view(topkWeight.shape ++ Seq(-1)) * topkWeight.unsqueeze(-1)).sum(dim = 1).view(origShape)
    } else {
      moeInfer(xFlattened, flatTopkIdx, topkWeight.view(-1, 1)).view(origShape*)
    }

    sharedExperts.foreach { experts =>
      experts.foreach { expert =>
        y += expert(identity)
      }
    }

    y
  }

  private def moeInfer(x: Tensor[ParamType], flatExpertIndices: Tensor[Int64], flatExpertWeights: Tensor[?]): Tensor[ParamType] = {
    torch.no_grad {
      val expertCache = torch.zerosLike(x)
      val idxs = flatExpertIndices.argsort()
      val tokensPerExpert = flatExpertIndices.bincount().cpu().numpy() //.cumsum(0)
      val tokenIdxs = idxs / configValue.numExpertsPerTok

      tokensPerExpert.getArray.zipWithIndex.foreach { case (endIdx, i) =>
        val startIdx = if (i == 0) 0 else tokensPerExpert(i - 1)
        if (startIdx != endIdx) {
          val expert = experts(i)
          val expTokenIdx = tokenIdxs(startIdx.::(endIdx) )
          val expertTokens = x(expTokenIdx)
          val expertOut = expert(expertTokens).to(expertCache.dtype)
          expertOut.mul(flatExpertWeights(idxs(startIdx.::(endIdx) )))
          expertCache.scatter_add(0, expTokenIdx.view(-1, 1).repeat(1, x.shape.last), expertOut)
        }
      }

      expertCache
    }
  }

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}
