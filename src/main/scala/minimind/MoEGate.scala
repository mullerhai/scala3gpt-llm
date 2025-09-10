package minimind

import torch.{Default, Float32, Float64, FloatNN, Int64, Tensor, nn}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.functional as F

class MoEGate[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
  val topK = config.numExpertsPerTok
  val nRoutedExperts = config.nRoutedExperts
  val scoringFunc = config.scoringFunc
  val alpha = config.auxLossAlpha
  val seqAux = config.seqAux
  val normTopkProb = config.normTopkProb
  val gatingDim = config.hiddenSize
  val weight = torch.empty(Array(nRoutedExperts, gatingDim))
  

  register_parameter("weight", weight)

  resetParameters()

  private def resetParameters(): Unit = {
    torch.nn.init.kaiming_normal_(weight, a= math.sqrt(5d) )
//    torch.kaiming_uniform_(weight, a= math.sqrt(5d) )
    
//    KaimingUniform(weight, a = math.sqrt(5).toParamType[ParamType])
  }

  override def forward(hiddenStates: Tensor[ParamType]): (Tensor[Int64], Tensor[?], Tensor[?]) = {
//    val (bsz, seqLen, h) = hiddenStates.shape
    val bsz = hiddenStates.shape(0)
    val seqLen = hiddenStates.shape(1)
    val h = hiddenStates.shape(2)
    val hiddenStatesFlattened = hiddenStates.view(-1, h)
    val logits = torch.nn.functional.linear(hiddenStatesFlattened, weight.to(hiddenStatesFlattened.dtype))
    val scores = scoringFunc match {
      case "softmax" => torch.softmax(logits, dim = -1)
      case _ => throw new NotImplementedError(s"unsupportable scoring function for MoE gating: $scoringFunc")
    }

    val topkWeight_topkIdx = torch.topk(scores, k = topK, dim = -1, sorted = false)
    val topkWeight = topkWeight_topkIdx._1
    val topkIdx = topkWeight_topkIdx._2

    val normalizedTopkWeight = if (topK > 1 && normTopkProb) {
      val denominator = topkWeight.sum(dim = -1, keepdim = true) + 1e-20
      topkWeight / denominator
    } else {
      topkWeight
    }

    val auxLoss = if (isTraining && alpha > 0.0) {
      val scoresForAux = scores
      val auxTopk = topK
      val topkIdxForAuxLoss = topkIdx.view(bsz, -1)
      if (seqAux) {
        val scoresForSeqAux = scoresForAux.view(bsz, seqLen, -1)
        val ce = torch.zeros(Array(bsz, nRoutedExperts), device = hiddenStates.device)
        ce.scatter_add(1, topkIdxForAuxLoss, torch.ones(Array(bsz, seqLen * auxTopk), device = hiddenStates.device)).div((seqLen * auxTopk / nRoutedExperts) )
        (ce * scoresForSeqAux.mean(dim = 1)).sum(dim = 1).mean() * alpha
      } else {
        val maskCe = torch.nn.functional.oneHot(topkIdxForAuxLoss.view(-1), numClasses = nRoutedExperts)
        val ce = maskCe.mean(0)
        val Pi = scoresForAux.mean(0)
        val fi = ce * nRoutedExperts
        (Pi * fi).sum() * alpha
      }
    } else {
      torch.tensor(0.0, device = hiddenStates.device)
    }

    (topkIdx, normalizedTopkWeight, auxLoss)
  }

  def apply(hiddenStates: Tensor[ParamType]): (Tensor[Int64], Tensor[?], Tensor[?]) = forward(hiddenStates)
}
