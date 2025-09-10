package minimind

import torch.{Default, Tensor, nn, FloatNN, ::}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.functional as F

class Attention[ParamType <: FloatNN: Default](args: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
  val numKeyValueHeads = if (args.numKeyValueHeads == 0) args.numAttentionHeads else args.numKeyValueHeads
  assert(args.numAttentionHeads % numKeyValueHeads == 0)
  val nLocalHeads = args.numAttentionHeads
  val nLocalKvHeads = numKeyValueHeads
  val nRep = nLocalHeads / nLocalKvHeads
  val headDim = args.hiddenSize / args.numAttentionHeads
  val qProj = nn.Linear(args.hiddenSize, args.numAttentionHeads * headDim, bias = false)
  val kProj = nn.Linear(args.hiddenSize, numKeyValueHeads * headDim, bias = false)
  val vProj = nn.Linear(args.hiddenSize, numKeyValueHeads * headDim, bias = false)
  val oProj = nn.Linear(args.numAttentionHeads * headDim, args.hiddenSize, bias = false)
  val attnDropout = nn.Dropout(args.dropout.toFloat)
  val residDropout = nn.Dropout(args.dropout.toFloat)
  val dropoutValue = args.dropout
  val flash = true //torch.nn.functional.hasScaledDotProductAttention && args.flashAttn

  register_module("qProj", qProj)
  register_module("kProj", kProj)
  register_module("vProj", vProj)
  register_module("oProj", oProj)
  register_module("attnDropout", attnDropout)
  register_module("residDropout", residDropout)

  override def forward(x: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) = {
//    val (bsz, seqLen, _) = x.shape
    val bsz = x.shape(0)
    val seqLen = x.shape(1)
    val xq = qProj(x)
    val xk = kProj(x)
    val xv = vProj(x)
    val xqReshaped = xq.view(bsz, seqLen, nLocalHeads, headDim)
    val xkReshaped = xk.view(bsz, seqLen, nLocalKvHeads, headDim)
    val xvReshaped = xv.view(bsz, seqLen, nLocalKvHeads, headDim)

    val (cos, sin) = positionEmbeddings
    val (qEmbed, kEmbed) = applyRotaryPosEmb(xqReshaped, xkReshaped, cos(0.::(seqLen)), sin(0.::(seqLen)))

    var xkFinal = xkReshaped
    var xvFinal = xvReshaped
    if (pastKeyValue.isDefined) {
      xkFinal = torch.cat(Seq(pastKeyValue.get._1, xkFinal), dim = 1)
      xvFinal = torch.cat(Seq(pastKeyValue.get._2, xvFinal), dim = 1)
    }
    val pastKv = if (useCache) Some((xkFinal, xvFinal)) else None

    val xqTransposed = xqReshaped.transpose(1, 2)
    val xkTransposed = repeatKv(xkFinal, nRep).transpose(1, 2)
    val xvTransposed = repeatKv(xvFinal, nRep).transpose(1, 2)

    val output = if (flash && seqLen != 1) {
      val dropoutP = if (isTraining) dropoutValue else 0.0
      val attnMaskProcessed = attentionMask.map { mask =>
        mask.view(bsz, 1, 1, -1).expand(bsz, nLocalHeads, seqLen, -1).bools()
      }
      F.scaled_dot_product_attention(xqTransposed, xkTransposed, xvTransposed, attn_mask = attnMaskProcessed.to(x.dtype), dropout_p = dropoutP, is_causal = true)
    } else {
      val scores = (xqTransposed @@ xkTransposed.transpose(-2, -1)) / math.sqrt(headDim) //.toParamType[ParamType]
      val causalMask = torch.triu(torch.full(Seq(seqLen, seqLen), Double.NegativeInfinity, device = scores.device), diagonal = 1).unsqueeze(0).unsqueeze(0)
      val scoresWithMask = scores + causalMask

      attentionMask.foreach { mask =>
        val extendedMask = mask.unsqueeze(1).unsqueeze(2)
        val extendedMaskProcessed = (1.0 - extendedMask) * -1e9
        scoresWithMask += extendedMaskProcessed
      }

      val softmaxScores = torch.softmax(scoresWithMask.float, dim = -1).to(xqTransposed.dtype)
      val droppedScores = attnDropout(softmaxScores)
      droppedScores @@ xvTransposed
    }

    val outputReshaped = output.transpose(1, 2).reshape(bsz, seqLen, -1)
    val finalOutput = residDropout(oProj(outputReshaped))
    (finalOutput, pastKv)
  }

  def apply(x: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) =
    forward(x, positionEmbeddings, pastKeyValue, useCache, attentionMask)
}
