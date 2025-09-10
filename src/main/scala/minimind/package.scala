import torch.{---, ::, DType, Default, Float32, FloatNN, Tensor, nn}
import torch.nn.modules.{HasParams, TensorModule}

import scala.math.pow

package object minimind {

  // 预计算旋转位置编码频率
  def precomputeFreqsCis[ParamType <: FloatNN : Default](dim: Int, end: Int = 32 * 1024, theta: Double = 1e6): (Tensor[Float32], Tensor[Float32]) = {
    val tensorArrage = torch.arange(0, dim, 2)(0, dim / 2).float.toArray // dim
    val powArray = tensorArrage.map(num => pow(theta, num / dim))
    val freqs = torch.tensor(powArray.map(num => 1.0 /num).toSeq)//, dtype = DType.float32)
//    val freqs = (1.0 / (theta.pow(torch.arange[ParamType](0, dim, 2).slice(0, dim / 2) / dim.toParamType[ParamType]))).toType[ParamType]
    val t = torch.arange(end = end)
    val outerFreqs = torch.outer(t, freqs)
    val freqsCos = torch.cat(Seq(torch.cos(outerFreqs), torch.cos(outerFreqs)), dim = -1)
    val freqsSin = torch.cat(Seq(torch.sin(outerFreqs), torch.sin(outerFreqs)), dim = -1)
    (freqsCos.to(DType.float32), freqsSin.to(DType.float32))
  }

  // 应用旋转位置编码
  def applyRotaryPosEmb[ParamType <: FloatNN : Default](q: Tensor[ParamType], k: Tensor[ParamType], cos: Tensor[ParamType], sin: Tensor[ParamType], positionIds: Option[Tensor[ParamType]] = None, unsqueezeDim: Int = 1): (Tensor[ParamType], Tensor[ParamType]) = {
    def rotateHalf(x: Tensor[ParamType]): Tensor[ParamType] = {
      val halfSize = x.shape.last / 2
      torch.cat(Seq(-x(---,halfSize.::), x(---, 0.::(halfSize))), dim = -1)
    }

    val qEmbed = (q * cos.unsqueeze(unsqueezeDim)) + (rotateHalf(q) * sin.unsqueeze(unsqueezeDim))
    val kEmbed = (k * cos.unsqueeze(unsqueezeDim)) + (rotateHalf(k) * sin.unsqueeze(unsqueezeDim))
    (qEmbed, kEmbed)
  }

  // 重复键值对
  def repeatKv[ParamType <: FloatNN : Default](x: Tensor[ParamType], nRep: Int): Tensor[ParamType] = {
//    val (bs, slen, numKeyValueHeads, headDim) :Array[Int]= x.shape
    val  bs = x.shape(0)
    val slen = x.shape(1)
    
    val numQueryHeads = x.shape(2)
    val numKeyValueHeads = x.shape(2)
    val headDim = x.shape(3)
    if (nRep == 1) x
    else {
      val res =x(---, ---, --- ,0.::(numKeyValueHeads)).unsqueeze(3).expand(bs, slen, numKeyValueHeads, nRep, headDim).reshape(bs, slen, numKeyValueHeads * nRep, headDim)
      res.to(x.dtype)
    }
  }
}
