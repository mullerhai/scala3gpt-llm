import llm.{GPT, GPTConfig}
import torch.{DType, Float32}
import torch.Device.{CPU, CUDA}
import torch.*

@main
def main(): Unit =
  println("Hello Scala3GPT !")
  val config = GPTConfig()
  val device = if torch.cuda.isAvailable then CUDA else CPU
  val model = GPT[Float32](config).to(device)
  val mockData: Tensor[Float32] = torch.ones(Array(12, 512)).to(device)
  val mockTargetData: Tensor[Float32] = torch.torch_normal(0.2d, 0.3d, Array(12, 1)).to(device)
  val (logit, loss) = model(mockData, Option(mockTargetData))

  println(s"view logit ${logit.item}")
  println(s"view loss ${loss.get.item}")
  println(s"finish ...")

