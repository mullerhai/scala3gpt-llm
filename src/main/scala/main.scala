import llm.{GPT, GPTConfig}
import torch.{DType, Float32}
import torch.Device.{CPU, CUDA}
import torch.*
import scala.reflect.ClassTag  // 新增：导入ClassTag
def testNumpy2(): Unit =
  // 修复：为 Float 类型提供 ClassTag（Float32 对应 Scala 的 Float 类型）
  //  implicit val floatClassTag: ClassTag[Float] = ClassTag.Float
  val a: Tensor[Float32] = torch.normal[Float32](0.0f, 1.0f, Array(12, 512))
  println(a.numpy[Float32]().printArray())

//def testNumpy(): Unit =
//  implicit val float32ClassTag: ClassTag[torch.DTypeToScala[Float32]] = ClassTag(classOf[Float])
//  val a = torch.ones[Float32](Array(12, 512))
//  println(a.numpy[Float]())

@main
def main(): Unit =
  println("Hello Scala3GPT !")
  testNumpy2()
  val config = GPTConfig()
  val device = if torch.cuda.isAvailable then CUDA else CPU
  val model = GPT[Float32](config).to(device)
  val mockData: Tensor[Float32] = torch.ones(Array(12, 512)).to(device)
  val mockTargetData: Tensor[Float32] = torch.torch_normal(0.2d, 0.3d, Array(12, 1)).to(device)
  val (logit, loss) = model(mockData, Option(mockTargetData))

  println(s"view logit ${logit.item}")
  println(s"view loss ${loss.get.item}")
  println(s"finish ...")

