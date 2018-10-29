import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object FMPersist extends App {

  override def main(args: Array[String]): Unit = {

    val hdfs_folder = "hdfs:///user/app/2018S/users/e01326657/"

    val track_uri_range = 2262292
    val names_max_num = 30000
    val name_range = 29999

    val numFeatures = track_uri_range + 1 + names_max_num

    // spark settings
    val spark = SparkSession.builder.appName("FMPersist")
      .config("spark.kryoserializer.buffer.max", "2047m")
      .config("spark.driver.maxResultSize", "0")
      .config("spark.rpc.message.maxSize", "600")

      .config("spark.yarn.executor.memoryOverhead", "8192")
      .config("spark.yarn.driver.memoryOverhead", "8192")
      .config("spark.default.parallelism", "500")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

      .config("spark.yarn.scheduler.heartbeat.interval-ms", "720000")
      .config("spark.executor.heartbeatInterval", "720")
      .config("spark.network.timeout", "7200")

      .getOrCreate()
    val sc = spark.sparkContext


    // we only want to see the evaluation result printlns and not all the other info logs
    sc.setLogLevel("ERROR")


    // load training and test sets
    val numPartitions = 1000

    val training = MLUtils.loadLibSVMFile(sc, hdfs_folder + "training", numFeatures, numPartitions)
    val testing = MLUtils.loadLibSVMFile(sc, hdfs_folder + "test", numFeatures, numPartitions)


    // with best parameters
    trainAndEvaluate(sc, hdfs_folder, training, testing, numIterations = 200, dim = (true, true, 20))
  }

  def trainAndEvaluate(sc: SparkContext,
                       hdfs_folder: String,
                       training: RDD[LabeledPoint],
                       testing: RDD[LabeledPoint],
                       numIterations: Int = 20,
                       stepSize: Double = 0.01,
                       dim: (Boolean, Boolean, Int) = (true, true, 20),
                       regParam: (Double, Double, Double) = (0.0, 0.0, 0.05),
                       initStd: Double = 0.01): Unit = {
    println(s"Train and evaluate with numIterations = ${numIterations}, stepSize = ${stepSize}, dim = ${dim}, regParam = ${regParam}, initStd = ${initStd}")

    val fm: FMModel = FMWithSGD.train(
      training,
      task = 1,
      numIterations = numIterations,
      stepSize = stepSize,
      dim = dim,
      regParam = regParam,
      initStd = initStd)

    fm.save(sc, hdfs_folder + "model")
  }
}