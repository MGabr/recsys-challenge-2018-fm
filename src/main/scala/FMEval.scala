import org.apache.spark.ml.feature.{StringIndexerModel, LabeledPoint => _}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable

object FMEval extends App {

  override def main(args: Array[String]): Unit = {

    val hdfs_folder = "hdfs:///user/app/2018S/users/e01326657/"

    val trackUriRange = 2262292  // 1 too high because of error in features files
    val names_max_num = 30000

    val numFeatures = trackUriRange + 1 + names_max_num


    // spark settings
    val spark = SparkSession.builder.appName("FMEval")
      .config("spark.kryoserializer.buffer.max", "2047m")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.rpc.message.maxSize", "600")
      .getOrCreate()
    val sc = spark.sparkContext


    // load instances
    val numPartitions = 100
    val testing = MLUtils.loadLibSVMFile(sc, hdfs_folder + "test", numFeatures, numPartitions)


    // load required models
    val fm = FMModel.load(sc, hdfs_folder + "model")


    // evaluate rankings
    val predictionAndLabels: RDD[(Double, Double)] = fm
      .predict(testing.map(_.features))
      .map(x => if (x >= 0.5) 1.0 else 0.0)
      .zip(testing.map(_.label.toDouble))

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // 20lf 20 iters: 0.5567634899096736
    // 20lf 60 iters: 0.5907228928767718
    // 20lf 200 iters: 0.6093563066695667
    // 30lf 20 iters: 0.5023011916765822
    val auPRC = metrics.areaUnderPR
    println(s"Area under precision-recall curve = ${auPRC}")

    // 20lf 20 iters: 0.5657794193487357
    // 20lf 60 iters: 0.6145827959301915
    // 20lf 200 iters: 0.6413915855839916
    // 30lf 20 iters: 0.49432997975489046
    val auROC = metrics.areaUnderROC
    println(s"Area under ROC = ${auROC}")
  }
}
