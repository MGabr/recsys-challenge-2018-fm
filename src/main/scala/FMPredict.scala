import org.apache.spark.ml.feature.{StringIndexerModel, LabeledPoint => _}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD
import org.apache.spark.mllib.regression._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable

object FMPredict extends App {

  override def main(args: Array[String]): Unit = {

    val hdfs_folder = "hdfs:///user/app/2018S/users/e01326657/"

    val trackUriRange = 2262292  // 1 too high because of error in features files
    val names_max_num = 30000

    val numFeatures = trackUriRange + 1 + names_max_num


    // spark settings
    val spark = SparkSession.builder.appName("FMPredict")
      .config("spark.kryoserializer.buffer.max", "2047m")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.rpc.message.maxSize", "600")
      .getOrCreate()
    val sc = spark.sparkContext


    // load instances
    val challenge_features = spark.read.load(hdfs_folder + "challenge")
      .select("pid", "feature_vec")


    // load required models
    val fm = FMModel.load(sc, hdfs_folder + "model")

    val trackUriIndexer = StringIndexerModel.load(hdfs_folder + "track_uri_indexer_model")


    // make and save predictions as csv - still have to add first line
    def top_500(row: Row): (Long, Seq[Int]) = {

      // per playlist values
      val pid = row.getLong(0)
      val feature_indices = mutable.TreeSet(row.getAs[SparseVector](1).indices.distinct: _*)
      val values = Array.fill(feature_indices.size + 1)(1.0)

      var probabilities = Seq[(Double, Int)]()
      var worst_500_probability = 10.0
      // predict for top 100k tracks only instead of all in trackUriRange
      for (track_to_predict_uri <- 0 until 100000) {

        // only predict for tracks not yet in playlist
        if (!feature_indices.contains(track_to_predict_uri)) {

          // add and remove from mutable TreeSet for better performance
          feature_indices += track_to_predict_uri
          val feature_vector = new mllib.linalg.SparseVector(numFeatures, feature_indices.toArray, values)
          val probability = fm.predict(feature_vector)
          feature_indices -= track_to_predict_uri

          // don't add track if it is worse than the worst of the first 500
          if (probabilities.length < 500 && probability < worst_500_probability) {
              worst_500_probability = probability
          }
          if (probability >= worst_500_probability) {
            probabilities = probabilities :+ (probability, track_to_predict_uri)
          }
        }
      }
      (pid, probabilities.sortBy(t => -t._1).take(500).map(_._2))
    }

    challenge_features
      .rdd
      .map(top_500)
      .map(t => t._1.toInt + "," + t._2.map(track_uri_id => trackUriIndexer.labels(track_uri_id)).mkString(","))
      .repartition(1000)
      .saveAsTextFile(hdfs_folder + "submission.csv")
  }
}
