import org.apache.spark.ml.feature.{LabeledPoint => _, _}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

object PosInstancesToTrain extends App {

  override def main(args: Array[String]): Unit = {

    val hdfs_folder = "hdfs:///user/app/2018S/users/e01326657/"

    val pid_range = 999999
    val track_uri_range = 2262292
    val names_max_num = 30000
    val name_range = 29999


    // spark settings
    val spark = SparkSession.builder.appName("PosInstancesToTrain")
      .config("spark.kryoserializer.buffer.max", "2047m")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.rpc.message.maxSize", "600")
      .getOrCreate()
    val sc = spark.sparkContext


    // load required models
    val name_vectorizer_model = CountVectorizerModel.load(hdfs_folder + "names_vectorizer_model")
    val track_uri_indexer_model = StringIndexerModel.load(hdfs_folder + "track_uri_indexer_model")


    // load mpd
    val df = spark.read
      .option("multiLine", true)
      .json("hdfs:/user/app/2018S/public/recsys_spotify_2018/mpd.v1/mpd.slice.*.json")

    val playlists = df.select(explode(col("playlists")).as("playlist"))

    var pos_tracks = playlists
      .select("playlist.pid", "playlist.name", "playlist.tracks")
      .select(col("pid"), col("name"), explode_outer(col("tracks")).as("track"))
      .select(col("pid"), col("name"), col("track.track_uri"))


    // map track uri id strings to ids
    pos_tracks = track_uri_indexer_model
      .transform(pos_tracks)
      .withColumn("track_uri_id", col("track_uri_id").cast("integer"))


    // combine positive and negative tracks
    var instances = pos_tracks
      .select(col("pid"), col("name"), col("track_uri_id"))
      .groupBy("pid", "name").agg(collect_list("track_uri_id").as("track_uri_ids"))
      .select("track_uri_ids", "name")


    // convert track uri ids into vector - encode track uri ids as 1s in track uri id features
    def indicesToSparseVector(indices: Seq[Int]): SparseVector = {
      if (indices == null) {
        new SparseVector(track_uri_range + 1, Array(), Array())
      } else {
        // some challenge set playlists contain same track multiple times, therefore remove duplicates
        val sorted_indices = indices.sorted.distinct.toArray

        new SparseVector(
          track_uri_range + 1,
          sorted_indices,
          Array.fill(sorted_indices.length)(1.0))
      }
    }
    val indicesToSparseVectorUDF = udf((indices: Seq[Int]) => indicesToSparseVector(indices))

    instances = instances.select(
      col("name"),
      indicesToSparseVectorUDF(col("track_uri_ids")).as("track_uri_vec"))


    // convert names into vector - tokenize, remove stop words and encode names as 1s in name features
    instances = instances.na.fill("")

    val tokenizer = new RegexTokenizer()
      .setInputCol("name")
      .setOutputCol("_name_tokens")
    instances = tokenizer.transform(instances)

    val sw_remover = new StopWordsRemover()
      .setInputCol("_name_tokens")
      .setOutputCol("name_tokens")
    instances = sw_remover.transform(instances)

    instances = name_vectorizer_model.transform(instances)


    // combine vectors
    val track_uri_sizeHint = new VectorSizeHint()
      .setInputCol("track_uri_vec")
      .setSize(track_uri_range + 1)
    instances = track_uri_sizeHint.transform(instances)

    val names_sizeHint = new VectorSizeHint()
      .setInputCol("names_vec")
      .setSize(name_range + 1)
    instances = names_sizeHint.transform(instances)

    val assembler = new VectorAssembler()
      .setInputCols(Array("track_uri_vec", "names_vec"))
      .setOutputCol("feature_vec")
    instances = assembler.transform(instances)


    // split into training and test instances
    instances = instances.select(col("feature_vec"))

    var Array(training_instances, test_instances) = instances.randomSplit(Array(0.99, 0.01), 1)


    // save instances as libSVM
    def to_labeled_point(row: Row): LabeledPoint = {
      LabeledPoint(
        1.0,
        mllib.linalg.SparseVector.fromML(row.getAs[SparseVector](0)))
    }

    val test_labeled_points = test_instances.rdd.map(to_labeled_point)
    MLUtils.saveAsLibSVMFile(test_labeled_points, hdfs_folder + "test")

    val training_labeled_points = training_instances.rdd.map(to_labeled_point)
    MLUtils.saveAsLibSVMFile(training_labeled_points, hdfs_folder + "training")
  }
}
