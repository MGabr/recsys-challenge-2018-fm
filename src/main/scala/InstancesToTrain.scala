import org.apache.spark.ml.feature.{LabeledPoint => _, _}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

object InstancesToTrain extends App {

  override def main(args: Array[String]): Unit = {

    val hdfs_folder = "hdfs:///user/app/2018S/users/e01326657/"

    val pid_range = 999999
    val track_uri_range = 2262292
    val names_max_num = 30000
    val name_range = 29999


    // spark settings
    val spark = SparkSession.builder.appName("InstancesToTrain")
      .config("spark.kryoserializer.buffer.max", "2047m")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.rpc.message.maxSize", "600")
      .config("spark.sql.broadcastTimeout", "3000")
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
      .select(col("pid").cast("integer"), lower(col("name")).as("pname"), explode_outer(col("tracks")).as("track"))
      .select("pid", "pname", "track.track_uri", "track.artist_uri")


    // map track uri id strings to ids
    pos_tracks = track_uri_indexer_model
      .transform(pos_tracks)
      .withColumn("track_uri_id", col("track_uri_id").cast("integer"))


    // create base dataframe for negative examples
    val pos_tracks_count = 66346428
    val neg_tracks_try_count = pos_tracks_count

    val schema = StructType(List(StructField("target", IntegerType)))
    val rdd = sc.range(0, neg_tracks_try_count).map(_ => Row(0))

    var neg_tracks = spark.createDataFrame(rdd, schema)


    // create random (pid, track) examples
    neg_tracks = neg_tracks
      .withColumn("pid", round(rand(1) * pid_range).cast("integer"))
      .withColumn("track_uri_id", round(rand(2) * track_uri_range).cast("integer"))


    // filter out bad examples:

    // remove (playlist, track) combinations already in dataset (positive instances)
    neg_tracks = neg_tracks
      .join(pos_tracks, List("track_uri_id", "pid"), "left_anti")

    // remove (playlist name, track) combinations already in dataset (positive instances)
    neg_tracks = neg_tracks
      .join(pos_tracks.select("pid", "pname").distinct(), List("pid"))
      .join(pos_tracks, List("track_uri_id", "pname"), "left_anti")

    // remove (playlist, artist) combinations already in the dataset (positive instances)
    neg_tracks = neg_tracks
      .join(pos_tracks.select("track_uri_id", "artist_uri").distinct(), List("track_uri_id"))
      .join(pos_tracks, List("artist_uri", "pid"), "left_anti")


    // combine positive and negative tracks
    neg_tracks = neg_tracks
      .groupBy("pid")
      .agg(first("target").as("target"), first("pname").as("names"), collect_set("track_uri_id").as("track_uri_ids"))
      .select("target", "pid", "track_uri_ids", "names")

    pos_tracks = pos_tracks
      .groupBy("pid")
      .agg(first("pname").as("names"), collect_set("track_uri_id").as("track_uri_ids"))
      .withColumn("target", lit(1))
      .select("target", "pid", "track_uri_ids", "names")

    var instances = pos_tracks.union(neg_tracks)


    // convert track uri ids into vector - encode track uri ids as 1s in track uri id features
    def indicesToSparseVector(indices: Seq[Int]): SparseVector = {
      if (indices == null) {
        new SparseVector(track_uri_range + 1, Array(), Array())
      } else {
        // some challenge set playlists contain same track multiple times, therefore remove duplicates
        val sorted_indices = indices.sorted.toArray

        new SparseVector(
          track_uri_range + 1,
          sorted_indices,
          Array.fill(sorted_indices.length)(1.0))
      }
    }
    val indicesToSparseVectorUDF = udf((indices: Seq[Int]) => indicesToSparseVector(indices))

    instances = instances.select(
      col("target"),
      col("names"),
      indicesToSparseVectorUDF(col("track_uri_ids")).as("track_uri_vec"))


    // convert names into vector - tokenize, remove stop words and encode names as 1s in name features
    instances = instances.na.fill("")

    val tokenizer = new RegexTokenizer()
      .setInputCol("names")
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
    instances = instances.select("target", "feature_vec")

    var Array(training_instances, test_instances) = instances.randomSplit(Array(0.99, 0.01), 1)


    // save instances as libSVM
    def to_labeled_point(row: Row): LabeledPoint = {
      LabeledPoint(
        row.getInt(0).toDouble,
        mllib.linalg.SparseVector.fromML(row.getAs[SparseVector](1)))
    }

    val test_labeled_points = test_instances.rdd.map(to_labeled_point)
    MLUtils.saveAsLibSVMFile(test_labeled_points, hdfs_folder + "test")

    val training_labeled_points = training_instances.rdd.map(to_labeled_point)
    MLUtils.saveAsLibSVMFile(training_labeled_points, hdfs_folder + "training")
  }
}
