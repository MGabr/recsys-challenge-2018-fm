import org.apache.spark.ml.feature.{LabeledPoint => _, _}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

object ChallengeFeatures extends App {

  override def main(args: Array[String]): Unit = {

    val hdfs_folder = "hdfs:///user/app/2018S/users/e01326657/"

    val track_uri_range = 2262292  // 1 too high
    val names_max_num = 30000
    val name_range = 29999


    // spark settings
    val spark = SparkSession.builder.appName("ChallengeFeatures")
      .config("spark.kryoserializer.buffer.max", "2047m")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.rpc.message.maxSize", "600")
      .getOrCreate()
    val sc = spark.sparkContext


    // load required models
    val name_vectorizer_model = CountVectorizerModel.load(hdfs_folder + "names_vectorizer_model")
    val track_uri_indexer_model = StringIndexerModel.load(hdfs_folder + "track_uri_indexer_model")


    // load challenge tracks
    val df = spark.read
      .option("multiLine", true)
      .json("hdfs:/user/app/2018S/public/recsys_spotify_2018/challenge.v1/challenge_set.json")

    val playlists = df.select(explode(col("playlists")).as("playlist"))

    var challenge_tracks = playlists
      .select("playlist.pid", "playlist.name", "playlist.tracks")
      .select(col("pid"), col("name"), explode_outer(col("tracks")).as("track"))
      .select("pid", "name", "track.track_uri")
      .repartition(1000, col("pid"))  // important for performance


    // map track uri id strings to ids
    challenge_tracks = track_uri_indexer_model
      .transform(challenge_tracks)
      .withColumn("track_uri_id", col("track_uri_id").cast("integer"))


    // add names and tracks of playlist - we then have one instance per (playlist, track to predict) combination
    var challenge_features = challenge_tracks
      .select("pid", "name", "track_uri_id")
      .groupBy("pid", "name").agg(collect_list("track_uri_id").as("track_uri_ids"))


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

    challenge_features = challenge_features.select(
      col("pid"),
      col("name"),
      indicesToSparseVectorUDF(col("track_uri_ids")).as("track_uri_vec"))


    // convert names into vector - tokenize, remove stop words and encode names as 1s in name features
    challenge_features = challenge_features.na.fill("")

    val tokenizer = new RegexTokenizer()
      .setInputCol("name")
      .setOutputCol("_name_tokens")
    challenge_features = tokenizer.transform(challenge_features)

    val sw_remover = new StopWordsRemover()
      .setInputCol("_name_tokens")
      .setOutputCol("name_tokens")
    challenge_features = sw_remover.transform(challenge_features)

    challenge_features = name_vectorizer_model.transform(challenge_features)


    // combine vectors
    val track_uri_sizeHint = new VectorSizeHint()
      .setInputCol("track_uri_vec")
      .setSize(track_uri_range + 1)
    challenge_features = track_uri_sizeHint.transform(challenge_features)

    val names_sizeHint = new VectorSizeHint()
      .setInputCol("names_vec")
      .setSize(name_range + 1)
    challenge_features = names_sizeHint.transform(challenge_features)

    val assembler = new VectorAssembler()
      .setInputCols(Array("track_uri_vec", "names_vec"))
      .setOutputCol("feature_vec")
    challenge_features = assembler.transform(challenge_features)


    // save instances as dataframe
    challenge_features
      .select(col("pid"), col("feature_vec"))
      .write.save(hdfs_folder + "challenge")
  }
}
