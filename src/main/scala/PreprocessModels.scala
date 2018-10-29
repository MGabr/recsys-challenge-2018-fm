import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, concat_ws, explode, explode_outer}

object PreprocessModels extends App {

  override def main(args: Array[String]): Unit = {

    val hdfs_folder = "hdfs:///user/app/2018S/users/e01429253/"
    val names_max_num = 30000


    // spark settings
    val spark = SparkSession.builder.appName("PreprocessModels")
      .config("spark.kryoserializer.buffer.max", "2047m")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.rpc.message.maxSize", "600")
      .getOrCreate()
    val sc = spark.sparkContext


    // load tracks
    val df = spark.read
      .option("multiLine", true)
      .json("hdfs:/user/app/2018S/public/recsys_spotify_2018/mpd.v1/mpd.slice.*.json")

    val playlists = df.select(explode(col("playlists")).as("playlist"))

    var pos_tracks = playlists
      .select("playlist.pid", "playlist.name", "playlist.tracks")
      .select(col("pid"), col("name"), explode_outer(col("tracks")).as("track"))
      .select(
        col("pid"),
        concat_ws(" ", col("name"), col("track.track_name"), col("track.artist_name"), col("track.album_name")).as("names"),
        col("track.track_uri"),
        col("track.album_uri"),
        col("track.artist_uri"))


    // build names count vectorizer model
    val tokenizer = new RegexTokenizer()
      .setInputCol("names")
      .setOutputCol("name_tokens")

    val token_pos_tracks = tokenizer.transform(pos_tracks)

    val names_vectorizer = new CountVectorizer()
      .setInputCol("name_tokens")
      .setOutputCol("names_vec")
      .setBinary(true)
      .setVocabSize(names_max_num)
    val names_vectorizer_model = names_vectorizer.fit(token_pos_tracks)
    names_vectorizer_model.write.overwrite().save(hdfs_folder + "names_vectorizer_model")


    // build track uri indexer model
    val track_uri_indexer = new StringIndexer()
      .setInputCol("track_uri")
      .setOutputCol("track_uri_id")
      .setHandleInvalid("keep")
    val track_uri_indexer_model = track_uri_indexer.fit(pos_tracks)
    track_uri_indexer_model.write.overwrite().save(hdfs_folder + "track_uri_indexer_model")


    // build album uri indexer model
    val album_uri_indexer = new StringIndexer()
      .setInputCol("album_uri")
      .setOutputCol("album_uri_id")
      .setHandleInvalid("keep")
    val album_uri_indexer_model = album_uri_indexer.fit(pos_tracks)
    album_uri_indexer_model.write.overwrite().save(hdfs_folder + "album_uri_indexer_model")


    // build artist uri indexer model
    val artist_uri_indexer = new StringIndexer()
      .setInputCol("artist_uri")
      .setOutputCol("artist_uri_id")
      .setHandleInvalid("keep")
    val artist_uri_indexer_model = artist_uri_indexer.fit(pos_tracks)
    artist_uri_indexer_model.write.overwrite().save(hdfs_folder + "artist_uri_indexer_model")
  }
}
